import os
import json
import re
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# تحميل المتغيرات من ملف .env (مثل HF_TOKEN)
# هذا يتيح لنا إخفاء المفاتيح السرية في ملف منفصل
# بدلاً من كتابتها مباشرة في الكود
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# إنشاء تطبيق Flask — هو الإطار الذي يشغّل الخادم
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# تفعيل CORS — يسمح للمتصفح بإرسال طلبات من
# موقع مختلف (frontend على دومين آخر) إلى هذا الخادم
# بدون هذا السطر يرفض المتصفح الطلبات تلقائياً
# ─────────────────────────────────────────────
CORS(app)


# ─────────────────────────────────────────────
# الصفحة الرئيسية: GET /
# تعرض ملف index.html من مجلد templates
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# المسار الوحيد: POST /analyze
# يستقبل نصاً، يرسله إلى نموذج الذكاء الاصطناعي،
# ويعيد تحليلاً منظماً على شكل JSON
# ─────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():

    # ── قراءة المفتاح السري من البيئة ──────────
    # HF_TOKEN هو مفتاح Hugging Face الذي يأذن لنا
    # باستخدام نموذج الذكاء الاصطناعي
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return jsonify({"error": "HF_TOKEN غير موجود في متغيرات البيئة"}), 500

    # ── قراءة الجسم (body) القادم من المتصفح ───
    # نتوقع JSON بالشكل: { "text": "..." }
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "الطلب يجب أن يكون JSON صحيحاً"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "حقل text لا يمكن أن يكون فارغاً"}), 400

    # ── بناء الـ prompt الذي سنرسله للنموذج ────
    # نطلب من النموذج تحليل النص وإعادة JSON محدد الشكل
    prompt = f"""Analyze the following opportunity text and respond with ONLY a valid JSON object.
No explanation, no markdown, no code fences — just raw JSON.

IMPORTANT: You must write ALL values entirely in Arabic, regardless of the language the input text is written in. The summary, category, and every tag must be in Arabic.

The JSON must have exactly these three keys:
- "summary": a short one-sentence summary of the opportunity, written in Arabic (string)
- "category": the category this opportunity belongs to, written in Arabic — e.g. "تقنية", "صحة", "تمويل", "تعليم" (string)
- "tags": an array of 3 to 5 relevant keyword tags, all written in Arabic (array of strings)

Opportunity text:
\"\"\"{text}\"\"\"

Respond with raw JSON only (all values in Arabic):"""

    # ── إرسال الطلب إلى Hugging Face Router API ─
    # نستخدم نموذج Llama 3.1 عبر واجهة برمجية متوافقة مع OpenAI
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.2,
    }

    # ── استدعاء API والتعامل مع أخطاء الشبكة ───
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return jsonify({"error": "انتهت مهلة الاتصال بـ Hugging Face API"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"خطأ في الاتصال بـ API: {str(e)}"}), 502

    # ── استخراج النص من رد النموذج ──────────────
    # الرد يأتي بالشكل: {"choices": [{"message": {"content": "..."}}]}
    try:
        ai_text = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError):
        return jsonify({"error": "رد غير متوقع من Hugging Face API"}), 502

    # ── تنظيف الرد من أسوار الـ Markdown إن وُجدت ─
    # أحياناً يُحيط النموذج JSON بـ ```json ... ```
    # هذا الكود يحذف تلك الأسوار ليبقى JSON نظيفاً
    ai_text = re.sub(r"^```(?:json)?\s*", "", ai_text)
    ai_text = re.sub(r"\s*```$", "", ai_text)
    ai_text = ai_text.strip()

    # ── تحويل النص إلى كائن Python والتحقق من مفاتيحه ─
    try:
        result = json.loads(ai_text)
    except json.JSONDecodeError:
        return jsonify({
            "error": "النموذج لم يُعِد JSON صالحاً",
            "raw": ai_text,
        }), 502

    # ── التحقق من أن المفاتيح الثلاثة موجودة ────
    required_keys = {"summary", "category", "tags"}
    missing = required_keys - result.keys()
    if missing:
        return jsonify({
            "error": f"المفاتيح التالية مفقودة في رد النموذج: {missing}",
            "raw": result,
        }), 502

    # ── إرجاع النتيجة للمتصفح ───────────────────
    return jsonify(result), 200


# ─────────────────────────────────────────────
# نقطة البداية: تشغيل الخادم عند تشغيل الملف مباشرة
# PORT تأتي من البيئة؛ إن لم تُحدَّد نستخدم 8080
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
