#!/usr/bin/env python3
"""
Gemini Sub-Agent - Multimodal Support ONLY

LOCAL-FIRST: Only used when local can't handle (images, video)
Free tier: 1500 requests/day
Graceful degradation: Falls back to local if unavailable
"""

import os
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip3 install google-generativeai")

class GeminiAgent:
    def __init__(self):
        self.available = False
        self.model = None

        if not GEMINI_AVAILABLE:
            return

        # Check for API key
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            print("⚠️  GOOGLE_API_KEY not set. Gemini unavailable (multimodal fallback only).")
            print("   Get key from: https://ai.google.dev/")
            print("   Set: export GOOGLE_API_KEY='your_key'")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.available = True
            print("✅ Gemini 2.0 Flash connected (multimodal support active)")
        except Exception as e:
            print(f"⚠️  Gemini initialization failed: {e}")

    def query(self, prompt, images=None, video=None, fallback_response=None):
        """
        Query Gemini with multimodal support

        Args:
            prompt: Text query
            images: List of image paths (optional)
            video: Video path (optional)
            fallback_response: Local response to use if Gemini fails

        Returns:
            Response dict with backend info
        """
        if not self.available:
            return {
                "response": fallback_response or "Gemini unavailable. Install: pip3 install google-generativeai",
                "model": "gemini-2.0-flash-exp",
                "backend": "gemini",
                "available": False,
                "fallback_used": True
            }

        try:
            content = [prompt]

            # Add images
            if images:
                for img_path in images:
                    if os.path.exists(img_path):
                        img = genai.upload_file(img_path)
                        content.append(img)
                    else:
                        print(f"⚠️  Image not found: {img_path}")

            # Add video
            if video and os.path.exists(video):
                vid = genai.upload_file(video)
                content.append(vid)

            # Generate response
            response = self.model.generate_content(content)

            return {
                "response": response.text,
                "model": "gemini-2.0-flash-exp",
                "backend": "gemini",
                "available": True,
                "multimodal": bool(images or video),
                "cost": 0,  # Free tier
                "privacy": "cloud"
            }

        except Exception as e:
            # Fallback to local if provided
            return {
                "response": fallback_response or f"Gemini error: {str(e)}",
                "model": "gemini-2.0-flash-exp",
                "backend": "gemini",
                "error": str(e),
                "fallback_used": bool(fallback_response),
                "privacy": "cloud"
            }

    def is_available(self):
        """Check if Gemini is available"""
        return self.available

# CLI
if __name__ == "__main__":
    import sys

    agent = GeminiAgent()

    if len(sys.argv) < 2:
        print("Gemini Sub-Agent - Usage:")
        print("  python3 gemini_wrapper.py test")
        print("  python3 gemini_wrapper.py query 'your question'")
        print("  python3 gemini_wrapper.py image 'describe this' /path/to/image.jpg")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "test":
        print(f"Gemini Available: {agent.is_available()}")
        if agent.is_available():
            result = agent.query("What is 2+2? Answer in 5 words.")
            print(f"Response: {result['response']}")
            print(f"Model: {result['model']}")

    elif cmd == "query" and len(sys.argv) > 2:
        query = sys.argv[2]
        result = agent.query(query)
        print(json.dumps(result, indent=2))

    elif cmd == "image" and len(sys.argv) > 3:
        query = sys.argv[2]
        images = sys.argv[3:]
        result = agent.query(query, images=images)
        print(json.dumps(result, indent=2))
