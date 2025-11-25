#!/usr/bin/env python3
"""
OpenAI/Codex Sub-Agent - Explicit Request Only

LOCAL-FIRST: Only used when user explicitly requests OpenAI/Codex
Not auto-routed - must be manually selected
Graceful degradation: Falls back to local if unavailable

Enhanced with Pydantic structured output support
"""

import os
import json
import sys
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not installed. Run: pip3 install openai")

# Pydantic support for structured outputs
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from pydantic_models import CodeGenerationResult, SecurityAnalysisResult
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

class OpenAIAgent:
    def __init__(self, pydantic_mode=False):
        """
        Initialize OpenAI agent with optional Pydantic structured output support

        Args:
            pydantic_mode: If True, support Pydantic structured outputs
        """
        self.available = False
        self.client = None
        self.pydantic_mode = pydantic_mode and PYDANTIC_AVAILABLE

        if not OPENAI_AVAILABLE:
            return

        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            print("⚠️  OPENAI_API_KEY not set. OpenAI/Codex unavailable (explicit request only).")
            print("   Get key from: https://platform.openai.com/api-keys")
            print("   Set: export OPENAI_API_KEY='your_key'")
            return

        try:
            self.client = OpenAI(api_key=api_key)
            self.available = True
            print("✅ OpenAI connected (explicit request only, not auto-routed)")
        except Exception as e:
            print(f"⚠️  OpenAI initialization failed: {e}")

    def query(self, prompt, model="gpt-4-turbo", fallback_response=None):
        """
        Query OpenAI models

        Args:
            prompt: Text query
            model: Model to use (gpt-4-turbo, gpt-3.5-turbo, etc.)
            fallback_response: Local response to use if OpenAI fails

        Returns:
            Response dict with backend info
        """
        if not self.available:
            return {
                "response": fallback_response or "OpenAI unavailable. Install: pip3 install openai",
                "model": model,
                "backend": "openai",
                "available": False,
                "fallback_used": True
            }

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )

            return {
                "response": response.choices[0].message.content,
                "model": model,
                "backend": "openai",
                "available": True,
                "tokens": response.usage.total_tokens,
                "cost": self.estimate_cost(response.usage.total_tokens, model),
                "privacy": "cloud"
            }

        except Exception as e:
            # Fallback to local if provided
            return {
                "response": fallback_response or f"OpenAI error: {str(e)}",
                "model": model,
                "backend": "openai",
                "error": str(e),
                "fallback_used": bool(fallback_response),
                "privacy": "cloud"
            }

    def query_structured(self, prompt, response_model, model="gpt-4-turbo"):
        """
        Query OpenAI with structured Pydantic output (BETA feature)

        Args:
            prompt: Text query describing what to generate
            response_model: Pydantic model class for structured output
            model: OpenAI model to use

        Returns:
            Pydantic model instance with validated structured output

        Example:
            result = agent.query_structured(
                "Generate a secure password hashing function in Python",
                response_model=CodeGenerationResult,
                model="gpt-4-turbo"
            )
            print(result.code)  # Validated code
            print(result.language)  # Validated language
        """
        if not self.available:
            raise RuntimeError("OpenAI not available")

        if not PYDANTIC_AVAILABLE:
            raise RuntimeError("Pydantic not available. Install: pip install pydantic")

        try:
            # OpenAI structured outputs (requires pydantic_model parameter)
            # Note: This uses OpenAI's beta structured outputs feature
            completion = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates structured outputs."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_model,
            )

            # Extract parsed Pydantic model
            return completion.choices[0].message.parsed

        except Exception as e:
            raise RuntimeError(f"Structured output generation failed: {e}")

    def estimate_cost(self, tokens, model):
        """Estimate API cost"""
        costs = {
            'gpt-4-turbo': 0.020,  # $0.02 per 1K tokens (average)
            'gpt-3.5-turbo': 0.002,  # $0.002 per 1K tokens
            'gpt-4': 0.040  # $0.04 per 1K tokens
        }

        cost_per_1k = costs.get(model, 0.020)
        return (tokens / 1000) * cost_per_1k

    def is_available(self):
        """Check if OpenAI is available"""
        return self.available

# CLI
if __name__ == "__main__":
    import sys

    agent = OpenAIAgent()

    if len(sys.argv) < 2:
        print("OpenAI Sub-Agent - Usage:")
        print("  python3 openai_wrapper.py test")
        print("  python3 openai_wrapper.py query 'your question' [model]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "test":
        print(f"OpenAI Available: {agent.is_available()}")
        if agent.is_available():
            result = agent.query("What is 2+2? Answer in 5 words.", model="gpt-3.5-turbo")
            print(f"Response: {result['response']}")
            print(f"Model: {result['model']}")
            print(f"Cost: ${result['cost']:.4f}")

    elif cmd == "query" and len(sys.argv) > 2:
        query = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4-turbo"
        result = agent.query(query, model=model)
        print(json.dumps(result, indent=2))
