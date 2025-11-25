"""
DSMIL Unified Platform - Sub-Agent Framework

LOCAL-FIRST Philosophy:
- Default to local DeepSeek (no guardrails, private, free)
- Cloud backends OPTIONAL (only when needed)
- Gemini: Multimodal only (images/video)
- Codex/OpenAI: Explicit request only

All cloud backends gracefully degrade to local if unavailable.
"""

__version__ = "1.0.0"
__all__ = ['GeminiAgent', 'OpenAIAgent']
