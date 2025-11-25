#!/usr/bin/env python3
"""
DSMIL Unified AI Orchestrator - LOCAL-FIRST Architecture

Philosophy:
1. DEFAULT: Local DeepSeek (privacy, no guardrails, zero cost, DSMIL-attested)
2. Gemini: ONLY for multimodal (images/video local can't handle)
3. OpenAI: ONLY when explicitly requested by user
4. All cloud backends OPTIONAL - graceful degradation to local

Routing Priority:
  Multimodal query → Try Gemini → Fallback to local
  Explicit request → Use requested backend → Fallback to local
  Everything else → Local DeepSeek (default)
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsmil_ai_engine import DSMILAIEngine
from sub_agents.gemini_wrapper import GeminiAgent
from sub_agents.openai_wrapper import OpenAIAgent
from smart_router import SmartRouter
from web_search import WebSearch

class UnifiedAIOrchestrator:
    def __init__(self):
        # Primary: Local AI (always available)
        self.local = DSMILAIEngine()

        # Smart Router (NEW - automatic query routing)
        self.router = SmartRouter()

        # Web Search (NEW - for current information)
        self.web = WebSearch()

        # Optional: Cloud backends (graceful degradation)
        self.gemini = GeminiAgent()
        self.openai = OpenAIAgent()

        print("🎯 DSMIL Unified Orchestrator - LOCAL-FIRST + SMART ROUTING")
        print(f"   Local AI: ✅ DeepSeek R1 + DeepSeek Coder + Qwen Coder")
        print(f"   Smart Router: ✅ Auto-detects code/general/complex queries")
        print(f"   Gemini Pro: {'✅ Available' if self.gemini.is_available() else '⚠️  Not configured'} (multimodal only)")
        print(f"   OpenAI Pro: {'✅ Available' if self.openai.is_available() else '⚠️  Not configured'} (explicit request only)")

    def query(self, prompt, force_backend=None, images=None, video=None, **kwargs):
        """
        Unified query interface with LOCAL-FIRST routing

        Args:
            prompt: User query
            force_backend: "local", "gemini", "openai" (explicit override)
            images: List of image paths (triggers Gemini)
            video: Video path (triggers Gemini)
            **kwargs: Additional args (model preference, etc.)

        Returns:
            Unified response with backend info and routing decision
        """
        start_time = time.time()

        # Use Smart Router to determine best model
        routing_decision = self.router.route(
            prompt,
            has_images=bool(images or video),
            user_preference=force_backend
        )

        # Extract routing info
        selected_model = routing_decision['model']
        routing_reason = routing_decision['reason']
        routing_explanation = routing_decision['explanation']
        needs_web_search = routing_decision.get('web_search', False)

        # Determine backend (cloud vs local)
        if selected_model == "gemini-pro":
            backend = "gemini"
        elif selected_model in ["gpt-4-turbo", "gpt-3.5-turbo"]:
            backend = "openai"
        else:
            backend = "local"  # All coding and general models are local

        # Execute on selected backend
        result = {}

        if backend == "local":
            # Check if web search is needed
            if needs_web_search and self.web.ddgs_available:
                # Perform web search first
                search_results = self.web.search(prompt, max_results=5)

                if 'error' not in search_results:
                    # Enhance prompt with web results
                    web_context = self._format_web_results(search_results)
                    enhanced_prompt = f"{web_context}\n\nOriginal query: {prompt}\n\nProvide answer based on search results above."

                    # Generate response with web context
                    result = self.local.generate(enhanced_prompt, model_selection=selected_model)
                    result['web_search'] = {
                        'performed': True,
                        'source': search_results['source'],
                        'result_count': search_results['count'],
                        'urls': [r['url'] for r in search_results['results']]
                    }
                else:
                    # Web search failed, continue without it
                    result = self.local.generate(prompt, model_selection=selected_model)
                    result['web_search'] = {'performed': False, 'error': search_results.get('error')}
            else:
                # No web search needed or unavailable
                result = self.local.generate(prompt, model_selection=selected_model)
                result['web_search'] = {'performed': False}

            # Add metadata
            result['backend'] = 'local'
            result['cost'] = 0.0
            result['privacy'] = 'local'
            result['dsmil_attested'] = True
            result['routing'] = {
                'selected_model': selected_model,
                'reason': routing_reason,
                'explanation': routing_explanation,
                'emoji_tag': self.router.explain_routing(routing_decision, format='emoji')
            }

        elif backend == "gemini":
            # Try Gemini for multimodal
            local_response = None
            if not images and not video:
                # Get local response as fallback for non-multimodal
                local_result = self.local.generate(prompt)
                local_response = local_result.get('response')

            result = self.gemini.query(prompt, images=images, video=video, fallback_response=local_response)
            result['dsmil_attested'] = False

        elif backend == "openai":
            # Try OpenAI (explicit request only)
            local_result = self.local.generate(prompt)
            local_response = local_result.get('response')

            result = self.openai.query(prompt, model=kwargs.get('model', 'gpt-4-turbo'), fallback_response=local_response)
            result['dsmil_attested'] = False

        else:
            # Unknown backend → local
            result = self.local.generate(prompt)
            result['backend'] = 'local_deepseek'
            result['cost'] = 0.0
            result['privacy'] = 'local'
            result['dsmil_attested'] = True
            routing_reason = "unknown_backend_fallback"

        # Add enhanced routing metadata
        if 'routing' not in result:
            result['routing'] = {
                'selected_model': selected_model,
                'reason': routing_reason,
                'explanation': routing_explanation
            }

        result['routed_to'] = backend
        result['total_time'] = round(time.time() - start_time, 2)
        result['timestamp'] = time.time()

        return result

    def _format_web_results(self, search_results):
        """Format web search results for AI context"""
        context = f"Web search results for: {search_results['query']}\n\n"

        for i, result in enumerate(search_results['results'], 1):
            context += f"[{i}] {result['title']}\n"
            context += f"    {result['snippet']}\n"
            context += f"    URL: {result['url']}\n\n"

        return context

    def get_status(self):
        """Get comprehensive status of all backends"""
        local_status = self.local.get_status()

        return {
            "backends": {
                "local_deepseek": {
                    "available": True,  # Always available
                    "priority": "PRIMARY (default for all queries)",
                    "models": local_status.get('models', {}),
                    "dsmil_attested": True,
                    "cost_per_query": 0,
                    "privacy": "local"
                },
                "gemini_pro": {
                    "available": self.gemini.is_available(),
                    "priority": "MULTIMODAL ONLY (images/video)",
                    "model": "gemini-2.0-flash-exp" if self.gemini.is_available() else "not_configured",
                    "student_edition": True,
                    "dsmil_attested": False,
                    "cost_per_query": 0,  # Student free tier
                    "privacy": "cloud"
                },
                "openai_pro": {
                    "available": self.openai.is_available(),
                    "priority": "EXPLICIT REQUEST ONLY (not auto-routed)",
                    "models": ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"],
                    "dsmil_attested": False,
                    "cost_per_1k_tokens": "$0.02",
                    "privacy": "cloud"
                }
            },
            "routing_philosophy": "LOCAL-FIRST",
            "default_backend": "local_deepseek",
            "dsmil": local_status.get('dsmil', {}),
            "total_compute": "76.4 TOPS (NPU 26.4 + GPU 40 + NCS2 10)"
        }

# CLI
if __name__ == "__main__":
    import json

    orchestrator = UnifiedAIOrchestrator()

    if len(sys.argv) < 2:
        print("\nDSMIL Unified Orchestrator - Usage:")
        print("  python3 unified_orchestrator.py status")
        print("  python3 unified_orchestrator.py query 'your question'")
        print("  python3 unified_orchestrator.py query 'your question' --backend gemini")
        print("  python3 unified_orchestrator.py query 'your question' --backend openai")
        print("  python3 unified_orchestrator.py image 'describe this' /path/to/image.jpg")
        print("\nLOCAL-FIRST: Defaults to local DeepSeek for privacy and no guardrails")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))

    elif cmd == "query" and len(sys.argv) > 2:
        query = sys.argv[2]
        backend = None

        # Check for --backend flag
        if '--backend' in sys.argv:
            idx = sys.argv.index('--backend')
            if idx + 1 < len(sys.argv):
                backend = sys.argv[idx + 1]

        print(f"\n🎯 Query: {query}")
        result = orchestrator.query(query, force_backend=backend)

        print(f"\n{'='*60}")
        print(result['response'])
        print(f"{'='*60}")
        print(f"Backend: {result['routed_to']}")
        if 'routing' in result:
            print(f"Routing: {result['routing']['emoji_tag']}")
        print(f"Model: {result.get('model', 'N/A')}")
        if 'inference_time' in result:
            print(f"Time: {result['inference_time']}s")
        print(f"Cost: ${result.get('cost', 0):.4f}")
        print(f"Privacy: {result.get('privacy', 'N/A')}")
        print(f"DSMIL Attested: {result.get('dsmil_attested', False)}")

    elif cmd == "image" and len(sys.argv) > 3:
        query = sys.argv[2]
        images = sys.argv[3:]

        print(f"\n🎯 Multimodal Query: {query}")
        print(f"   Images: {len(images)}")

        result = orchestrator.query(query, images=images)
        print(json.dumps(result, indent=2))
