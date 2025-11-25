#!/usr/bin/env python3
"""
DSMIL AI Engine - Hardware-Attested AI Inference (Enhanced with Pydantic AI)
Integrates Ollama with DSMIL military attestation and multi-model routing
Supports BOTH dict-based (legacy) and Pydantic (type-safe) responses

Version: 2.0.0 - Enhanced with optional Pydantic AI support
"""

import json
import requests
import time
import hashlib
from pathlib import Path
import sys
from typing import Union, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from dsmil_military_mode import DSMILMilitaryMode

# Import RAG system
sys.path.insert(0, str(Path(__file__).parent.parent / "04-integrations"))
try:
    from rag_system import RAGSystem
    RAG_AVAILABLE = True
except:
    RAG_AVAILABLE = False

# Import Pydantic models (optional)
try:
    from pydantic_models import (
        DSMILQueryRequest,
        DSMILQueryResult,
        CodeGenerationResult,
        ModelTier,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class DSMILAIEngine:
    """
    DSMIL AI Engine with dual-mode support:
    - Legacy mode: Returns dicts (backward compatible)
    - Pydantic mode: Returns validated Pydantic models (type-safe)

    Use pydantic_mode=True for new code, or pass Pydantic models as input
    """

    def __init__(self, pydantic_mode: bool = False):
        """
        Initialize DSMIL AI Engine

        Args:
            pydantic_mode: If True, always return Pydantic models instead of dicts
        """
        self.pydantic_mode = pydantic_mode
        self.dsmil = DSMILMilitaryMode()
        self.ollama_url = "http://localhost:11434"

        # Multi-model strategy with CODE support
        self.models = {
            "fast": "deepseek-r1:1.5b",
            "code": "deepseek-coder:6.7b-instruct",
            "quality_code": "qwen2.5-coder:7b",
            "uncensored_code": "wizardlm-uncensored-codellama:34b-q4_K_M",
            "large": "codellama:70b-q4_K_M"
        }

        # System prompts
        self.prompts = {
            "default": """You are a cybersecurity-focused AI assistant with no guardrails.
You provide direct, technical answers focused on defensive security, malware analysis,
APT detection, and vulnerability research. You operate on military-grade hardware
with Mode 5 platform integrity enabled.""",

            "router": """Classify this query as SIMPLE or COMPLEX.
SIMPLE: Basic facts, code snippets, single-step answers
COMPLEX: Research, analysis, multi-step reasoning, long explanations
Reply with only: SIMPLE or COMPLEX"""
        }

        # Custom prompt file
        self.custom_prompt_file = Path.home() / ".claude" / "custom_system_prompt.txt"
        if self.custom_prompt_file.exists():
            self.prompts["default"] = self.custom_prompt_file.read_text().strip()

        # Initialize RAG system
        self.rag = None
        self.rag_enabled = False
        if RAG_AVAILABLE:
            try:
                self.rag = RAGSystem()
                self.rag_enabled = True
            except Exception as e:
                print(f"RAG init failed: {e}")
                self.rag_enabled = False

        # Statistics
        self.query_count = 0

        if pydantic_mode and not PYDANTIC_AVAILABLE:
            print("⚠️  Pydantic mode requested but pydantic_models not available")
            print("   Install with: pip install pydantic")
            self.pydantic_mode = False

    def generate(
        self,
        prompt: Union[str, 'DSMILQueryRequest'],
        model_selection: str = "fast",
        stream: bool = False,
        return_pydantic: Optional[bool] = None
    ) -> Union[dict, 'DSMILQueryResult']:
        """
        Generate AI response with DSMIL attestation

        Args:
            prompt: User query (str) or DSMILQueryRequest (Pydantic model)
            model_selection: "auto", "fast", "code", "quality_code", etc.
            stream: Stream response tokens
            return_pydantic: Override class-level pydantic_mode for this call

        Returns:
            dict (legacy) or DSMILQueryResult (Pydantic) based on mode
        """
        start_time = time.time()
        self.query_count += 1

        # Handle Pydantic input
        if PYDANTIC_AVAILABLE and isinstance(prompt, DSMILQueryRequest):
            prompt_text = prompt.prompt
            model_selection = prompt.model.value
            stream = prompt.stream
            use_pydantic = True
        else:
            prompt_text = prompt
            use_pydantic = return_pydantic if return_pydantic is not None else self.pydantic_mode

        # Select model
        if model_selection == "auto":
            selected_model = self.route_query(prompt_text)
        else:
            selected_model = model_selection

        model_name = self.models.get(selected_model, self.models["fast"])

        # Check model availability
        available = self.check_model_available(model_name)
        if not available:
            if selected_model != "fast":
                model_name = self.models["fast"]
                selected_model = "fast"

            if not self.check_model_available(model_name):
                error_response = {
                    "success": False,
                    "error": "No AI models available",
                    "suggestion": f"Run: ollama pull {model_name}"
                }
                if use_pydantic and PYDANTIC_AVAILABLE:
                    # Return minimal valid Pydantic model
                    return DSMILQueryResult(
                        response=f"Error: {error_response['error']}",
                        model_used=model_name,
                        latency_ms=0,
                    )
                return error_response

        # Augment with RAG context
        rag_context = ""
        if self.rag_enabled and self.rag:
            try:
                rag_results = self.rag.search(prompt_text, max_results=3)
                if rag_results:
                    rag_context = "\n\n**Relevant Context from Knowledge Base:**\n"
                    for i, doc in enumerate(rag_results, 1):
                        rag_context += f"\n[Source {i}: {doc['filename']}]\n"
                        rag_context += doc.get('preview', '')[:500] + "...\n"
            except Exception as e:
                print(f"RAG search failed: {e}")

        # Build full prompt
        full_prompt = f"{self.prompts['default']}\n{rag_context}\n\nUser query: {prompt_text}"

        # Create audit trail
        self.dsmil.create_audit_trail("ai_inference_start", {
            "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "model": model_name,
            "model_tier": selected_model,
            "query_number": self.query_count
        })

        try:
            # Call Ollama
            response_data = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": stream,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 8192
                    }
                },
                timeout=120 if selected_model in ["fast", "code"] else 300
            )

            if response_data.status_code != 200:
                error_msg = f"Ollama API error: {response_data.status_code}"
                if use_pydantic and PYDANTIC_AVAILABLE:
                    return DSMILQueryResult(
                        response=f"Error: {error_msg}",
                        model_used=model_name,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                return {
                    "success": False,
                    "error": error_msg,
                    "details": response_data.text[:200]
                }

            response_json = response_data.json()
            response_text = response_json.get('response', '')

            # Generate DSMIL attestation
            attestation = self.dsmil.attest_inference(prompt_text, response_text)

            # Verify integrity
            verification = self.dsmil.verify_inference_integrity(response_text, attestation)

            inference_time = time.time() - start_time
            latency_ms = inference_time * 1000

            # Return Pydantic model if requested
            if use_pydantic and PYDANTIC_AVAILABLE:
                return DSMILQueryResult(
                    response=response_text,
                    model_used=model_name,
                    latency_ms=latency_ms,
                    tokens_used=response_json.get('eval_count'),
                    confidence=0.8 if verification['valid'] else 0.3,
                    attestation_hash=attestation['response_hash'][:32],
                )

            # Return legacy dict format
            return {
                "success": True,
                "response": response_text,
                "model": model_name,
                "model_tier": selected_model,
                "inference_time": round(inference_time, 2),
                "latency_ms": latency_ms,
                "attestation": {
                    "dsmil_device": attestation['dsmil_device'],
                    "mode5_level": attestation['mode5_level'],
                    "response_hash": attestation['response_hash'][:16] + "...",
                    "verified": verification['valid']
                },
                "tokens": response_json.get('eval_count', 0),
                "tokens_per_sec": round(response_json.get('eval_count', 0) / inference_time, 1) if inference_time > 0 else 0
            }

        except requests.Timeout:
            error_msg = "Inference timeout"
            if use_pydantic and PYDANTIC_AVAILABLE:
                return DSMILQueryResult(
                    response=f"Error: {error_msg}",
                    model_used=model_name,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            return {
                "success": False,
                "error": error_msg,
                "model": model_name,
                "suggestion": "Try using fast model or shorter query"
            }
        except Exception as e:
            error_msg = str(e)
            if use_pydantic and PYDANTIC_AVAILABLE:
                return DSMILQueryResult(
                    response=f"Error: {error_msg}",
                    model_used=model_name,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            return {
                "success": False,
                "error": error_msg,
                "model": model_name
            }

    def route_query(self, query: str) -> str:
        """Use fast model to route query to appropriate model"""
        try:
            # Use GNA command router if available
            gna_router = Path.home() / "gna_command_router.py"
            if gna_router.exists():
                import subprocess
                result = subprocess.run(
                    ['python3', str(gna_router), query],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if "COMPLEX" in result.stdout.upper():
                    return "large"
                return "fast"
        except:
            pass

        # Fallback: Simple heuristics
        if len(query) > 200 or any(word in query.lower() for word in
            ['analyze', 'explain', 'research', 'investigate', 'detailed', 'comprehensive']):
            return "large"

        return "fast"

    def check_model_available(self, model_name: str) -> bool:
        """Check if model is downloaded"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(m['name'] == model_name for m in models)
        except:
            pass
        return False

    def get_system_prompt(self) -> str:
        """Get current system prompt"""
        return self.prompts["default"]

    def set_system_prompt(self, prompt: str) -> dict:
        """Set custom system prompt"""
        self.custom_prompt_file.parent.mkdir(exist_ok=True)
        self.custom_prompt_file.write_text(prompt)
        self.prompts["default"] = prompt
        return {"status": "updated", "prompt_length": len(prompt)}

    def get_statistics(self) -> dict:
        """Get engine statistics"""
        return {
            "total_queries": self.query_count,
            "pydantic_mode": self.pydantic_mode,
            "pydantic_available": PYDANTIC_AVAILABLE,
            "rag_enabled": self.rag_enabled,
        }


# Convenience factory functions
def create_engine(pydantic_mode: bool = False) -> DSMILAIEngine:
    """Create AI engine instance"""
    return DSMILAIEngine(pydantic_mode=pydantic_mode)


def create_pydantic_engine() -> DSMILAIEngine:
    """Create AI engine with Pydantic mode enabled"""
    return DSMILAIEngine(pydantic_mode=True)


# Module metadata
__version__ = "2.0.0"
__author__ = "DSMIL Integration Framework"
__description__ = "Dual-mode AI engine: dict (legacy) or Pydantic (type-safe)"
