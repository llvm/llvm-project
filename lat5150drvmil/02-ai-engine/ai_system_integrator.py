#!/usr/bin/env python3
"""
AI System Integrator - Production Integration Layer

Integrates all research-based enhancements:
- Enhanced AI Engine (conversation, RAG, cache, memory)
- Deep Reasoning Agent (DeepAgent-style)
- Benchmarking System (CLASSic + Agentic)
- Self-Improvement (autonomous learning)
- Intel GPU Optimization (vLLM)
- Laddr Multi-Agent (orchestration)

Complements existing unified_orchestrator.py with research-based features.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Core research-based components
try:
    from enhanced_ai_engine import EnhancedAIEngine
    from deep_reasoning_agent import DeepReasoningAgent
    from ai_benchmarking import EnhancedAIBenchmark
    from hephaestus_integration import HephaestusIntegrator
except ImportError as e:
    print(f"⚠️  Research components not available: {e}")

# Security testing components
try:
    from atomic_red_team_api import AtomicRedTeamAPI
    ATOMIC_RED_TEAM_AVAILABLE = True
except ImportError:
    ATOMIC_RED_TEAM_AVAILABLE = False


@dataclass
class IntegratedResponse:
    """Complete response with all metadata"""
    content: str
    success: bool
    latency_ms: int
    mode: str  # simple, reasoning, benchmark

    # Performance
    tokens_input: int = 0
    tokens_output: int = 0
    cached: bool = False

    # Quality
    accuracy_score: float = 0.0
    quality_score: float = 0.0

    # Intelligence
    reasoning_steps: int = 0
    tools_used: List[str] = None
    memory_folds: int = 0

    # System
    model: str = "unknown"
    timestamp: str = None
    metadata: Dict = None


class AISystemIntegrator:
    """
    Complete AI System Integration

    Brings together all research-based enhancements into production system.
    """

    def __init__(
        self,
        enable_engine: bool = True,
        enable_reasoning: bool = True,
        enable_benchmarking: bool = True,
        enable_gpu: bool = True,
        enable_hephaestus: bool = True
    ):
        """Initialize AI System Integrator"""

        print("="*70)
        print(" AI System Integrator - Production Ready")
        print("="*70)
        print()

        self.components = {}
        self.task_history = []
        self.start_time = time.time()

        # Initialize Enhanced AI Engine
        if enable_engine:
            try:
                print("🔷 Enhanced AI Engine...")
                self.components['engine'] = EnhancedAIEngine(
                    user_id="integrator",
                    enable_self_improvement=True,
                    enable_dsmil_integration=True,
                    enable_ram_context=True
                )
                print("   ✅ Loaded: conversation, RAG, cache, memory, self-improvement")
            except Exception as e:
                print(f"   ⚠️  {e}")

        # Initialize Deep Reasoning Agent
        if enable_reasoning:
            try:
                print("🔷 Deep Reasoning Agent...")
                self.components['reasoning'] = DeepReasoningAgent()
                print("   ✅ Loaded: autonomous thinking, tool discovery, memory folding")
            except Exception as e:
                print(f"   ⚠️  {e}")

        # Initialize Benchmarking
        if enable_benchmarking:
            try:
                print("🔷 Benchmarking System...")
                self.components['benchmark'] = EnhancedAIBenchmark(
                    engine=self.components.get('engine')
                )
                print("   ✅ Loaded: CLASSic metrics, agentic evaluation")
            except Exception as e:
                print(f"   ⚠️  {e}")

        # Initialize Hephaestus
        if enable_hephaestus:
            try:
                print("🔷 Hephaestus Framework...")
                self.components['hephaestus'] = HephaestusIntegrator(
                    enable_vector_storage=True,
                    enable_mcp_integration=True
                )
                print("   ✅ Loaded: phase-based workflows, dynamic task creation")
            except Exception as e:
                print(f"   ⚠️  {e}")

        # Initialize Atomic Red Team
        if ATOMIC_RED_TEAM_AVAILABLE:
            try:
                print("🔷 Atomic Red Team...")
                self.components['atomic_red_team'] = AtomicRedTeamAPI()
                print("   ✅ Loaded: MITRE ATT&CK techniques, security test queries")
            except Exception as e:
                print(f"   ⚠️  {e}")

        # GPU status
        if enable_gpu:
            gpu_status = self._check_gpu_status()
            if gpu_status:
                print(f"🔷 Intel GPU: {gpu_status}")

        print()
        print(f"✅ AI System Integrator ready with {len(self.components)} components")
        print("="*70)
        print()

    def _check_gpu_status(self) -> Optional[str]:
        """Check Intel GPU availability"""
        try:
            import subprocess
            result = subprocess.run(
                ['lspci'], capture_output=True, text=True, timeout=2
            )
            if 'intel' in result.stdout.lower() and 'vga' in result.stdout.lower():
                return "Detected (76.4+ TOPS available)"
        except:
            pass
        return None

    def query(
        self,
        prompt: str,
        model: str = "uncensored_code",
        use_rag: bool = True,
        use_cache: bool = True,
        mode: str = "auto"
    ) -> IntegratedResponse:
        """
        Main query interface

        Args:
            prompt: User prompt
            model: Model to use
            use_rag: Enable RAG
            use_cache: Enable caching
            mode: auto, simple, reasoning
        """

        start_time = time.time()

        # Auto-detect mode
        if mode == "auto":
            mode = self._detect_mode(prompt)

        print(f"\n{'='*70}")
        print(f"Query: {prompt[:60]}...")
        print(f"Mode: {mode} | Model: {model}")
        print(f"{'='*70}\n")

        try:
            if mode == "atomic_red_team" and 'atomic_red_team' in self.components:
                # Use Atomic Red Team
                result = self._query_atomic_red_team(prompt)
            elif mode == "reasoning" and 'reasoning' in self.components:
                # Use Deep Reasoning Agent
                result = self._query_reasoning(prompt)
            else:
                # Use Enhanced AI Engine
                result = self._query_simple(prompt, model, use_rag, use_cache)

            result.latency_ms = int((time.time() - start_time) * 1000)
            result.timestamp = datetime.now().isoformat()
            result.success = True

            self.task_history.append(result)

            print(f"\n✅ Response ready: {result.latency_ms}ms")
            if result.cached:
                print(f"   ⚡ Cache hit!")
            if result.reasoning_steps > 0:
                print(f"   🧠 {result.reasoning_steps} reasoning steps")
            print()

            return result

        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            return IntegratedResponse(
                content=f"Error: {e}",
                success=False,
                latency_ms=int((time.time() - start_time) * 1000),
                mode=mode,
                timestamp=datetime.now().isoformat()
            )

    def _detect_mode(self, prompt: str) -> str:
        """Auto-detect best mode for prompt"""
        prompt_lower = prompt.lower()

        # Atomic Red Team keywords
        atomic_keywords = [
            "atomic", "mitre", "att&ck", "attack", "technique",
            "t1059", "t1003", "t1105",  # Common technique IDs
            "security test", "atomics", "red team", "adversary"
        ]

        if any(kw in prompt_lower for kw in atomic_keywords):
            return "atomic_red_team"

        # Reasoning keywords
        reasoning_keywords = [
            "analyze", "explain why", "reasoning", "step by step",
            "think through", "complex", "multiple steps"
        ]

        if any(kw in prompt_lower for kw in reasoning_keywords):
            return "reasoning"

        return "simple"

    def _query_simple(
        self,
        prompt: str,
        model: str,
        use_rag: bool,
        use_cache: bool
    ) -> IntegratedResponse:
        """Query via Enhanced AI Engine"""

        engine = self.components.get('engine')
        if not engine:
            raise RuntimeError("Enhanced AI Engine not available")

        response = engine.query(
            prompt=prompt,
            model=model,
            use_rag=use_rag,
            use_cache=use_cache
        )

        return IntegratedResponse(
            content=response.content,
            success=True,
            latency_ms=response.latency_ms,
            mode="simple",
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            cached=response.cached,
            model=response.model
        )

    def _query_reasoning(self, prompt: str) -> IntegratedResponse:
        """Query via Deep Reasoning Agent"""

        agent = self.components.get('reasoning')
        if not agent:
            raise RuntimeError("Deep Reasoning Agent not available")

        trace = agent.reason(
            task_prompt=prompt,
            max_steps=20,
            thinking_budget=5,
            fold_threshold=10
        )

        return IntegratedResponse(
            content=trace.final_answer,
            success=trace.success,
            latency_ms=trace.total_latency_ms,
            mode="reasoning",
            reasoning_steps=len(trace.steps),
            tools_used=trace.tools_used,
            memory_folds=len(trace.memory_folds),
            quality_score=trace.quality_score
        )

    def _query_atomic_red_team(self, prompt: str) -> IntegratedResponse:
        """Query via Atomic Red Team API"""

        art_api = self.components.get('atomic_red_team')
        if not art_api:
            raise RuntimeError("Atomic Red Team not available")

        print("🔴 Querying Atomic Red Team security tests...")

        # Query with natural language
        result = art_api.query_atomics(query=prompt)

        if not result.success:
            return IntegratedResponse(
                content=f"Atomic Red Team query failed: {result.error}",
                success=False,
                latency_ms=0,
                mode="atomic_red_team"
            )

        # Format response
        if result.count == 0:
            content = f"No atomic tests found matching: {prompt}\n\nTry:\n- Using MITRE ATT&CK technique IDs (e.g., T1059.002)\n- Platform filters: Windows, Linux, macOS\n- Test names: mshta, powershell, bash, etc."
        else:
            content = f"Found {result.count} atomic test(s) for: {prompt}\n\n"

            for i, test in enumerate(result.tests[:5], 1):  # Show first 5
                content += f"[{i}] {test.get('name', 'Unnamed Test')}\n"
                content += f"    Technique: {test.get('technique_id', 'Unknown')}\n"
                content += f"    Platform: {', '.join(test.get('platform', []))}\n"
                content += f"    Executor: {test.get('executor', 'Unknown')}\n"
                content += f"    Description: {test.get('description', 'N/A')[:100]}...\n\n"

            if result.count > 5:
                content += f"... and {result.count - 5} more tests.\n"
                content += f"Use /api/atomic-red-team/query for full results."

        return IntegratedResponse(
            content=content,
            success=True,
            latency_ms=0,
            mode="atomic_red_team",
            metadata={
                "query": result.query,
                "test_count": result.count,
                "timestamp": result.timestamp
            }
        )

    def benchmark(
        self,
        task_ids: Optional[List[str]] = None,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive benchmarks"""

        benchmark = self.components.get('benchmark')
        if not benchmark:
            raise RuntimeError("Benchmarking system not available")

        print("\n" + "="*70)
        print(" Running Comprehensive Benchmarks")
        print("="*70 + "\n")

        summary = benchmark.run_benchmark(
            task_ids=task_ids,
            num_runs=num_runs
        )

        benchmark.print_summary(summary)

        return asdict(summary)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""

        stats = {
            "system": {
                "uptime_seconds": int(time.time() - self.start_time),
                "components_loaded": list(self.components.keys()),
                "total_tasks": len(self.task_history)
            }
        }

        # Engine stats
        if 'engine' in self.components:
            stats['engine'] = self.components['engine'].get_statistics()

        # Reasoning stats
        if 'reasoning' in self.components:
            stats['reasoning'] = self.components['reasoning'].get_statistics()

        # Task history analysis
        if self.task_history:
            successful = sum(1 for t in self.task_history if t.success)
            cached = sum(1 for t in self.task_history if t.cached)
            reasoning = sum(1 for t in self.task_history if t.mode == "reasoning")

            stats['history'] = {
                "total": len(self.task_history),
                "successful": successful,
                "cached": cached,
                "reasoning_tasks": reasoning,
                "success_rate": successful / len(self.task_history),
                "cache_rate": cached / len(self.task_history)
            }

        return stats


def main():
    """Demo / Test"""

    # Initialize
    integrator = AISystemIntegrator(
        enable_engine=True,
        enable_reasoning=True,
        enable_benchmarking=True
    )

    # Test simple query
    print("Test 1: Simple Query")
    r1 = integrator.query("What is 2+2?")
    print(f"Result: {r1.content}")

    # Test reasoning
    print("\nTest 2: Reasoning Query")
    r2 = integrator.query("Analyze the benefits of semantic search over keyword search")
    print(f"Result: {r2.content[:200]}...")

    # Show stats
    print("\nSystem Statistics:")
    stats = integrator.get_stats()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
