#!/usr/bin/env python3
"""
Multi-Model Evaluation Framework
Based on ai-that-works Episode #16: "Multi-Model Evaluation"

Key Capabilities:
- Test prompts across multiple models in parallel
- Compare results (latency, tokens, quality)
- Detect regressions between versions
- Recommend optimal model for each query type
- A/B testing for prompts

Benefits:
- Quality assurance for prompts
- Model performance comparison
- Regression detection
- Optimize model routing
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json
from pathlib import Path
import statistics


@dataclass
class EvaluationResult:
    """
    Result from evaluating a prompt on a single model

    Attributes:
        model: Model name
        prompt: Original prompt
        response: Model response
        latency_ms: Response time in milliseconds
        tokens_input: Input token count
        tokens_output: Output token count
        timestamp: When evaluation occurred
        error: Error message if evaluation failed
        metadata: Additional metrics
    """
    model: str
    prompt: str
    response: str
    latency_ms: int
    tokens_input: int
    tokens_output: int
    timestamp: datetime
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "model": self.model,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "response": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class ComparisonReport:
    """
    Comparison report across multiple models

    Attributes:
        prompt_hash: Hash of the prompt
        models_tested: List of model names tested
        fastest_model: Model with lowest latency
        most_efficient: Model with best tokens/latency ratio
        recommended_model: Overall recommended model
        results: Individual results per model
        analysis: Detailed analysis
    """
    prompt_hash: str
    models_tested: List[str]
    fastest_model: str
    most_efficient: str
    recommended_model: str
    results: Dict[str, EvaluationResult]
    analysis: Dict[str, Any]


class MultiModelEvaluator:
    """
    Evaluate prompts across multiple models for quality assurance

    Use Cases:
    - Test new prompts across all models before deployment
    - Compare model performance for different query types
    - Detect regressions when models are updated
    - A/B test prompt variations
    - Optimize model routing based on query characteristics
    """

    def __init__(self, engine, results_dir: str = "evaluation_results"):
        """
        Initialize multi-model evaluator

        Args:
            engine: AI engine instance (DSMILAIEngine or EnhancedAIEngine)
            results_dir: Directory to store evaluation results
        """
        self.engine = engine
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.history = []

    async def evaluate_prompt(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        temperature: float = 0.7,
        parallel: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate a prompt across multiple models

        Args:
            prompt: Prompt to evaluate
            models: List of model names (uses all if None)
            temperature: Sampling temperature
            parallel: Run evaluations in parallel (faster)

        Returns:
            Dictionary mapping model names to evaluation results
        """
        # Use all available models if not specified
        if models is None:
            if hasattr(self.engine, 'models'):
                models = list(self.engine.models.keys())
            elif hasattr(self.engine, 'models_config'):
                models = list(self.engine.models_config.get('models', {}).keys())
            else:
                models = ["fast", "code", "quality_code", "uncensored_code", "large"]

        if parallel:
            # Run all evaluations in parallel
            tasks = [
                self._evaluate_single(prompt, model, temperature)
                for model in models
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            results = {}
            for model, result in zip(models, results_list):
                if isinstance(result, Exception):
                    # Handle evaluation error
                    results[model] = EvaluationResult(
                        model=model,
                        prompt=prompt,
                        response="",
                        latency_ms=0,
                        tokens_input=0,
                        tokens_output=0,
                        timestamp=datetime.now(),
                        error=str(result)
                    )
                else:
                    results[model] = result
        else:
            # Run sequentially
            results = {}
            for model in models:
                results[model] = await self._evaluate_single(prompt, model, temperature)

        # Store in history
        self.history.append({
            "prompt": prompt,
            "timestamp": datetime.now(),
            "results": results
        })

        return results

    async def _evaluate_single(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> EvaluationResult:
        """
        Evaluate prompt on a single model

        Args:
            prompt: Prompt text
            model: Model name
            temperature: Sampling temperature

        Returns:
            EvaluationResult for this model
        """
        start_time = time.time()

        try:
            # Try different engine methods
            if hasattr(self.engine, 'generate'):
                # DSMILAIEngine
                result = self.engine.generate(prompt, model_selection=model)
                response = result.get('response', '')
                tokens_output = result.get('tokens', 0)
                error = result.get('error')

            elif hasattr(self.engine, 'query'):
                # EnhancedAIEngine
                result = self.engine.query(
                    prompt,
                    model=model,
                    temperature=temperature,
                    use_rag=False,  # Disable RAG for pure model comparison
                    use_cache=False  # Disable cache for fresh evaluation
                )
                response = result.content
                tokens_output = result.tokens_output
                error = None

            else:
                raise ValueError("Engine does not have generate() or query() method")

            latency_ms = int((time.time() - start_time) * 1000)
            tokens_input = len(prompt.split())  # Rough estimate

            return EvaluationResult(
                model=model,
                prompt=prompt,
                response=response,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                timestamp=datetime.now(),
                error=error
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return EvaluationResult(
                model=model,
                prompt=prompt,
                response="",
                latency_ms=latency_ms,
                tokens_input=0,
                tokens_output=0,
                timestamp=datetime.now(),
                error=str(e)
            )

    def compare_results(
        self,
        results: Dict[str, EvaluationResult],
        optimize_for: str = "quality"
    ) -> ComparisonReport:
        """
        Compare results across models

        Args:
            results: Dictionary of evaluation results
            optimize_for: Optimization criterion ("speed", "quality", "efficiency")

        Returns:
            ComparisonReport with analysis
        """
        # Filter out failed results
        valid_results = {
            model: result
            for model, result in results.items()
            if result.error is None and result.latency_ms > 0
        }

        if not valid_results:
            raise ValueError("No valid results to compare")

        # Find fastest model
        fastest_model = min(
            valid_results.items(),
            key=lambda x: x[1].latency_ms
        )[0]

        # Find most efficient (tokens per ms)
        most_efficient = max(
            valid_results.items(),
            key=lambda x: x[1].tokens_output / max(x[1].latency_ms, 1)
        )[0]

        # Recommend model based on optimization criterion
        if optimize_for == "speed":
            recommended = fastest_model
        elif optimize_for == "efficiency":
            recommended = most_efficient
        else:  # quality
            # For quality, prefer models with longer responses (more detailed)
            # and reasonable latency
            recommended = max(
                valid_results.items(),
                key=lambda x: x[1].tokens_output / (x[1].latency_ms / 1000)
            )[0]

        # Calculate statistics
        latencies = [r.latency_ms for r in valid_results.values()]
        token_counts = [r.tokens_output for r in valid_results.values()]

        analysis = {
            "models_tested": len(valid_results),
            "models_failed": len(results) - len(valid_results),
            "latency_stats": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "token_stats": {
                "mean": statistics.mean(token_counts),
                "median": statistics.median(token_counts),
                "min": min(token_counts),
                "max": max(token_counts),
                "stdev": statistics.stdev(token_counts) if len(token_counts) > 1 else 0
            },
            "efficiency_scores": {
                model: result.tokens_output / max(result.latency_ms, 1)
                for model, result in valid_results.items()
            }
        }

        # Generate prompt hash for tracking
        prompt_hash = hashlib.md5(
            list(results.values())[0].prompt.encode()
        ).hexdigest()[:8]

        return ComparisonReport(
            prompt_hash=prompt_hash,
            models_tested=list(valid_results.keys()),
            fastest_model=fastest_model,
            most_efficient=most_efficient,
            recommended_model=recommended,
            results=results,
            analysis=analysis
        )

    def detect_regressions(
        self,
        baseline: Dict[str, EvaluationResult],
        current: Dict[str, EvaluationResult],
        latency_threshold: float = 1.2,  # 20% slower = regression
        token_threshold: float = 0.8  # 20% fewer tokens = regression
    ) -> List[str]:
        """
        Detect regressions by comparing baseline vs current results

        Args:
            baseline: Baseline evaluation results
            current: Current evaluation results
            latency_threshold: Multiplier for latency regression (>1.0)
            token_threshold: Multiplier for token regression (<1.0)

        Returns:
            List of regression warnings
        """
        regressions = []

        for model in baseline.keys():
            if model not in current:
                regressions.append(f"{model}: Not tested in current evaluation")
                continue

            base = baseline[model]
            curr = current[model]

            # Check for errors
            if curr.error and not base.error:
                regressions.append(f"{model}: New error - {curr.error}")

            # Check latency regression
            if base.latency_ms > 0 and curr.latency_ms > 0:
                latency_ratio = curr.latency_ms / base.latency_ms
                if latency_ratio > latency_threshold:
                    regressions.append(
                        f"{model}: Latency regression - "
                        f"{base.latency_ms}ms → {curr.latency_ms}ms "
                        f"({latency_ratio:.1%} increase)"
                    )

            # Check token regression (shorter responses may indicate quality drop)
            if base.tokens_output > 0 and curr.tokens_output > 0:
                token_ratio = curr.tokens_output / base.tokens_output
                if token_ratio < token_threshold:
                    regressions.append(
                        f"{model}: Output quality regression - "
                        f"{base.tokens_output} → {curr.tokens_output} tokens "
                        f"({(1-token_ratio):.1%} decrease)"
                    )

        return regressions

    def save_results(
        self,
        results: Dict[str, EvaluationResult],
        filename: Optional[str] = None
    ) -> Path:
        """
        Save evaluation results to JSON file

        Args:
            results: Evaluation results to save
            filename: Optional filename (generates if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(
                list(results.values())[0].prompt.encode()
            ).hexdigest()[:8]
            filename = f"eval_{timestamp}_{prompt_hash}.json"

        filepath = self.results_dir / filename

        # Serialize results
        data = {
            model: result.to_dict()
            for model, result in results.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        if not self.history:
            return {"evaluations_count": 0}

        return {
            "evaluations_count": len(self.history),
            "total_models_tested": sum(
                len(h["results"]) for h in self.history
            ),
            "results_dir": str(self.results_dir)
        }


async def main():
    """Demo usage"""
    print("=== Multi-Model Evaluator Demo ===\n")

    # Mock engine for demo
    class MockEngine:
        models = {
            "fast": "deepseek-r1:1.5b",
            "code": "deepseek-coder:6.7b",
            "quality": "qwen2.5-coder:7b"
        }

        def generate(self, prompt, model_selection="fast"):
            # Simulate different models with different characteristics
            import random
            time.sleep(random.uniform(0.1, 0.5))  # Simulate latency

            responses = {
                "fast": "Quick response with basic information.",
                "code": "More detailed response with code examples and explanations.",
                "quality": "Comprehensive response with in-depth analysis, multiple approaches, and detailed code examples."
            }

            base_latency = {"fast": 100, "code": 300, "quality": 500}

            return {
                "response": responses.get(model_selection, "Default response"),
                "tokens": len(responses.get(model_selection, "").split()) * 2,
                "model": self.models.get(model_selection),
                "inference_time": base_latency.get(model_selection, 200) / 1000
            }

    engine = MockEngine()
    evaluator = MultiModelEvaluator(engine)

    # Test prompt
    prompt = "Explain the benefits of event-driven architecture for AI agents."

    print("1. Evaluating prompt across all models...")
    results = await evaluator.evaluate_prompt(prompt, parallel=True)

    print(f"\n   Tested {len(results)} models")
    for model, result in results.items():
        if result.error:
            print(f"   ❌ {model}: ERROR - {result.error}")
        else:
            print(f"   ✅ {model}: {result.latency_ms}ms, {result.tokens_output} tokens")

    # Compare results
    print("\n2. Comparing results...")
    comparison = evaluator.compare_results(results, optimize_for="quality")

    print(f"\n   Fastest model: {comparison.fastest_model}")
    print(f"   Most efficient: {comparison.most_efficient}")
    print(f"   Recommended: {comparison.recommended_model}")
    print(f"\n   Latency stats: {comparison.analysis['latency_stats']}")
    print(f"   Token stats: {comparison.analysis['token_stats']}")

    # Simulate regression test
    print("\n3. Detecting regressions...")
    baseline = results
    # Simulate slower current results
    current = await evaluator.evaluate_prompt(prompt, parallel=True)

    regressions = evaluator.detect_regressions(baseline, current)
    if regressions:
        print("   ⚠️  Regressions detected:")
        for reg in regressions:
            print(f"      - {reg}")
    else:
        print("   ✅ No regressions detected")

    # Save results
    print("\n4. Saving results...")
    filepath = evaluator.save_results(results)
    print(f"   Saved to: {filepath}")

    # Statistics
    print("\n5. Statistics:")
    stats = evaluator.get_statistics()
    print(f"   Total evaluations: {stats['evaluations_count']}")
    print(f"   Models tested: {stats['total_models_tested']}")


if __name__ == "__main__":
    asyncio.run(main())
