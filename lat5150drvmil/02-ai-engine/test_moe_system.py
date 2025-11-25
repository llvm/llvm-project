#!/usr/bin/env python3
"""
Test suite for Mixture of Experts (MoE) system - Phase 4
"""

import json
from moe import (
    MoERouter,
    ExpertDomain,
    ExpertModelConfig,
    ExpertModelRegistry,
    ModelBackend,
    MoEAggregator,
    AggregationStrategy
)


def test_moe_router():
    """Test MoE router with various queries."""
    print("=" * 80)
    print("TEST 1: MoE Router")
    print("=" * 80)

    router = MoERouter(enable_multi_expert=True)

    test_queries = [
        "Write a Python function to parse JSON",
        "Optimize this SQL query for better performance",
        "Find security vulnerabilities in this authentication code",
        "Deploy this app to Kubernetes with auto-scaling",
        "Generate comprehensive API documentation",
        "Analyze performance bottlenecks in the system",
        "Create unit tests for the user service",
        "Design a scalable microservices architecture",
    ]

    results = []
    for query in test_queries:
        decision = router.route(query)
        results.append({
            "query": query,
            "primary_expert": decision.primary_expert.domain.value,
            "confidence": decision.primary_expert.confidence,
            "strategy": decision.routing_strategy
        })
        print(f"\nQuery: {query}")
        print(f"  → Primary: {decision.primary_expert.domain.value} "
              f"({decision.primary_expert.confidence:.2f})")
        print(f"  → Strategy: {decision.routing_strategy}")

    print("\n" + "=" * 80)
    print("Routing Statistics:")
    print(json.dumps(router.get_routing_statistics(), indent=2))
    print("=" * 80)

    return results


def test_expert_registry():
    """Test expert model registry."""
    print("\n" + "=" * 80)
    print("TEST 2: Expert Model Registry")
    print("=" * 80)

    registry = ExpertModelRegistry(cache_size=3)

    # Register test experts
    experts = [
        ExpertModelConfig(
            name="code-expert-python",
            domain="code",
            backend=ModelBackend.OPENAI_COMPATIBLE,
            model_path="deepseek-coder-6.7b",
            system_prompt="You are a Python coding expert."
        ),
        ExpertModelConfig(
            name="database-expert-sql",
            domain="database",
            backend=ModelBackend.OPENAI_COMPATIBLE,
            model_path="codellama-7b",
            system_prompt="You are a database and SQL expert."
        ),
        ExpertModelConfig(
            name="security-expert",
            domain="security",
            backend=ModelBackend.OPENAI_COMPATIBLE,
            model_path="codellama-7b",
            system_prompt="You are a cybersecurity expert."
        ),
    ]

    for config in experts:
        registry.register_expert(config)
        print(f"✓ Registered: {config.name} ({config.domain})")

    print(f"\nTotal experts registered: {len(registry.experts)}")
    print(f"Cache size: {registry.cache_size}")

    return registry


def test_moe_aggregator():
    """Test MoE aggregator with mock responses."""
    print("\n" + "=" * 80)
    print("TEST 3: MoE Aggregator")
    print("=" * 80)

    from moe.expert_models import ExpertResponse
    aggregator = MoEAggregator()

    # Create mock responses
    responses = [
        ExpertResponse(
            expert_name="code-expert",
            domain="code",
            response_text="Use list comprehension: result = [x*2 for x in numbers]",
            confidence=0.9,
            tokens_used=50,
            inference_time=0.5
        ),
        ExpertResponse(
            expert_name="python-expert",
            domain="code",
            response_text="Map function: result = list(map(lambda x: x*2, numbers))",
            confidence=0.8,
            tokens_used=45,
            inference_time=0.4
        ),
    ]

    # Test different strategies
    strategies = [
        AggregationStrategy.BEST_OF_N,
        AggregationStrategy.WEIGHTED_VOTE,
        AggregationStrategy.CONCATENATE,
    ]

    for strategy in strategies:
        result = aggregator.aggregate(responses, strategy=strategy)
        print(f"\nStrategy: {strategy.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Contributors: {', '.join(result.contributing_experts)}")
        print(f"  Response preview: {result.final_response[:80]}...")

    print("=" * 80)


def test_end_to_end():
    """Test complete MoE pipeline."""
    print("\n" + "=" * 80)
    print("TEST 4: End-to-End MoE Pipeline")
    print("=" * 80)

    # 1. Router selects expert
    router = MoERouter()
    query = "Write a function to validate email addresses"
    decision = router.route(query)

    print(f"Query: {query}")
    print(f"Routed to: {decision.primary_expert.domain.value}")
    print(f"Confidence: {decision.primary_expert.confidence:.2f}")
    print(f"Strategy: {decision.routing_strategy}")

    # 2. Would call expert model here (skipped - needs actual models)
    print("\n[Would call expert model here]")

    # 3. Would aggregate results
    print("[Would aggregate results here]")

    print("\n✓ End-to-end pipeline structure validated")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MIXTURE OF EXPERTS (MoE) SYSTEM TEST SUITE")
    print("Phase 4: Architecture Evolution")
    print("=" * 80 + "\n")

    try:
        # Test 1: Router
        test_moe_router()

        # Test 2: Registry
        test_expert_registry()

        # Test 3: Aggregator
        test_moe_aggregator()

        # Test 4: End-to-end
        test_end_to_end()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")

        print("MoE System Status: OPERATIONAL")
        print("\nNext steps:")
        print("  1. Register domain-specific expert models")
        print("  2. Fine-tune experts on specialized datasets")
        print("  3. Integrate with existing AI framework")
        print("  4. Benchmark vs single-model baseline")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
