#!/usr/bin/env python3
"""
Interactive MoE (Mixture of Experts) System Probe

Tests expert routing, model selection, and aggregation strategies.

Usage:
    python 02_test_moe_system.py
"""

import sys
sys.path.insert(0, '../../02-ai-engine')

from moe import MoERouter, ExpertDomain, AggregationStrategy


def test_routing_queries():
    """Test router with real queries"""
    print("=" * 80)
    print("  MoE ROUTER TEST")
    print("=" * 80)

    router = MoERouter(enable_multi_expert=True)

    test_queries = [
        ("Code", "Write a Python function to parse JSON and validate schema"),
        ("Database", "Optimize this PostgreSQL query to reduce execution time"),
        ("Security", "Find SQL injection vulnerabilities in this authentication code"),
        ("Infrastructure", "Deploy this microservice to Kubernetes with auto-scaling"),
        ("Documentation", "Generate API documentation for this REST endpoint"),
        ("Analysis", "Analyze the performance bottlenecks in this system"),
        ("Testing", "Create comprehensive unit tests for the user service"),
        ("Strategic", "Design a scalable event-driven architecture"),
        ("Operations", "Set up monitoring and alerting for production failures"),
    ]

    for category, query in test_queries:
        decision = router.route(query)
        print(f"\n[{category}] {query}")
        print(f"  → Primary: {decision.primary_expert.domain.value} ({decision.primary_expert.confidence:.2f})")
        print(f"  → Strategy: {decision.routing_strategy}")

        if decision.secondary_experts:
            print(f"  → Secondary: {', '.join([e.domain.value for e in decision.secondary_experts])}")

    print("\n" + "=" * 80)
    print("Routing Statistics:")
    import json
    print(json.dumps(router.get_routing_statistics(), indent=2))


def test_custom_query():
    """Test router with custom query"""
    print("\n" + "=" * 80)
    print("  CUSTOM QUERY TEST")
    print("=" * 80)

    router = MoERouter(enable_multi_expert=True)

    query = input("\nEnter your query: ")
    decision = router.route(query)

    print(f"\nRouting Decision:")
    print(router.explain_routing(decision))


def interactive_menu():
    """Interactive menu"""
    while True:
        print("\n" + "=" * 80)
        print("  MoE SYSTEM INTERACTIVE PROBE")
        print("=" * 80)
        print("\n1. Test routing with predefined queries")
        print("2. Test with custom query")
        print("3. Show router statistics")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            test_routing_queries()
        elif choice == "2":
            test_custom_query()
        elif choice == "3":
            router = MoERouter()
            print("\nRouter Statistics:")
            import json
            print(json.dumps(router.get_routing_statistics(), indent=2))
        elif choice == "0":
            break


if __name__ == "__main__":
    interactive_menu()
