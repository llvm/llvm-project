#!/usr/bin/env python3
"""
Test Atomic Red Team Natural Language Integration

Demonstrates how natural language queries are automatically routed to
Atomic Red Team security testing framework.
"""

import sys
from ai_system_integrator import AISystemIntegrator

def test_nl_queries():
    """Test various natural language queries"""

    print("=" * 70)
    print(" Testing Atomic Red Team Natural Language Integration")
    print("=" * 70)
    print()

    # Initialize integrator
    print("Initializing AI System Integrator...")
    integrator = AISystemIntegrator(
        enable_engine=True,
        enable_reasoning=False,
        enable_benchmarking=False,
        enable_gpu=False,
        enable_hephaestus=False
    )
    print()

    # Test queries
    test_queries = [
        "Show me atomic tests for T1059.002",
        "Find mshta atomics for Windows",
        "List all MITRE ATT&CK technique tests for macOS",
        "Search for red team security tests with PowerShell",
        "What atomic tests are available for Linux?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_queries)}: {query}")
        print(f"{'='*70}\n")

        try:
            response = integrator.query(prompt=query)

            print(f"Mode detected: {response.mode}")
            print(f"Success: {response.success}")
            print(f"Latency: {response.latency_ms}ms")
            print(f"\nResponse:\n{response.content}\n")

            if response.metadata:
                print(f"Metadata: {response.metadata}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*70)
    print(" Testing Complete")
    print("="*70)


if __name__ == "__main__":
    try:
        test_nl_queries()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
