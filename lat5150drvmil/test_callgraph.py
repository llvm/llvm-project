#!/usr/bin/env python3
"""
Test call graph analysis (Phase 3)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AI_ENGINE_DIR = PROJECT_ROOT / "02-ai-engine"

if str(AI_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_DIR))

from codebase_learner import CodebaseLearner

def main():
    print("=== Phase 3: Call Graph Analysis Test ===\n")

    # Initialize learner
    learner = CodebaseLearner(workspace_root=str(PROJECT_ROOT))

    # Learn from execution_engine.py (a good example file)
    print("1. Learning from execution_engine.py...")
    result = learner.learn_from_file(str(AI_ENGINE_DIR / "execution_engine.py"))
    print(f"   Functions: {result['functions']}, Classes: {result['classes']}")

    # Get call graph stats
    print("\n2. Call Graph Statistics:")
    stats = learner.get_call_graph_stats()
    print(f"   Total functions: {stats['total_functions']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Dead code count: {stats['dead_code']['total']}")
    print(f"   Cycles: {stats['cycle_count']}")
    print(f"   Avg calls per function: {stats['avg_calls_per_function']:.2f}")

    # Show hotspots
    if stats['hotspots']:
        print("\n3. Top 5 Hotspots (most-called functions):")
        for func, call_count in stats['hotspots'][:5]:
            print(f"   {func}: called {call_count} times")

    # Show complex functions
    if stats['complex_functions']:
        print("\n4. Top 5 Complex Functions (call many others):")
        for func, callee_count in stats['complex_functions'][:5]:
            print(f"   {func}: calls {callee_count} other functions")

    # Test impact analysis
    print("\n5. Impact Analysis Example:")
    if stats['total_functions'] > 0:
        # Pick first function in call graph
        example_func = list(learner.function_locations.keys())[0]
        impact = learner.find_impact(example_func)

        if "error" not in impact:
            print(f"   Function: {example_func}")
            print(f"   Direct callers: {impact['direct_caller_count']}")
            print(f"   Transitive callers: {impact['transitive_caller_count']}")
            print(f"   Impact score: {impact['impact_score']}/100")
            print(f"   Risk level: {impact['risk_level']}")

    # Show dead code
    print("\n6. Dead Code:")
    dead_code = learner.find_dead_code()
    if dead_code['total'] > 0:
        print(f"   Dead functions: {len(dead_code['functions'])}")
        print(f"   Dead methods: {len(dead_code['methods'])}")
        if dead_code['functions']:
            print(f"   Examples: {dead_code['functions'][:3]}")
    else:
        print("   No dead code found (all functions are called)")

    # Show cycles
    print("\n7. Dependency Cycles:")
    cycles = learner.find_dependency_cycles()
    if cycles:
        print(f"   Found {len(cycles)} cycles:")
        for i, cycle in enumerate(cycles[:3], 1):
            print(f"   Cycle {i}: {' -> '.join(cycle)} -> {cycle[0]}")
    else:
        print("   No circular dependencies detected ✓")

    print("\n✓ Call graph analysis complete!")


if __name__ == "__main__":
    main()
