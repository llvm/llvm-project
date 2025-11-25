#!/usr/bin/env python3
"""
Test script for dual-mode AI engine (dict vs Pydantic)
Verifies backward compatibility and type-safe mode work correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE

if PYDANTIC_AVAILABLE:
    from pydantic_models import DSMILQueryRequest, DSMILQueryResult, ModelTier


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def test_legacy_mode():
    """Test 1: Legacy dict mode (backward compatibility)"""
    print(f"\n{Colors.CYAN}{'='*70}")
    print("Test 1: Legacy Dict Mode (Backward Compatibility)")
    print(f"{'='*70}{Colors.RESET}\n")

    try:
        engine = DSMILAIEngine(pydantic_mode=False)
        result = engine.generate("What is 2+2?", model_selection="fast")

        # Verify dict response
        assert isinstance(result, dict), "Result should be a dict"
        assert 'success' in result or 'response' in result, "Dict should have success or response key"

        print(f"{Colors.GREEN}✓ Legacy mode works{Colors.RESET}")
        print(f"  Type: {type(result).__name__}")
        print(f"  Keys: {list(result.keys())[:5]}")
        if result.get('success'):
            print(f"  Response length: {len(result.get('response', ''))}")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Legacy mode failed: {e}{Colors.RESET}")
        return False


def test_pydantic_mode():
    """Test 2: Pydantic type-safe mode"""
    if not PYDANTIC_AVAILABLE:
        print(f"\n{Colors.YELLOW}⊘ Pydantic mode skipped (not installed){Colors.RESET}")
        return True

    print(f"\n{Colors.CYAN}{'='*70}")
    print("Test 2: Pydantic Type-Safe Mode")
    print(f"{'='*70}{Colors.RESET}\n")

    try:
        engine = DSMILAIEngine(pydantic_mode=True)
        request = DSMILQueryRequest(
            prompt="What is the capital of France?",
            model=ModelTier.FAST
        )
        result = engine.generate(request)

        # Verify Pydantic response
        assert isinstance(result, DSMILQueryResult), f"Result should be DSMILQueryResult, got {type(result)}"
        assert hasattr(result, 'response'), "Result should have .response attribute"
        assert hasattr(result, 'model_used'), "Result should have .model_used attribute"
        assert hasattr(result, 'latency_ms'), "Result should have .latency_ms attribute"

        print(f"{Colors.GREEN}✓ Pydantic mode works{Colors.RESET}")
        print(f"  Type: {type(result).__name__}")
        print(f"  Model: {result.model_used}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Response length: {len(result.response)}")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Pydantic mode failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_mode():
    """Test 3: Hybrid mode (per-call override)"""
    if not PYDANTIC_AVAILABLE:
        print(f"\n{Colors.YELLOW}⊘ Hybrid mode skipped (Pydantic not installed){Colors.RESET}")
        return True

    print(f"\n{Colors.CYAN}{'='*70}")
    print("Test 3: Hybrid Mode (Per-Call Override)")
    print(f"{'='*70}{Colors.RESET}\n")

    try:
        # Create engine in legacy mode
        engine = DSMILAIEngine(pydantic_mode=False)

        # Call 1: Default to dict
        result1 = engine.generate("Hello", model_selection="fast")
        assert isinstance(result1, dict), "Default should return dict"

        # Call 2: Override to Pydantic
        result2 = engine.generate("Hello", model_selection="fast", return_pydantic=True)
        assert isinstance(result2, DSMILQueryResult), "Override should return Pydantic"

        print(f"{Colors.GREEN}✓ Hybrid mode works{Colors.RESET}")
        print(f"  Call 1 type: {type(result1).__name__} (default dict)")
        print(f"  Call 2 type: {type(result2).__name__} (override Pydantic)")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Hybrid mode failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False


def test_pydantic_input():
    """Test 4: Pydantic request as input"""
    if not PYDANTIC_AVAILABLE:
        print(f"\n{Colors.YELLOW}⊘ Pydantic input test skipped (not installed){Colors.RESET}")
        return True

    print(f"\n{Colors.CYAN}{'='*70}")
    print("Test 4: Pydantic Request Input")
    print(f"{'='*70}{Colors.RESET}\n")

    try:
        # Legacy engine can accept Pydantic request
        engine = DSMILAIEngine(pydantic_mode=False)
        request = DSMILQueryRequest(
            prompt="Test query",
            model=ModelTier.FAST,
            temperature=0.5
        )

        # When input is Pydantic, output should be Pydantic too
        result = engine.generate(request)
        assert isinstance(result, DSMILQueryResult), "Pydantic input should return Pydantic output"

        print(f"{Colors.GREEN}✓ Pydantic request input works{Colors.RESET}")
        print(f"  Input type: DSMILQueryRequest")
        print(f"  Output type: {type(result).__name__}")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Pydantic input failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """Test 5: Engine statistics"""
    print(f"\n{Colors.CYAN}{'='*70}")
    print("Test 5: Engine Statistics")
    print(f"{'='*70}{Colors.RESET}\n")

    try:
        engine = DSMILAIEngine(pydantic_mode=True)
        engine.generate("Test", model_selection="fast")

        stats = engine.get_statistics()

        assert 'total_queries' in stats, "Stats should have total_queries"
        assert 'pydantic_mode' in stats, "Stats should have pydantic_mode"
        assert 'pydantic_available' in stats, "Stats should have pydantic_available"

        print(f"{Colors.GREEN}✓ Statistics work{Colors.RESET}")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Pydantic mode: {stats['pydantic_mode']}")
        print(f"  Pydantic available: {stats['pydantic_available']}")
        print(f"  RAG enabled: {stats['rag_enabled']}")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Statistics failed: {e}{Colors.RESET}")
        return False


def main():
    print(f"\n{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║     DSMIL AI Engine - Dual-Mode Test Suite                          ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝{Colors.RESET}")

    print(f"\n{Colors.YELLOW}Environment:{Colors.RESET}")
    print(f"  Pydantic AI available: {PYDANTIC_AVAILABLE}")

    results = []

    # Run tests
    results.append(("Legacy Dict Mode", test_legacy_mode()))
    results.append(("Pydantic Type-Safe Mode", test_pydantic_mode()))
    results.append(("Hybrid Mode", test_hybrid_mode()))
    results.append(("Pydantic Request Input", test_pydantic_input()))
    results.append(("Engine Statistics", test_statistics()))

    # Summary
    print(f"\n{Colors.CYAN}{'='*70}")
    print("Test Summary")
    print(f"{'='*70}{Colors.RESET}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{Colors.GREEN}✓ PASS" if result else f"{Colors.RED}✗ FAIL"
        print(f"  {status:<20}{Colors.RESET} {name}")

    print()
    if passed == total:
        print(f"{Colors.GREEN}All tests passed! ({passed}/{total}){Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}Some tests failed ({passed}/{total}){Colors.RESET}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
