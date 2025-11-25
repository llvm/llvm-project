#!/usr/bin/env python3
"""
Simple import and syntax test for Pydantic AI integration
Tests that all modules can be imported without errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

# Test 1: Core engine import
try:
    from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE, create_engine, create_pydantic_engine
    print("✓ dsmil_ai_engine imports successfully")
    print(f"  PYDANTIC_AVAILABLE: {PYDANTIC_AVAILABLE}")
except Exception as e:
    print(f"✗ Failed to import dsmil_ai_engine: {e}")
    sys.exit(1)

# Test 2: Pydantic models import (if available)
if PYDANTIC_AVAILABLE:
    try:
        from pydantic_models import (
            DSMILQueryRequest,
            DSMILQueryResult,
            CodeGenerationResult,
            SecurityAnalysisResult,
            ModelTier,
            AIEngineConfig,
        )
        print("✓ pydantic_models imports successfully")
    except Exception as e:
        print(f"✗ Failed to import pydantic_models: {e}")
        sys.exit(1)
else:
    print("⊘ Pydantic models not available (install with: pip install pydantic)")

# Test 3: Create engines
try:
    engine_legacy = create_engine(pydantic_mode=False)
    print("✓ Legacy engine created successfully")

    if PYDANTIC_AVAILABLE:
        engine_pydantic = create_pydantic_engine()
        print("✓ Pydantic engine created successfully")

except Exception as e:
    print(f"✗ Failed to create engines: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify engine attributes
try:
    assert hasattr(engine_legacy, 'generate'), "Engine should have generate method"
    assert hasattr(engine_legacy, 'get_statistics'), "Engine should have get_statistics method"
    assert engine_legacy.pydantic_mode == False, "Legacy engine should have pydantic_mode=False"

    if PYDANTIC_AVAILABLE:
        assert engine_pydantic.pydantic_mode == True, "Pydantic engine should have pydantic_mode=True"

    print("✓ Engine attributes verified")

except Exception as e:
    print(f"✗ Engine verification failed: {e}")
    sys.exit(1)

# Test 5: Verify statistics
try:
    stats = engine_legacy.get_statistics()
    assert 'pydantic_mode' in stats, "Stats should have pydantic_mode"
    assert 'pydantic_available' in stats, "Stats should have pydantic_available"
    print("✓ Statistics method works")

except Exception as e:
    print(f"✗ Statistics failed: {e}")
    sys.exit(1)

# Test 6: Pydantic model creation (if available)
if PYDANTIC_AVAILABLE:
    try:
        request = DSMILQueryRequest(
            prompt="Test query",
            model=ModelTier.FAST,
            temperature=0.7
        )
        assert request.prompt == "Test query"
        assert request.model == ModelTier.FAST
        print("✓ Pydantic model creation works")

        # Test validation
        try:
            invalid_request = DSMILQueryRequest(
                prompt="",  # Invalid: too short
                model=ModelTier.FAST
            )
            print("✗ Validation should have failed for empty prompt")
            sys.exit(1)
        except Exception:
            print("✓ Pydantic validation works")

    except Exception as e:
        print(f"✗ Pydantic model test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*70)
print("All import and syntax tests passed!")
print("="*70)
print("\nNote: Full functionality tests require Ollama running.")
print("Run: python3 test_dual_mode.py (when Ollama is available)")
