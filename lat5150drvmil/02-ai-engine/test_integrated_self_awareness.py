#!/usr/bin/env python3
"""
Comprehensive tests for Integrated Self-Awareness Engine

Tests:
- Component integration (vector DB, cognitive memory, DSMIL, TPM, crypto, MCP)
- Persistent state management
- Knowledge integration with cognitive memory
- Capability discovery and tracking
- API compatibility
"""

import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from integrated_self_awareness import (
    IntegratedSelfAwarenessEngine,
    SystemComponentType,
    IntegratedCapability,
    SystemState
)

# ANSI colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def test_initialization():
    """Test engine initialization"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 1: Engine Initialization{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        # Create temp directories
        with tempfile.TemporaryDirectory() as tmpdir:
            state_db = Path(tmpdir) / "test_state.db"
            vector_db = Path(tmpdir) / "vectordb"

            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(state_db),
                vector_db_path=str(vector_db),
                postgres_url=None  # Skip PostgreSQL in test
            )

            print(f"{GREEN}✓{NC} Engine initialized")
            print(f"  Components discovered: {len(engine.components)}")
            print(f"  Capabilities discovered: {len(engine.capabilities)}")

            # Check state database exists
            if state_db.exists():
                print(f"{GREEN}✓{NC} State database created")
            else:
                print(f"{RED}✗{NC} State database not created")
                return False

            return True

    except Exception as e:
        print(f"{RED}✗{NC} Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_discovery():
    """Test component discovery and registration"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 2: Component Discovery{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(Path(tmpdir) / "test.db"),
                vector_db_path=str(Path(tmpdir) / "vectordb"),
                postgres_url=None
            )

            print(f"\n{YELLOW}Components by type:{NC}")
            for comp_type in SystemComponentType:
                count = sum(1 for c in engine.components.values()
                           if c.component_type == comp_type)
                if count > 0:
                    print(f"  {comp_type.value}: {count}")

            # Test component status
            active_count = sum(1 for c in engine.components.values()
                              if c.status == "active")
            print(f"\n{GREEN}✓{NC} Active components: {active_count}/{len(engine.components)}")

            return True

    except Exception as e:
        print(f"{RED}✗{NC} Component discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_capability_discovery():
    """Test integrated capability discovery"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 3: Integrated Capability Discovery{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(Path(tmpdir) / "test.db"),
                vector_db_path=str(Path(tmpdir) / "vectordb"),
                postgres_url=None
            )

            print(f"\n{YELLOW}Discovered capabilities:{NC}")
            for cap_id, cap in engine.capabilities.items():
                status = f"{GREEN}✓{NC}" if cap.confidence >= 0.8 else f"{YELLOW}~{NC}"
                print(f"  {status} {cap.name}")
                print(f"      Confidence: {cap.confidence:.1%}")
                print(f"      Required: {[t.value for t in cap.required_components]}")

            fully_available = sum(1 for c in engine.capabilities.values()
                                 if c.confidence >= 1.0)
            print(f"\n{GREEN}✓{NC} Fully available capabilities: {fully_available}/{len(engine.capabilities)}")

            return True

    except Exception as e:
        print(f"{RED}✗{NC} Capability discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_persistence():
    """Test persistent state management"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 4: State Persistence{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_db = Path(tmpdir) / "test.db"

            # Create engine and save state
            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(state_db),
                vector_db_path=str(Path(tmpdir) / "vectordb"),
                postgres_url=None
            )

            # Save multiple snapshots
            engine.save_state_snapshot()
            print(f"{GREEN}✓{NC} State snapshot saved")

            # Track capability usage
            if engine.capabilities:
                cap_id = list(engine.capabilities.keys())[0]
                engine.track_capability_usage(
                    capability_id=cap_id,
                    execution_time_ms=123.45,
                    success=True,
                    components_used=["test_comp"],
                    error_message=None
                )
                print(f"{GREEN}✓{NC} Capability usage tracked")

            # Record learning event
            engine.record_learning_event(
                event_type="test_event",
                context="Testing state persistence",
                insight="State persistence works correctly",
                confidence=0.95
            )
            print(f"{GREEN}✓{NC} Learning event recorded")

            # Load historical state
            history = engine.load_historical_state(hours_back=24)
            print(f"{GREEN}✓{NC} Historical state loaded: {len(history)} snapshots")

            return True

    except Exception as e:
        print(f"{RED}✗{NC} State persistence failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_integration():
    """Test knowledge integration with cognitive memory"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 5: Knowledge Integration (Cognitive Memory){NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(Path(tmpdir) / "test.db"),
                vector_db_path=str(Path(tmpdir) / "vectordb"),
                postgres_url=None  # Would use real PostgreSQL in production
            )

            # Check if cognitive memory is available
            if hasattr(engine, 'cognitive_memory'):
                print(f"{GREEN}✓{NC} Cognitive memory system available")

                # Try to integrate knowledge
                engine.integrate_knowledge_into_cognitive_memory()
                print(f"{GREEN}✓{NC} Knowledge integrated into cognitive memory")

                # Try to query
                results = engine.query_system_knowledge("vector database")
                print(f"{GREEN}✓{NC} Knowledge query executed: {len(results)} results")

            else:
                print(f"{YELLOW}⊘{NC} Cognitive memory not available (PostgreSQL required)")
                print(f"  This is expected in test environment")

            return True

    except Exception as e:
        print(f"{RED}✗{NC} Knowledge integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_compatibility():
    """Test API compatibility methods"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 6: API Compatibility{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(Path(tmpdir) / "test.db"),
                vector_db_path=str(Path(tmpdir) / "vectordb"),
                postgres_url=None
            )

            # Test discover_capabilities()
            capabilities = engine.discover_capabilities()
            print(f"{GREEN}✓{NC} discover_capabilities(): {len(capabilities)} capabilities")

            # Test discover_resources()
            resources = engine.discover_resources()
            print(f"{GREEN}✓{NC} discover_resources():")
            for key, value in resources.items():
                count = len(value) if isinstance(value, list) else 0
                if count > 0:
                    print(f"      {key}: {count}")

            # Test update_system_state()
            engine.update_system_state()
            print(f"{GREEN}✓{NC} update_system_state(): executed")

            # Test get_comprehensive_report()
            report = engine.get_comprehensive_report()
            print(f"{GREEN}✓{NC} get_comprehensive_report():")
            print(f"      system_name: {report.get('system_name', 'N/A')}")
            print(f"      self_awareness_level: {report.get('self_awareness_level', 'N/A')}")

            return True

    except Exception as e:
        print(f"{RED}✗{NC} API compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_report():
    """Test comprehensive report generation"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST 7: Comprehensive Report{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = IntegratedSelfAwarenessEngine(
                workspace_path="/home/user/LAT5150DRVMIL",
                state_db_path=str(Path(tmpdir) / "test.db"),
                vector_db_path=str(Path(tmpdir) / "vectordb"),
                postgres_url=None
            )

            report = engine.get_comprehensive_report()

            # Validate report structure
            required_keys = [
                "system_name",
                "self_awareness_level",
                "timestamp",
                "integrated_components",
                "integrated_capabilities",
                "system_state"
            ]

            for key in required_keys:
                if key in report:
                    print(f"{GREEN}✓{NC} Report contains '{key}'")
                else:
                    print(f"{RED}✗{NC} Report missing '{key}'")
                    return False

            # Print summary
            print(f"\n{YELLOW}Report Summary:{NC}")
            print(f"  System: {report['system_name']}")
            print(f"  Awareness Level: {report['self_awareness_level']}")
            print(f"  Components: {report['integrated_components']['total']}")
            print(f"  Capabilities: {report['integrated_capabilities']['total']}")

            return True

    except Exception as e:
        print(f"{RED}✗{NC} Comprehensive report test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}INTEGRATED SELF-AWARENESS ENGINE - TEST SUITE{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    tests = [
        ("Initialization", test_initialization),
        ("Component Discovery", test_component_discovery),
        ("Capability Discovery", test_capability_discovery),
        ("State Persistence", test_state_persistence),
        ("Knowledge Integration", test_knowledge_integration),
        ("API Compatibility", test_api_compatibility),
        ("Comprehensive Report", test_comprehensive_report),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{RED}✗{NC} {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{GREEN}TEST SUMMARY{NC}")
    print(f"{BLUE}{'='*70}{NC}\n")

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for test_name, result in results:
        status = f"{GREEN}✓ PASS{NC}" if result else f"{RED}✗ FAIL{NC}"
        print(f"  {status}  {test_name}")

    print(f"\n{BLUE}{'='*70}{NC}")
    if failed == 0:
        print(f"{GREEN}ALL TESTS PASSED ({passed}/{len(results)}){NC}")
    else:
        print(f"{YELLOW}SOME TESTS FAILED ({passed} passed, {failed} failed){NC}")
    print(f"{BLUE}{'='*70}{NC}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
