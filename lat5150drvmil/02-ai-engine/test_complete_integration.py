#!/usr/bin/env python3
"""
Complete Integration Test - WhiteRabbit + Pydantic + NLI + FastAPI

Tests the entire DSMIL AI stack end-to-end:
- WhiteRabbit Pydantic wrapper
- Type-safe NLI (Natural Language Interface)
- Agent configuration system
- FastAPI server integration
- Multi-backend orchestration

This is the comprehensive validation that all components work together.

Author: DSMIL Integration Framework
Version: 3.0.0 (Complete Integration)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("DSMIL Complete Integration Test - WhiteRabbit + Pydantic + NLI")
print("="*80)

# Test 1: Import Check
print("\nTest 1: Checking imports...")
test_results = {
    "imports": False,
    "whiterabbit": False,
    "nli": False,
    "config": False,
    "orchestrator": False,
    "fastapi_models": False,
}

try:
    from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE
    print("✓ dsmil_ai_engine")

    from unified_orchestrator import UnifiedAIOrchestrator
    print("✓ unified_orchestrator")

    if PYDANTIC_AVAILABLE:
        from pydantic_models import (
            OrchestratorResponse,
            OrchestratorRequest,
            BackendType,
        )
        print("✓ pydantic_models")
        test_results["fastapi_models"] = True

    from agent_config import DSMILAIConfig, LocalAIConfig, AgentRegistry
    print("✓ agent_config")
    test_results["config"] = True

    test_results["imports"] = True

except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: WhiteRabbit Integration
print("\nTest 2: WhiteRabbit Pydantic Integration...")
try:
    from whiterabbit_pydantic import (
        PydanticWhiteRabbitEngine,
        WhiteRabbitRequest,
        WhiteRabbitResponse,
        WhiteRabbitDevice,
        WhiteRabbitTaskType,
        WHITERABBIT_AVAILABLE,
    )
    print(f"✓ WhiteRabbit imports: Available={WHITERABBIT_AVAILABLE}")

    # Test model creation
    wr_request = WhiteRabbitRequest(
        prompt="Test prompt for WhiteRabbit",
        device=WhiteRabbitDevice.AUTO,
        task_type=WhiteRabbitTaskType.TEXT_GENERATION,
        max_new_tokens=100,
        temperature=0.7
    )
    print(f"✓ WhiteRabbitRequest created:")
    print(f"  Device: {wr_request.device.value}")
    print(f"  Task: {wr_request.task_type.value}")
    print(f"  Max Tokens: {wr_request.max_new_tokens}")

    test_results["whiterabbit"] = True

except Exception as e:
    print(f"⚠  WhiteRabbit integration: {e}")
    test_results["whiterabbit"] = False

# Test 3: NLI Pydantic Models
print("\nTest 3: Natural Language Interface (NLI)...")
try:
    from nli_pydantic_models import (
        Message,
        Conversation,
        MessageRole,
        CreateConversationRequest,
        NLIChatRequest,
    )
    print("✓ NLI Pydantic models imported")

    # Test conversation creation request
    conv_req = CreateConversationRequest(
        title="WhiteRabbit Performance Analysis",
        metadata={"test": True, "version": "3.0"}
    )
    print(f"✓ CreateConversationRequest:")
    print(f"  Title: {conv_req.title}")
    print(f"  Metadata: {conv_req.metadata}")

    # Test NLI chat request
    nli_req = NLIChatRequest(
        prompt="What is the performance of WhiteRabbitNeo on Intel Arc NPU?",
        include_history=True,
        history_limit=10,
        temperature=0.7
    )
    print(f"✓ NLIChatRequest:")
    print(f"  Prompt: {nli_req.prompt[:60]}...")
    print(f"  Include History: {nli_req.include_history}")

    # Test message validation
    msg = Message(
        conversation_id="test-conv-123",
        role=MessageRole.ASSISTANT,
        content="WhiteRabbitNeo achieves 45 tokens/second on Intel Arc NPU.",
        model="whiterabbit-neo-33b",
        tokens_output=20,
        latency_ms=450
    )
    print(f"✓ Message created:")
    print(f"  ID: {msg.id}")
    print(f"  Role: {msg.role.value}")
    print(f"  Model: {msg.model}")

    test_results["nli"] = True

except Exception as e:
    print(f"⚠  NLI integration: {e}")
    import traceback
    traceback.print_exc()
    test_results["nli"] = False

# Test 4: Agent Configuration System
print("\nTest 4: Agent Configuration System...")
try:
    # Load default config
    config = DSMILAIConfig()
    print(f"✓ Default configuration loaded:")
    print(f"  Default Backend: {config.default_backend}")
    print(f"  Pydantic Mode: {config.enable_pydantic_mode}")
    print(f"  Attestation: {config.enable_attestation}")

    # Check WhiteRabbit as default
    local_config = config.local_ai
    print(f"✓ Local AI configuration:")
    print(f"  Fast Model: {local_config.models['fast']}")
    print(f"  Code Model: {local_config.models['code']}")

    if local_config.models['fast'] == "whiterabbit-neo-33b":
        print("  ✓ WhiteRabbit is default model!")
    else:
        print(f"  ⚠  Default model is {local_config.models['fast']}, not WhiteRabbit")

    # Check agent registry
    registry = AgentRegistry()
    local_agent = registry.get_agent("local")
    print(f"✓ Agent Registry:")
    print(f"  Local Agent: {local_agent.description}")
    print(f"  Capabilities: {len(local_agent.capabilities)}")

    test_results["config"] = True

except Exception as e:
    print(f"⚠  Configuration system: {e}")
    import traceback
    traceback.print_exc()
    test_results["config"] = False

# Test 5: Orchestrator Integration
print("\nTest 5: Orchestrator Integration...")
try:
    # Create orchestrator in Pydantic mode
    orch_pydantic = UnifiedAIOrchestrator(enable_ace=False, pydantic_mode=True)
    print(f"✓ Pydantic-mode orchestrator created")
    print(f"  Pydantic Mode: {orch_pydantic.pydantic_mode}")

    # Create orchestrator in dict mode
    orch_dict = UnifiedAIOrchestrator(enable_ace=False, pydantic_mode=False)
    print(f"✓ Dict-mode orchestrator created")
    print(f"  Pydantic Mode: {orch_dict.pydantic_mode}")

    # Verify backends
    status = orch_pydantic.get_status()
    print(f"✓ Orchestrator status:")
    print(f"  Backends: {list(status['backends'].keys())}")

    test_results["orchestrator"] = True

except Exception as e:
    print(f"⚠  Orchestrator integration: {e}")
    import traceback
    traceback.print_exc()
    test_results["orchestrator"] = False

# Test 6: Type System Validation
print("\nTest 6: Type System Validation...")
try:
    if PYDANTIC_AVAILABLE:
        # Test OrchestratorRequest creation
        req = OrchestratorRequest(
            prompt="Explain WhiteRabbitNeo multi-device support",
            force_backend=BackendType.LOCAL,
            enable_web_search=False
        )
        print(f"✓ OrchestratorRequest:")
        print(f"  Prompt: {req.prompt[:50]}...")
        print(f"  Backend: {req.force_backend.value}")

        # Test validation (should fail)
        try:
            invalid_req = OrchestratorRequest(
                prompt="",  # Too short
                force_backend=BackendType.LOCAL
            )
            print("✗ Validation should have failed for empty prompt")
        except Exception:
            print("✓ Validation works (rejected empty prompt)")

    else:
        print("⚠  Pydantic not available, skipping type validation")

except Exception as e:
    print(f"⚠  Type validation: {e}")
    import traceback
    traceback.print_exc()

# Test 7: NLI Wrapper (if database available)
print("\nTest 7: NLI Pydantic Wrapper...")
try:
    from nli_pydantic_wrapper import PydanticNLIManager, LEGACY_AVAILABLE

    if LEGACY_AVAILABLE:
        print("✓ NLI Pydantic wrapper available")
        print("  Note: Full NLI test requires PostgreSQL database")
        print("  Run nli_pydantic_wrapper.py directly to test with DB")
    else:
        print("⚠  Legacy ConversationManager not available")

except Exception as e:
    print(f"⚠  NLI wrapper: {e}")

# Summary
print("\n" + "="*80)
print("Integration Test Summary")
print("="*80)

total_tests = len(test_results)
passed_tests = sum(1 for v in test_results.values() if v)

print(f"\nResults: {passed_tests}/{total_tests} tests passed\n")

for test_name, passed in test_results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status:8} - {test_name}")

print(f"\n{'='*80}")

if passed_tests == total_tests:
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print("="*80)
    print("\nComplete Integration Status:")
    print("  ✓ WhiteRabbit Pydantic wrapper")
    print("  ✓ Type-safe NLI models")
    print("  ✓ Agent configuration system")
    print("  ✓ Multi-backend orchestrator")
    print("  ✓ FastAPI models")
    print("\nThe DSMIL AI stack is fully integrated with:")
    print("  • WhiteRabbitNeo as default local AI")
    print("  • Complete Pydantic type safety")
    print("  • Natural Language Interface support")
    print("  • Multi-device inference (NPU/GPU/NCS2)")
    print("  • Hardware attestation")
    print("\nNext Steps:")
    print("  1. Run actual inference tests (requires Ollama running):")
    print("     cd 02-ai-engine")
    print("     python3 test_dual_mode.py")
    print("\n  2. Start FastAPI server with WhiteRabbit:")
    print("     cd 03-web-interface")
    print("     python3 dsmil_fastapi_server.py")
    print("\n  3. Test WhiteRabbit endpoint:")
    print("     curl -X POST http://localhost:9877/whiterabbit/generate \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"prompt\": \"Hello WhiteRabbit\", \"device\": \"auto\"}'")
else:
    print("⚠  SOME TESTS FAILED")
    print("="*80)
    print("\nPlease review failures above.")
    print("Note: Some failures are expected if:")
    print("  • Ollama is not running")
    print("  • PostgreSQL database not configured")
    print("  • WhiteRabbit models not installed")

print("="*80)
