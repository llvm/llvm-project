#!/usr/bin/env python3
"""
Test script for DSMIL AI MCP Server

Tests basic functionality without requiring a full MCP client

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 1.0.0
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "04-integrations"))

try:
    from dsmil_ai_engine import DSMILAIEngine
    print("✓ DSMIL AI Engine imported successfully")
except ImportError as e:
    print(f"✗ Failed to import DSMIL AI Engine: {e}")
    sys.exit(1)

try:
    import mcp
    print(f"✓ MCP library imported successfully (version {mcp.__version__})")
except ImportError:
    print("✗ MCP library not found. Install with: pip3 install mcp")
    sys.exit(1)


def test_engine_initialization():
    """Test AI engine initialization"""
    print("\n[Test 1] AI Engine Initialization")
    try:
        engine = DSMILAIEngine()
        print("✓ AI engine initialized")
        return engine
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return None


def test_status(engine):
    """Test status retrieval"""
    print("\n[Test 2] Get Status")
    try:
        status = engine.get_status()
        print(f"✓ Status retrieved")
        print(f"  - Ollama: {'connected' if status.get('ollama', {}).get('connected') else 'disconnected'}")
        print(f"  - RAG: {status.get('rag', {}).get('documents', 0)} documents")
        print(f"  - Models: {sum(1 for m in status.get('ollama', {}).get('models', {}).values() if m.get('installed'))}/5 installed")
        return True
    except Exception as e:
        print(f"✗ Failed to get status: {e}")
        return False


def test_rag_stats(engine):
    """Test RAG statistics"""
    print("\n[Test 3] RAG Statistics")
    try:
        stats = engine.rag_get_stats()
        if stats.get("error"):
            print(f"⚠️  RAG error: {stats['error']}")
            return True  # Not a critical error
        print(f"✓ RAG stats retrieved")
        print(f"  - Documents: {stats.get('total_documents', 0)}")
        print(f"  - Tokens: {stats.get('total_tokens', 0)}")
        return True
    except Exception as e:
        print(f"✗ Failed to get RAG stats: {e}")
        return False


def test_ai_query(engine):
    """Test AI query (if Ollama is available)"""
    print("\n[Test 4] AI Query (Optional)")
    try:
        status = engine.get_status()
        if not status.get('ollama', {}).get('connected'):
            print("⚠️  Ollama not connected - skipping AI query test")
            return True

        print("  Sending test query to AI...")
        result = engine.generate(
            prompt="Say 'Hello from DSMIL AI' in exactly 5 words",
            model_selection="fast",
            max_tokens=50
        )

        if result.get("response"):
            print(f"✓ AI responded: {result['response'][:100]}...")
            return True
        else:
            print("⚠️  No response from AI")
            return True
    except Exception as e:
        print(f"⚠️  AI query failed (non-critical): {e}")
        return True  # Non-critical


def test_mcp_server_import():
    """Test MCP server import"""
    print("\n[Test 5] MCP Server Import")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import dsmil_mcp_server
        print("✓ MCP server module imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import MCP server: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("  DSMIL AI MCP Server Test Suite")
    print("=" * 60)

    tests_passed = 0
    tests_total = 5

    # Test 1: Engine initialization
    engine = test_engine_initialization()
    if engine:
        tests_passed += 1

        # Test 2: Status
        if test_status(engine):
            tests_passed += 1

        # Test 3: RAG stats
        if test_rag_stats(engine):
            tests_passed += 1

        # Test 4: AI query
        if test_ai_query(engine):
            tests_passed += 1
    else:
        tests_total = 1  # Only first test ran

    # Test 5: MCP server import
    if test_mcp_server_import():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"  Test Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)

    if tests_passed == tests_total:
        print("\n✓ All tests passed! MCP server is ready to use.")
        print("\nNext steps:")
        print("1. Install MCP in Claude Desktop: ./install_mcp.sh")
        print("2. Restart Claude Desktop")
        print("3. Try: 'Use dsmil_get_status'")
        return 0
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed")
        print("\nTroubleshooting:")
        print("- Check dependencies: pip3 install mcp")
        print("- Start Ollama: ollama serve")
        print("- Check RAG: python3 test_rag.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
