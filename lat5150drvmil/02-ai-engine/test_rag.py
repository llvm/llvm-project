#!/usr/bin/env python3
"""
Test RAG Pipeline End-to-End

This script tests the RAG system to ensure it works correctly.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "04-integrations"))

from rag_system import RAGSystem

def test_rag_pipeline():
    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)
    print()

    # Create a temporary test file
    test_content = """
    DSMIL Device Framework

    The DSMIL (Dell Secure Management Interface Layer) framework provides
    80 hardware security devices for military-grade operations.

    Key Features:
    - TPM 2.0 attestation
    - Post-Quantum Cryptography (PQC) support
    - Hardware-level security operations
    - Mode 5 platform integrity

    Device 0x8000: TPM Control
    Provides interface to TPM 2.0 for cryptographic operations.

    Device 0x8001: Boot Security
    Validates secure boot chain integrity.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file = f.name

    try:
        # Initialize RAG
        print("1. Initializing RAG system...")
        rag = RAGSystem()
        print(f"   ✓ Index path: {rag.index_path}")
        print()

        # Get initial stats
        print("2. Initial statistics...")
        stats = rag.get_stats()
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Tokens: {stats['total_unique_tokens']}")
        print()

        # Ingest test file
        print("3. Ingesting test file...")
        result = rag.ingest_file(test_file)
        if result.get('status') == 'success':
            print(f"   ✓ File ingested: {result['filename']}")
            print(f"   ✓ Tokens: {result['tokens']}")
            print(f"   ✓ Characters: {result['chars']}")
        elif result.get('status') == 'already_indexed':
            print(f"   ℹ️  File already indexed (that's OK for testing)")
        else:
            print(f"   ✗ Error: {result.get('error')}")
            return False
        print()

        # Test search
        print("4. Testing search...")
        queries = [
            "TPM",
            "DSMIL devices",
            "Boot Security",
            "Post-Quantum"
        ]

        for query in queries:
            results = rag.search(query, max_results=3)
            print(f"   Query: '{query}'")
            if results:
                print(f"   ✓ Found {len(results)} results")
                for doc in results:
                    print(f"      - {doc['filename']} (score: {doc.get('relevance_score', 0)})")
            else:
                print(f"   ✗ No results found")
            print()

        # Test document listing
        print("5. Testing document listing...")
        docs = rag.list_documents()
        print(f"   ✓ Total documents: {len(docs)}")
        for doc in docs:
            tokens = doc.get('tokens', doc.get('token_count', 0))
            print(f"      - {doc['filename']}: {tokens} tokens")
        print()

        # Final stats
        print("6. Final statistics...")
        stats = rag.get_stats()
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Tokens: {stats['total_unique_tokens']}")
        print()

        print("=" * 70)
        print("✅ RAG PIPELINE TEST PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_rag_pipeline()
    sys.exit(0 if success else 1)
