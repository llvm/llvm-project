#!/usr/bin/env python3
"""
Open R1 Enhanced RAG Query System
Combines semantic retrieval with chain-of-thought reasoning

This script integrates:
1. Transformer-based semantic retrieval (BAAI/bge-base-en-v1.5)
2. Open R1 reasoning model for step-by-step explanations
3. Optional feedback collection for continuous improvement

Usage:
    python3 openr1_query.py "your question here"
    python3 openr1_query.py --interactive
    python3 openr1_query.py --test
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system.transformer_upgrade import TransformerRetriever
from rag_system.openr1_reasoning import OpenR1Reasoner


class OpenR1RAG:
    """
    Enhanced RAG system with Open R1 reasoning

    Combines:
    - Semantic document retrieval
    - Chain-of-thought reasoning
    - Structured output formatting
    """

    def __init__(self, processed_docs_path: str = 'rag_system/processed_docs.json'):
        """
        Initialize Open R1 RAG system

        Args:
            processed_docs_path: Path to processed documents JSON
        """
        print("🔧 Initializing Open R1 Enhanced RAG System...")
        print()

        # Load documents
        print("[1/2] Loading transformer retriever...")
        start = time.time()
        with open(processed_docs_path, 'r') as f:
            data = json.load(f)

        self.retriever = TransformerRetriever(data['chunks'])
        retriever_time = time.time() - start
        print(f"      ✓ Loaded {len(data['chunks'])} chunks in {retriever_time:.2f}s")
        print()

        # Load Open R1
        print("[2/2] Loading Open R1 reasoning model...")
        print("      (First run: downloading model ~3.5GB with INT4 quantization)")
        start = time.time()
        self.reasoner = OpenR1Reasoner(quantize_model=True, use_ipex=True)
        reasoner_time = time.time() - start
        print(f"      ✓ Model loaded in {reasoner_time:.2f}s")
        print()

        # Get model info
        info = self.reasoner.get_model_info()
        print("📊 Model Configuration:")
        print(f"   Model: {info['model_name']}")
        print(f"   Parameters: {info['parameters']}")
        print(f"   Size: {info['model_size_mb']}")
        print(f"   Quantization: {'INT4' if info['quantized'] else 'FP32'}")
        print(f"   Intel IPEX: {'Enabled' if info['ipex_enabled'] else 'Disabled'}")
        print()

        print("✅ System ready!")
        print("=" * 70)
        print()

    def query(self, question: str, top_k: int = 3, show_sources: bool = True) -> Dict:
        """
        Query RAG system with Open R1 reasoning

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            show_sources: Display source documents

        Returns:
            Dict with reasoning trace, answer, and metadata
        """
        print(f"🔍 Query: {question}")
        print()

        # Step 1: Semantic retrieval
        print(f"[1/2] Retrieving top {top_k} relevant documents...")
        start = time.time()
        retrieved_docs = self.retriever.search(question, top_k=top_k)
        retrieval_time = time.time() - start

        print(f"      ✓ Retrieved {len(retrieved_docs)} documents in {retrieval_time:.3f}s")

        if show_sources:
            print()
            print("📚 Retrieved Sources:")
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                source = doc.get('source', 'Unknown')
                chunk_preview = doc.get('chunk', '')[:100].replace('\n', ' ')
                print(f"   [{i}] {source} (relevance: {score:.1%})")
                print(f"       {chunk_preview}...")
            print()

        # Step 2: Open R1 reasoning
        print("[2/2] Generating reasoned answer with Open R1...")
        start = time.time()
        result = self.reasoner.reason(question, retrieved_docs)
        reasoning_time = time.time() - start

        print(f"      ✓ Answer generated in {reasoning_time:.2f}s")
        print()

        # Add timing information
        result['retrieval_time'] = retrieval_time
        result['reasoning_time'] = reasoning_time
        result['total_time'] = retrieval_time + reasoning_time

        return result

    def display_result(self, result: Dict):
        """
        Display formatted result with reasoning trace

        Args:
            result: Query result from self.query()
        """
        print("=" * 70)
        print("🧠 REASONING TRACE")
        print("=" * 70)
        print()
        print(result['reasoning'])
        print()
        print("=" * 70)
        print("✨ FINAL ANSWER")
        print("=" * 70)
        print()
        print(result['answer'])
        print()
        print("=" * 70)
        print("⏱️  Performance")
        print("=" * 70)
        print(f"Retrieval:  {result['retrieval_time']:.3f}s")
        print(f"Reasoning:  {result['reasoning_time']:.2f}s")
        print(f"Total:      {result['total_time']:.2f}s")
        print(f"Documents:  {result['num_docs']}")
        print()


def interactive_mode():
    """Interactive query mode with Open R1 reasoning"""
    print()
    print("=" * 70)
    print("  Open R1 Enhanced RAG - Interactive Mode")
    print("=" * 70)
    print()

    # Initialize system
    rag = OpenR1RAG()

    print("🎯 Enter your questions (type 'exit' to quit, 'help' for commands)")
    print()

    while True:
        try:
            user_input = input("Query> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'q']:
                print()
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'help':
                print()
                print("Commands:")
                print("  exit       - Exit interactive mode")
                print("  help       - Show this help message")
                print("  info       - Show model information")
                print("  <query>    - Ask a question")
                print()
                continue

            if user_input.lower() == 'info':
                info = rag.reasoner.get_model_info()
                print()
                for key, value in info.items():
                    print(f"  {key}: {value}")
                print()
                continue

            # Process query
            print()
            result = rag.query(user_input)
            rag.display_result(result)

        except KeyboardInterrupt:
            print()
            print()
            print("👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def test_mode():
    """Test Open R1 with sample queries"""
    print()
    print("=" * 70)
    print("  Open R1 Enhanced RAG - Test Mode")
    print("=" * 70)
    print()

    # Test queries
    test_queries = [
        "What is the DSMIL AI system?",
        "How does the LAT5150DRVMIL handle encryption?",
        "What are the main security features?"
    ]

    # Initialize system
    rag = OpenR1RAG()

    for i, query in enumerate(test_queries, 1):
        print()
        print(f"{'=' * 70}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'=' * 70}")
        print()

        result = rag.query(query, show_sources=False)
        rag.display_result(result)

        if i < len(test_queries):
            print()
            print("─" * 70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Open R1 Enhanced RAG Query System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 openr1_query.py --interactive

  # Single query
  python3 openr1_query.py "What is DSMIL?"

  # Run test queries
  python3 openr1_query.py --test

  # Query with detailed sources
  python3 openr1_query.py --sources "How does encryption work?"
        """
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Question to ask the RAG system'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive query mode'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test queries'
    )
    parser.add_argument(
        '--sources',
        action='store_true',
        help='Show retrieved source documents'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of documents to retrieve (default: 3)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.interactive:
        interactive_mode()
    elif args.test:
        test_mode()
    elif args.query:
        # Single query mode
        print()
        print("=" * 70)
        print("  Open R1 Enhanced RAG")
        print("=" * 70)
        print()

        rag = OpenR1RAG()
        result = rag.query(args.query, top_k=args.top_k, show_sources=args.sources)
        rag.display_result(result)
    else:
        parser.print_help()
        print()
        print("💡 Tip: Try --interactive mode for continuous querying")


if __name__ == '__main__':
    main()
