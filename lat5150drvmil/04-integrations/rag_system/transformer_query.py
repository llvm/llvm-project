#!/usr/bin/env python3
"""
Transformer-based RAG Query Interface
Enhanced semantic search using HuggingFace transformers
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict

try:
    from transformer_upgrade import TransformerRetriever
except ImportError:
    print("❌ transformer_upgrade.py not found or dependencies missing!")
    print("\nInstall dependencies:")
    print("  pip install sentence-transformers")
    sys.exit(1)


class TransformerRAG:
    """RAG interface using transformer embeddings"""

    def __init__(self, data_path='rag_system/processed_docs.json'):
        """Initialize transformer-based RAG"""
        print("Loading Transformer RAG system...")

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data not found: {data_path}")

        with open(data_path, 'r') as f:
            data = json.load(f)

        self.chunks = data['chunks']
        self.stats = data['stats']

        # Try to load pre-computed embeddings
        embeddings_path = 'rag_system/transformer_embeddings.npz'

        if Path(embeddings_path).exists():
            print("Loading pre-computed embeddings...")
            self.retriever = TransformerRetriever.load_embeddings(
                self.chunks,
                embeddings_path
            )
        else:
            print("No pre-computed embeddings found. Building from scratch...")
            print("(This will take 1-2 minutes on first run)")
            self.retriever = TransformerRetriever(self.chunks)
            self.retriever.save_embeddings()

        print("✓ Transformer RAG ready!\n")

    def query(self, question: str, top_k: int = 3, show_sources: bool = True) -> str:
        """
        Semantic query using transformers

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            show_sources: Show source files

        Returns:
            Formatted response with context
        """
        # Retrieve semantically similar chunks
        results = self.retriever.search(question, top_k=top_k)

        # Format response
        response = []
        response.append(f"Query: {question}\n")
        response.append("="*60)

        # Combine context
        context_parts = []
        sources = set()

        for i, (chunk, score) in enumerate(results, 1):
            if score > 0.3:  # Semantic similarity threshold
                context_parts.append(chunk['text'])
                sources.add(chunk['metadata']['filepath'])

        if not context_parts:
            response.append("\nNo relevant information found.")
            return '\n'.join(response)

        # Display relevant information
        response.append("\nRelevant Information:")
        response.append("-"*60)

        for i, part in enumerate(context_parts, 1):
            response.append(f"\n[Context {i}]")
            response.append(part[:500])
            if len(part) > 500:
                response.append("...")

        if show_sources:
            response.append("\n" + "="*60)
            response.append("Sources:")
            for source in sorted(sources):
                response.append(f"  📄 {source}")

        return '\n'.join(response)

    def compare_with_tfidf(self, query: str):
        """Compare transformer vs TF-IDF results"""
        from document_processor import TFIDFRetriever

        print("Comparing Transformer vs TF-IDF:\n")
        print("="*70)
        print(f"Query: {query}")
        print("="*70)

        # Transformer results
        print("\n🤖 TRANSFORMER RESULTS:")
        print("-"*70)
        transformer_results = self.retriever.search(query, top_k=3)
        for i, (chunk, score) in enumerate(transformer_results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   File: {chunk['metadata']['filepath']}")
            print(f"   Text: {chunk['text'][:150]}...")

        # TF-IDF results
        print("\n\n📊 TF-IDF RESULTS (old):")
        print("-"*70)
        tfidf = TFIDFRetriever(self.chunks)
        tfidf_results = tfidf.search(query, top_k=3)
        for i, (chunk, score) in enumerate(tfidf_results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   File: {chunk['metadata']['filepath']}")
            print(f"   Text: {chunk['text'][:150]}...")

        print("\n" + "="*70)


def interactive_mode(rag: TransformerRAG, collect_feedback: bool = True):
    """Interactive query mode with optional feedback collection"""
    # Initialize feedback collector
    if collect_feedback:
        try:
            from feedback_collector import FeedbackCollector
            feedback = FeedbackCollector()
            print("\n💡 Feedback collection enabled! You'll be asked to rate results (1-10).")
        except ImportError:
            print("\n⚠️  feedback_collector.py not found. Feedback disabled.")
            collect_feedback = False
            feedback = None
    else:
        feedback = None

    print("\n" + "="*60)
    print("LAT5150DRVMIL Transformer RAG - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question for semantic search")
    print("  - 'compare <query>' - Compare with TF-IDF")
    print("  - 'stats' - Show feedback statistics")
    print("  - 'quit' or 'exit' - Exit")
    print("\n" + "="*60 + "\n")

    while True:
        try:
            user_input = input("Transformer-RAG> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                if collect_feedback and feedback:
                    print()
                    feedback.print_statistics()
                print("Goodbye!")
                break

            elif user_input.lower() == 'stats':
                if collect_feedback and feedback:
                    feedback.print_statistics()
                else:
                    print("Feedback collection not enabled.")
                print()
                continue

            elif user_input.lower().startswith('compare '):
                query = user_input[8:].strip()
                rag.compare_with_tfidf(query)
                print()

            else:
                # Get query results
                search_results = rag.retriever.search(user_input, top_k=3)
                answer = rag.query(user_input)
                print(f"\n{answer}\n")

                # Collect feedback
                if collect_feedback and feedback:
                    rating, auto_comment = feedback._prompt_for_rating()

                    if rating is not None:
                        explanation = auto_comment or ""

                        # If no auto-comment, ask for explanation
                        if not auto_comment:
                            print()
                            print("💭 Why did you rate it this way?")
                            print("   (Optional - helps improve the system)")
                            print("   Press Enter to skip")
                            print()
                            try:
                                user_explanation = input("Reason: ").strip()
                                if user_explanation:
                                    explanation = user_explanation
                            except KeyboardInterrupt:
                                pass

                        # Save feedback
                        feedback.collect_rating(
                            query=user_input,
                            results=search_results,
                            rating=rating,
                            comment=explanation,
                            auto_prompt=False
                        )

                        print("\n✓ Feedback saved. Thank you!\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            if collect_feedback and feedback:
                print()
                feedback.print_statistics()
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='LAT5150DRVMIL Transformer RAG Query'
    )
    parser.add_argument(
        'query',
        nargs='*',
        help='Query string'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with TF-IDF results'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of results (default: 3)'
    )
    parser.add_argument(
        '--no-feedback',
        action='store_true',
        help='Disable feedback collection'
    )

    args = parser.parse_args()

    # Initialize
    try:
        rag = TransformerRAG()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Process query
    if args.query:
        query_string = ' '.join(args.query)

        if args.compare:
            rag.compare_with_tfidf(query_string)
        else:
            answer = rag.query(query_string, top_k=args.top_k)
            print(answer)
    else:
        interactive_mode(rag, collect_feedback=not args.no_feedback)


if __name__ == '__main__':
    main()
