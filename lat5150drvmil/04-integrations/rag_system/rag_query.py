#!/usr/bin/env python3
"""
RAG Query Interface for LAT5150DRVMIL
Simple, fast documentation search and retrieval
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from document_processor import TFIDFRetriever


class LAT5150RAG:
    """Main RAG interface for LAT5150DRVMIL documentation"""

    def __init__(self, data_path='rag_system/processed_docs.json'):
        """Initialize RAG system with processed documents"""
        print("Loading RAG system...")

        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"Processed documents not found at {data_path}. "
                "Run document_processor.py first to build the index."
            )

        with open(data_path, 'r') as f:
            data = json.load(f)

        self.chunks = data['chunks']
        self.documents = data['documents']
        self.stats = data['stats']

        print(f"Loaded {self.stats['total_documents']} documents, "
              f"{self.stats['total_chunks']} chunks")

        # Build retriever
        self.retriever = TFIDFRetriever(self.chunks)
        print("RAG system ready!\n")

    def query(self, question: str, top_k: int = 3, show_sources: bool = True) -> str:
        """
        Query the documentation

        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            show_sources: Whether to show source files

        Returns:
            Formatted answer with context
        """
        # Retrieve relevant chunks
        results = self.retriever.search(question, top_k=top_k)

        # Format response
        response = []
        response.append(f"Query: {question}\n")
        response.append("="*60)

        # Combine context from top results
        context_parts = []
        sources = set()

        for i, (chunk, score) in enumerate(results, 1):
            if score > 0.05:  # Minimum relevance threshold
                context_parts.append(chunk['text'])
                sources.add(chunk['metadata']['filepath'])

        if not context_parts:
            response.append("\nNo relevant information found in documentation.")
            return '\n'.join(response)

        # Generate answer
        response.append("\nRelevant Information:")
        response.append("-"*60)

        for i, part in enumerate(context_parts, 1):
            response.append(f"\n[Context {i}]")
            response.append(part[:500])  # Limit length
            if len(part) > 500:
                response.append("...")

        if show_sources:
            response.append("\n" + "="*60)
            response.append("Sources:")
            for source in sorted(sources):
                response.append(f"  📄 {source}")

        return '\n'.join(response)

    def search_by_topic(self, topic: str) -> List[str]:
        """Find all documents related to a topic"""
        results = self.retriever.search(topic, top_k=10)
        files = set()

        for chunk, score in results:
            if score > 0.1:
                files.add(chunk['metadata']['filepath'])

        return sorted(files)

    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        return {
            **self.stats,
            'vocab_size': len(self.retriever.vocab),
            'categories': list(set(
                chunk['metadata'].get('category', 'unknown')
                for chunk in self.chunks
            ))
        }


def interactive_mode(rag: LAT5150RAG):
    """Interactive query mode"""
    print("\n" + "="*60)
    print("LAT5150DRVMIL RAG System - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question to search documentation")
    print("  - 'stats' - Show system statistics")
    print("  - 'topics <keyword>' - Find documents by topic")
    print("  - 'quit' or 'exit' - Exit")
    print("\n" + "="*60 + "\n")

    while True:
        try:
            user_input = input("LAT5150-RAG> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            elif user_input.lower() == 'stats':
                stats = rag.get_stats()
                print("\nSystem Statistics:")
                print("-"*40)
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()

            elif user_input.lower().startswith('topics '):
                topic = user_input[7:].strip()
                files = rag.search_by_topic(topic)
                print(f"\nDocuments related to '{topic}':")
                print("-"*40)
                for f in files:
                    print(f"  📄 {f}")
                print()

            else:
                # Regular query
                answer = rag.query(user_input)
                print(f"\n{answer}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='LAT5150DRVMIL RAG Query System'
    )
    parser.add_argument(
        'query',
        nargs='*',
        help='Query string (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of results to retrieve (default: 3)'
    )
    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source file names'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild document index before querying'
    )

    args = parser.parse_args()

    # Rebuild index if requested
    if args.rebuild:
        print("Rebuilding document index...")
        from document_processor import DocumentProcessor
        processor = DocumentProcessor('00-documentation')
        processor.process_all_documents()
        processor.save_processed_data()
        print()

    # Initialize RAG
    try:
        rag = LAT5150RAG()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun with --rebuild to create the index:")
        print("  python3 rag_query.py --rebuild")
        sys.exit(1)

    # Process query or enter interactive mode
    if args.query:
        query_string = ' '.join(args.query)
        answer = rag.query(
            query_string,
            top_k=args.top_k,
            show_sources=not args.no_sources
        )
        print(answer)
    else:
        interactive_mode(rag)


if __name__ == '__main__':
    main()
