#!/usr/bin/env python3
"""
RAG Manager - Simple interface for managing RAG knowledge base
Add documents, search, view stats, manage index

Enhanced with Pydantic support for type-safe responses
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Import RAG system
sys.path.insert(0, str(Path(__file__).parent))
from rag_system import RAGSystem

# Pydantic support
sys.path.insert(0, str(Path(__file__).parent.parent / "02-ai-engine"))
try:
    from pydantic_models import (
        RAGQueryRequest,
        RAGQueryResult,
        RetrievedDocument,
        DocumentMetadata,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

class RAGManager:
    def __init__(self, pydantic_mode=False):
        """
        Initialize RAG Manager with optional Pydantic support

        Args:
            pydantic_mode: If True, return Pydantic models instead of dicts
        """
        self.rag = RAGSystem()
        self.pydantic_mode = pydantic_mode and PYDANTIC_AVAILABLE

    def add_file(self, filepath):
        """Add single file to RAG"""
        result = self.rag.ingest_file(filepath)
        self.rag.save_index()
        return result

    def add_folder(self, folder_path, recursive=True):
        """Add all supported files from folder"""
        folder = Path(folder_path)

        if not folder.exists():
            return {"error": f"Folder not found: {folder_path}"}

        # Supported extensions
        extensions = ['.pdf', '.txt', '.md', '.log', '.c', '.h', '.py', '.sh', '.cpp', '.java']

        # Find files
        if recursive:
            files = []
            for ext in extensions:
                files.extend(folder.rglob(f'*{ext}'))
        else:
            files = []
            for ext in extensions:
                files.extend(folder.glob(f'*{ext}'))

        # Ingest each file
        results = {
            "total": len(files),
            "success": 0,
            "already_indexed": 0,
            "errors": 0,
            "files": []
        }

        for file in files:
            result = self.rag.ingest_file(file)

            if isinstance(result, dict):
                if result.get("status") == "success":
                    results["success"] += 1
                    results["files"].append({"file": str(file), "result": result})
                elif result.get("status") == "already_indexed":
                    results["already_indexed"] += 1
                else:
                    results["errors"] += 1
                    results["files"].append({"file": str(file), "error": result.get("error")})

        self.rag.save_index()
        return results

    def search(self, query, limit=10, min_score=0.0):
        """
        Search RAG index with optional Pydantic output

        Args:
            query: Search query string or RAGQueryRequest (if Pydantic mode)
            limit: Maximum number of results (default: 10)
            min_score: Minimum relevance score (default: 0.0)

        Returns:
            dict or RAGQueryResult (depending on pydantic_mode)
        """
        import time
        start_time = time.time()

        # Handle Pydantic request
        if PYDANTIC_AVAILABLE and isinstance(query, RAGQueryRequest):
            query_str = query.query
            limit = query.top_k
            min_score = query.min_score
        else:
            query_str = query

        # Perform search
        raw_results = self.rag.search(query_str, max_results=limit)

        # Return dict if not in Pydantic mode
        if not self.pydantic_mode:
            return raw_results

        # Convert to Pydantic model
        try:
            documents = []
            for result in raw_results.get('results', []):
                metadata = DocumentMetadata(
                    source=result.get('filename', 'unknown'),
                    title=result.get('filename', 'unknown'),
                    page=result.get('page'),
                    section=result.get('section')
                )

                doc = RetrievedDocument(
                    content=result.get('text', ''),
                    score=result.get('score', 0.0),
                    metadata=metadata,
                    chunk_id=result.get('chunk_id')
                )

                # Filter by min_score
                if doc.score >= min_score:
                    documents.append(doc)

            search_time_ms = (time.time() - start_time) * 1000

            return RAGQueryResult(
                documents=documents,
                query=query_str,
                total_found=len(documents),
                search_time_ms=search_time_ms
            )

        except Exception as e:
            print(f"Warning: Pydantic conversion failed: {e}")
            return raw_results

    def list_documents(self):
        """List all indexed documents"""
        docs = []
        for hash_val, doc in self.rag.documents.items():
            docs.append({
                "filename": doc.get('filename', 'unknown'),
                "tokens": doc.get('token_count', 0),
                "chars": doc.get('char_count', 0),
                "indexed": doc.get('indexed_at', 'unknown'),
                "hash": hash_val[:16] + "..."
            })
        return {
            "total_documents": len(docs),
            "documents": docs
        }

    def get_stats(self):
        """Get RAG system statistics"""
        total_tokens = sum(doc.get('token_count', 0) for doc in self.rag.documents.values())
        total_docs = len(self.rag.documents)

        return {
            "documents": total_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": total_tokens // total_docs if total_docs > 0 else 0,
            "index_path": str(self.rag.index_path),
            "index_size_mb": sum(f.stat().st_size for f in self.rag.index_path.glob('*')) / (1024*1024)
        }

    def remove_document(self, filename_or_hash):
        """Remove document from index"""
        # Try by hash first
        if filename_or_hash in self.rag.documents:
            doc = self.rag.documents.pop(filename_or_hash)
            self.rag.save_index()
            return {"status": "removed", "filename": doc['filename']}

        # Try by filename
        for hash_val, doc in list(self.rag.documents.items()):
            if doc['filename'] == filename_or_hash:
                self.rag.documents.pop(hash_val)
                self.rag.save_index()
                return {"status": "removed", "hash": hash_val[:16] + "..."}

        return {"error": "Document not found"}

    def clear_index(self, confirm=False):
        """Clear entire RAG index"""
        if not confirm:
            return {"error": "Must confirm clear operation", "docs_would_delete": len(self.rag.documents)}

        self.rag.documents = {}
        self.rag.tokens = {}
        self.rag.save_index()
        return {"status": "cleared", "deleted": len(self.rag.documents)}

# CLI
if __name__ == "__main__":
    manager = RAGManager()

    if len(sys.argv) < 2:
        print("RAG Manager - Usage:")
        print("  python3 rag_manager.py stats")
        print("  python3 rag_manager.py list")
        print("  python3 rag_manager.py add-file /path/to/file.pdf")
        print("  python3 rag_manager.py add-folder /path/to/folder")
        print("  python3 rag_manager.py search 'your query'")
        print("  python3 rag_manager.py remove filename_or_hash")
        print("  python3 rag_manager.py clear --confirm")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "stats":
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2))

    elif cmd == "list":
        docs = manager.list_documents()
        print(json.dumps(docs, indent=2))

    elif cmd == "add-file" and len(sys.argv) > 2:
        result = manager.add_file(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif cmd == "add-folder" and len(sys.argv) > 2:
        result = manager.add_folder(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif cmd == "search" and len(sys.argv) > 2:
        results = manager.search(sys.argv[2])
        print(json.dumps(results, indent=2))

    elif cmd == "remove" and len(sys.argv) > 2:
        result = manager.remove_document(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif cmd == "clear":
        confirm = "--confirm" in sys.argv
        result = manager.clear_index(confirm=confirm)
        print(json.dumps(result, indent=2))
