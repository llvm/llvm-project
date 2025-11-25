#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) System
Tokenize and index documents from folders
"""

import os
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
import subprocess

class RAGSystem:
    def __init__(self, index_path=None):
        if index_path is None:
            # Use dynamic path based on current user's home
            index_path = Path.home() / ".rag_index"
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True, parents=True)
        self.docs_db = self.index_path / "documents.json"
        self.tokens_db = self.index_path / "tokens.json"
        self.load_index()

    def load_index(self):
        """Load existing index"""
        if self.docs_db.exists():
            with open(self.docs_db, 'r') as f:
                self.documents = json.load(f)
        else:
            self.documents = {}

        if self.tokens_db.exists():
            with open(self.tokens_db, 'r') as f:
                self.tokens = json.load(f)
        else:
            self.tokens = {}

    def save_index(self):
        """Save index to disk"""
        with open(self.docs_db, 'w') as f:
            json.dump(self.documents, f, indent=2)

        with open(self.tokens_db, 'w') as f:
            json.dump(self.tokens, f, indent=2)

    def tokenize_text(self, text):
        """Simple tokenization - split on whitespace and punctuation"""
        # Convert to lowercase
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def extract_text_from_pdf(self, filepath):
        """Extract text from PDF using multiple methods"""
        methods = [
            ['pdftotext', str(filepath), '-'],
            ['strings', str(filepath)]
        ]

        for method in methods:
            try:
                result = subprocess.run(method, capture_output=True,
                                      text=True, timeout=60)
                if result.returncode == 0 and result.stdout:
                    return result.stdout
            except:
                continue

        return ""

    def ingest_file(self, filepath):
        """Ingest a single file into the RAG system"""
        filepath = Path(filepath)

        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()

        if file_hash in self.documents:
            return {"status": "already_indexed", "hash": file_hash}

        # Extract text based on file type
        text = ""
        file_type = filepath.suffix.lower()

        if file_type == '.pdf':
            text = self.extract_text_from_pdf(filepath)
        elif file_type in ['.txt', '.md', '.log', '.c', '.h', '.py', '.sh']:
            try:
                text = filepath.read_text(encoding='utf-8', errors='ignore')
            except:
                return {"error": "Failed to read text file"}
        else:
            return {"error": f"Unsupported file type: {file_type}"}

        if not text:
            return {"error": "No text extracted"}

        # Tokenize
        tokens = self.tokenize_text(text)

        # Store document metadata
        self.documents[file_hash] = {
            "path": str(filepath),
            "filename": filepath.name,
            "type": file_type,
            "size": filepath.stat().st_size,
            "token_count": len(tokens),
            "char_count": len(text),
            "indexed_at": datetime.now().isoformat(),
            "preview": text[:500]
        }

        # Build reverse index (token -> documents)
        for token in set(tokens):
            if token not in self.tokens:
                self.tokens[token] = []
            if file_hash not in self.tokens[token]:
                self.tokens[token].append(file_hash)

        self.save_index()

        return {
            "status": "success",
            "hash": file_hash,
            "filename": filepath.name,
            "tokens": len(set(tokens)),
            "chars": len(text)
        }

    def ingest_folder(self, folder_path, recursive=True, extensions=None):
        """Ingest all documents in a folder"""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            return {"error": f"Folder not found: {folder_path}"}

        if extensions is None:
            extensions = ['.pdf', '.txt', '.md', '.log', '.c', '.h', '.py', '.sh']

        results = {
            "processed": 0,
            "success": 0,
            "already_indexed": 0,
            "errors": 0,
            "files": []
        }

        # Find files
        pattern = "**/*" if recursive else "*"
        for filepath in folder_path.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in extensions:
                results["processed"] += 1
                result = self.ingest_file(filepath)

                if "error" in result:
                    results["errors"] += 1
                elif result.get("status") == "already_indexed":
                    results["already_indexed"] += 1
                else:
                    results["success"] += 1

                results["files"].append({
                    "file": filepath.name,
                    "result": result
                })

        self.save_index()
        return results

    def search(self, query, max_results=10):
        """Search indexed documents"""
        query_tokens = self.tokenize_text(query)

        # Score documents by token overlap
        scores = {}
        for token in query_tokens:
            if token in self.tokens:
                for doc_hash in self.tokens[token]:
                    scores[doc_hash] = scores.get(doc_hash, 0) + 1

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top results
        results = []
        for doc_hash, score in sorted_docs[:max_results]:
            if doc_hash in self.documents:
                doc = self.documents[doc_hash].copy()
                doc['relevance_score'] = score
                doc['hash'] = doc_hash
                results.append(doc)

        return results

    def get_stats(self):
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "total_unique_tokens": len(self.tokens),
            "index_size_bytes": self.docs_db.stat().st_size if self.docs_db.exists() else 0,
            "index_path": str(self.index_path)
        }

    def list_documents(self, limit=50):
        """List all indexed documents"""
        docs = []
        for hash, doc in list(self.documents.items())[:limit]:
            docs.append({
                "hash": hash[:12],
                "filename": doc['filename'],
                "type": doc['type'],
                "tokens": doc['token_count'],
                "size": doc['size'],
                "indexed": doc['indexed_at']
            })
        return docs

# Command line interface
if __name__ == "__main__":
    import sys

    rag = RAGSystem()

    if len(sys.argv) < 2:
        print("RAG System - Usage:")
        print("  python3 rag_system.py ingest FILE")
        print("  python3 rag_system.py ingest-folder FOLDER")
        print("  python3 rag_system.py search 'query text'")
        print("  python3 rag_system.py stats")
        print("  python3 rag_system.py list")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest" and len(sys.argv) > 2:
        result = rag.ingest_file(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif command == "ingest-folder" and len(sys.argv) > 2:
        result = rag.ingest_folder(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif command == "search" and len(sys.argv) > 2:
        query = ' '.join(sys.argv[2:])
        results = rag.search(query)
        print(f"\nFound {len(results)} results for: {query}\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc['filename']} (score: {doc['relevance_score']})")
            print(f"   {doc['preview'][:100]}...")
            print()

    elif command == "stats":
        stats = rag.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "list":
        docs = rag.list_documents()
        print(f"\nIndexed Documents ({len(docs)}):\n")
        for doc in docs:
            print(f"  {doc['filename']}: {doc['tokens']} tokens, {doc['size']} bytes")

    else:
        print("Unknown command:", command)
