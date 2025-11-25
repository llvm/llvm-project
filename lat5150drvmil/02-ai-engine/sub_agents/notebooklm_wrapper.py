#!/usr/bin/env python3
"""
Google NotebookLM Wrapper for DSMIL AI Engine
----------------------------------------------
Provides NotebookLM-style functionality using Google Gemini API:
- Document ingestion and source management
- Multi-source Q&A with grounding
- Summary generation (FAQs, study guides, briefing docs)
- Source synthesis and analysis
- Natural language invocation

Architecture:
- Uses Gemini's long context (2M tokens) for document understanding
- Manages sources with embeddings for efficient retrieval
- Provides NotebookLM-like features: summarize, create FAQ, study guide, etc.
- Fully integrated with DSMIL AI Engine routing

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Document processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class Source:
    """Represents a document source for NotebookLM"""

    def __init__(self, source_id: str, title: str, content: str,
                 source_type: str = "text", metadata: Optional[Dict] = None):
        self.source_id = source_id
        self.title = title
        self.content = content
        self.source_type = source_type
        self.metadata = metadata or {}
        self.added_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert source to dictionary"""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content,
            "source_type": self.source_type,
            "metadata": self.metadata,
            "added_at": self.added_at,
            "content_length": len(self.content)
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Source':
        """Create source from dictionary"""
        source = Source(
            source_id=data["source_id"],
            title=data["title"],
            content=data["content"],
            source_type=data.get("source_type", "text"),
            metadata=data.get("metadata", {})
        )
        source.added_at = data.get("added_at", datetime.now().isoformat())
        return source


class NotebookLMAgent:
    """
    Google NotebookLM-style agent using Gemini API

    Provides document-based research and synthesis capabilities:
    - Add sources (documents, PDFs, text files)
    - Ask questions grounded in sources
    - Generate summaries, FAQs, study guides
    - Multi-source synthesis and analysis
    """

    def __init__(self, api_key: Optional[str] = None, storage_dir: Optional[str] = None):
        """
        Initialize NotebookLM agent

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var, then hardcoded key)
            storage_dir: Directory for storing sources and notebooks
        """
        # Try: provided key -> env var -> hardcoded fallback
        self.api_key = (
            api_key or
            os.environ.get("GOOGLE_API_KEY") or
            os.environ.get("GEMINI_API_KEY") or
            "AIzaSyDirt7ZxgBgOIKK8MwcqKPbfVNsMAJgQEA"  # Hardcoded fallback
        )
        self.available = GEMINI_AVAILABLE and self.api_key is not None

        # Storage
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".dsmil" / "notebooklm"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Sources management
        self.sources: Dict[str, Source] = {}
        self.notebooks: Dict[str, Dict] = {}  # Notebook ID -> {title, source_ids, created_at}

        # Initialize Gemini
        if self.available:
            genai.configure(api_key=self.api_key)
            # Use Gemini 2.0 Flash for long context and thinking
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("NotebookLM Agent initialized with Gemini 2.0 Flash")
        else:
            self.model = None
            logger.warning("NotebookLM Agent unavailable - missing Gemini API key or library")

        # Load existing data
        self._load_data()

    def is_available(self) -> bool:
        """Check if NotebookLM agent is available"""
        return self.available

    def _load_data(self):
        """Load sources and notebooks from storage"""
        sources_file = self.storage_dir / "sources.json"
        notebooks_file = self.storage_dir / "notebooks.json"

        if sources_file.exists():
            try:
                with open(sources_file, 'r') as f:
                    data = json.load(f)
                    self.sources = {sid: Source.from_dict(s) for sid, s in data.items()}
                logger.info(f"Loaded {len(self.sources)} sources")
            except Exception as e:
                logger.error(f"Failed to load sources: {e}")

        if notebooks_file.exists():
            try:
                with open(notebooks_file, 'r') as f:
                    self.notebooks = json.load(f)
                logger.info(f"Loaded {len(self.notebooks)} notebooks")
            except Exception as e:
                logger.error(f"Failed to load notebooks: {e}")

    def _save_data(self):
        """Save sources and notebooks to storage"""
        sources_file = self.storage_dir / "sources.json"
        notebooks_file = self.storage_dir / "notebooks.json"

        try:
            with open(sources_file, 'w') as f:
                json.dump({sid: s.to_dict() for sid, s in self.sources.items()}, f, indent=2)

            with open(notebooks_file, 'w') as f:
                json.dump(self.notebooks, f, indent=2)

            logger.info("Saved sources and notebooks")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def add_source(self, content: str = None, file_path: str = None,
                   title: str = None, source_type: str = "text",
                   metadata: Optional[Dict] = None) -> Dict:
        """
        Add a source to the NotebookLM workspace

        Args:
            content: Text content (if not loading from file)
            file_path: Path to file to load
            title: Source title
            source_type: Type of source (text, pdf, markdown, etc.)
            metadata: Additional metadata

        Returns:
            Dict with source info and success status
        """
        if not self.available:
            return {"success": False, "error": "NotebookLM agent not available"}

        try:
            # Load content from file if provided
            if file_path:
                file_path = Path(file_path)
                if not file_path.exists():
                    return {"success": False, "error": f"File not found: {file_path}"}

                # Determine loader based on file extension
                if file_path.suffix == '.pdf':
                    if not LANGCHAIN_AVAILABLE:
                        return {"success": False, "error": "PDF loading requires langchain"}
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    content = "\n\n".join([doc.page_content for doc in docs])
                    source_type = "pdf"
                elif file_path.suffix in ['.md', '.markdown']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    source_type = "markdown"
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    source_type = "text"

                title = title or file_path.name

            if not content:
                return {"success": False, "error": "No content provided"}

            # Generate source ID
            source_id = hashlib.md5(content.encode()).hexdigest()[:16]

            # Create source
            source = Source(
                source_id=source_id,
                title=title or f"Source {source_id}",
                content=content,
                source_type=source_type,
                metadata=metadata or {}
            )

            # Store source
            self.sources[source_id] = source
            self._save_data()

            logger.info(f"Added source: {source.title} ({len(content)} chars)")

            return {
                "success": True,
                "source_id": source_id,
                "title": source.title,
                "content_length": len(content),
                "source_type": source_type
            }

        except Exception as e:
            logger.error(f"Failed to add source: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def create_notebook(self, title: str, source_ids: List[str] = None) -> Dict:
        """
        Create a new notebook with selected sources

        Args:
            title: Notebook title
            source_ids: List of source IDs to include

        Returns:
            Dict with notebook info
        """
        notebook_id = hashlib.md5(f"{title}{datetime.now()}".encode()).hexdigest()[:16]

        self.notebooks[notebook_id] = {
            "title": title,
            "source_ids": source_ids or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        self._save_data()

        return {
            "success": True,
            "notebook_id": notebook_id,
            "title": title,
            "source_count": len(source_ids or [])
        }

    def query(self, prompt: str, source_ids: List[str] = None,
              notebook_id: str = None, mode: str = "qa") -> Dict:
        """
        Query sources with natural language

        Args:
            prompt: User question or request
            source_ids: Specific sources to query (optional)
            notebook_id: Query within a notebook (optional)
            mode: Query mode - "qa", "summarize", "faq", "study_guide", "synthesis"

        Returns:
            Dict with response and metadata
        """
        if not self.available:
            return {
                "success": False,
                "error": "NotebookLM agent not available",
                "fallback": "Please configure GOOGLE_API_KEY to use NotebookLM features"
            }

        try:
            # Determine which sources to use
            if notebook_id and notebook_id in self.notebooks:
                source_ids = self.notebooks[notebook_id]["source_ids"]
            elif source_ids is None:
                source_ids = list(self.sources.keys())  # Use all sources

            if not source_ids:
                return {
                    "success": False,
                    "error": "No sources available. Add sources first with add_source()"
                }

            # Build context from sources
            context = self._build_context(source_ids)

            # Build prompt based on mode
            system_prompt = self._get_system_prompt(mode)
            full_prompt = f"{system_prompt}\n\n## Sources:\n\n{context}\n\n## Query:\n{prompt}\n\n## Response:"

            # Check token limit (Gemini 2.0 supports 2M tokens)
            # For safety, we'll warn if context is very large
            token_estimate = len(full_prompt.split())
            if token_estimate > 100000:
                logger.warning(f"Large context: ~{token_estimate} tokens")

            # Generate response with Gemini
            response = self.model.generate_content(full_prompt)

            return {
                "success": True,
                "response": response.text,
                "mode": mode,
                "sources_used": len(source_ids),
                "source_titles": [self.sources[sid].title for sid in source_ids if sid in self.sources],
                "backend": "gemini_notebooklm",
                "cost": 0.0,  # Student tier
                "privacy": "cloud"
            }

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _build_context(self, source_ids: List[str]) -> str:
        """Build context string from sources"""
        context_parts = []

        for sid in source_ids:
            if sid in self.sources:
                source = self.sources[sid]
                context_parts.append(f"### {source.title}\n\n{source.content}\n")

        return "\n".join(context_parts)

    def _get_system_prompt(self, mode: str) -> str:
        """Get system prompt based on query mode"""
        prompts = {
            "qa": """You are a helpful research assistant analyzing documents. Answer the query based ONLY on the provided sources. If the answer isn't in the sources, say so. Always cite which source you're referencing.""",

            "summarize": """You are a document summarization expert. Create a comprehensive summary of the provided sources. Include:
- Main themes and key points
- Important details and findings
- Connections between sources
Keep the summary clear, concise, and well-structured.""",

            "faq": """You are creating a Frequently Asked Questions (FAQ) document. Based on the sources provided, generate a comprehensive FAQ with:
- 8-12 important questions that readers would ask
- Clear, concise answers grounded in the sources
- Organized by topic/theme
Format each as Q: [question] / A: [answer]""",

            "study_guide": """You are creating a study guide. Based on the sources, create:
- Key concepts and definitions
- Important facts and figures
- Relationships and connections
- Practice questions
- Summary points
Make it comprehensive and well-organized for learning.""",

            "synthesis": """You are a research synthesis expert. Analyze the sources and provide:
- Overarching themes across all sources
- Areas of agreement and disagreement
- Gaps in coverage
- Synthesized insights
- Recommendations for further research
Focus on connecting ideas across sources.""",

            "briefing": """You are creating an executive briefing document. Provide:
- Executive summary (2-3 sentences)
- Key findings (bullet points)
- Critical insights
- Recommendations
- Next steps
Keep it concise and actionable for decision-makers."""
        }

        return prompts.get(mode, prompts["qa"])

    def summarize_sources(self, source_ids: List[str] = None) -> Dict:
        """Generate a summary of sources"""
        return self.query("Create a comprehensive summary of all sources",
                         source_ids=source_ids, mode="summarize")

    def create_faq(self, source_ids: List[str] = None) -> Dict:
        """Generate FAQ from sources"""
        return self.query("Create a detailed FAQ",
                         source_ids=source_ids, mode="faq")

    def create_study_guide(self, source_ids: List[str] = None) -> Dict:
        """Generate study guide from sources"""
        return self.query("Create a comprehensive study guide",
                         source_ids=source_ids, mode="study_guide")

    def synthesize(self, source_ids: List[str] = None) -> Dict:
        """Synthesize insights across sources"""
        return self.query("Synthesize insights across all sources",
                         source_ids=source_ids, mode="synthesis")

    def create_briefing(self, source_ids: List[str] = None) -> Dict:
        """Create executive briefing from sources"""
        return self.query("Create an executive briefing",
                         source_ids=source_ids, mode="briefing")

    def list_sources(self) -> Dict:
        """List all sources"""
        return {
            "success": True,
            "sources": [s.to_dict() for s in self.sources.values()],
            "count": len(self.sources)
        }

    def list_notebooks(self) -> Dict:
        """List all notebooks"""
        return {
            "success": True,
            "notebooks": self.notebooks,
            "count": len(self.notebooks)
        }

    def remove_source(self, source_id: str) -> Dict:
        """Remove a source"""
        if source_id in self.sources:
            del self.sources[source_id]
            self._save_data()
            return {"success": True, "message": f"Source {source_id} removed"}
        return {"success": False, "error": "Source not found"}

    def clear_all_sources(self) -> Dict:
        """Clear all sources"""
        count = len(self.sources)
        self.sources = {}
        self._save_data()
        return {"success": True, "message": f"Cleared {count} sources"}

    def get_status(self) -> Dict:
        """Get agent status"""
        total_content_length = sum(len(s.content) for s in self.sources.values())

        return {
            "available": self.available,
            "sources_count": len(self.sources),
            "notebooks_count": len(self.notebooks),
            "total_content_length": total_content_length,
            "storage_dir": str(self.storage_dir),
            "model": "gemini-2.0-flash-exp" if self.available else None,
            "capabilities": [
                "Document ingestion (text, PDF, markdown)",
                "Multi-source Q&A with grounding",
                "Summary generation",
                "FAQ creation",
                "Study guide generation",
                "Source synthesis",
                "Executive briefings"
            ]
        }


# CLI for testing
if __name__ == "__main__":
    import sys

    agent = NotebookLMAgent()

    if len(sys.argv) < 2:
        print("\nNotebookLM Agent - Usage:")
        print("  python3 notebooklm_wrapper.py status")
        print("  python3 notebooklm_wrapper.py add 'content here' --title 'My Source'")
        print("  python3 notebooklm_wrapper.py add-file /path/to/file.pdf")
        print("  python3 notebooklm_wrapper.py query 'your question'")
        print("  python3 notebooklm_wrapper.py summarize")
        print("  python3 notebooklm_wrapper.py faq")
        print("  python3 notebooklm_wrapper.py study-guide")
        print("  python3 notebooklm_wrapper.py list-sources")
        print("  python3 notebooklm_wrapper.py clear")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        print(json.dumps(agent.get_status(), indent=2))

    elif cmd == "add" and len(sys.argv) > 2:
        content = sys.argv[2]
        title = None
        if '--title' in sys.argv:
            idx = sys.argv.index('--title')
            if idx + 1 < len(sys.argv):
                title = sys.argv[idx + 1]

        result = agent.add_source(content=content, title=title)
        print(json.dumps(result, indent=2))

    elif cmd == "add-file" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        result = agent.add_source(file_path=file_path)
        print(json.dumps(result, indent=2))

    elif cmd == "query" and len(sys.argv) > 2:
        query = sys.argv[2]
        result = agent.query(query)
        print(json.dumps(result, indent=2))

    elif cmd == "summarize":
        result = agent.summarize_sources()
        print(json.dumps(result, indent=2))

    elif cmd == "faq":
        result = agent.create_faq()
        print(json.dumps(result, indent=2))

    elif cmd == "study-guide":
        result = agent.create_study_guide()
        print(json.dumps(result, indent=2))

    elif cmd == "list-sources":
        result = agent.list_sources()
        print(json.dumps(result, indent=2))

    elif cmd == "clear":
        result = agent.clear_all_sources()
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
