#!/usr/bin/env python3
"""
NotebookLM Subagent for ACE-FCA Workflow
----------------------------------------
Specialized subagent for document-based research and synthesis using NotebookLM.

Capabilities:
- Document ingestion and source management
- Multi-source Q&A with grounding
- Summary, FAQ, and study guide generation
- Source synthesis and analysis
- Executive briefings

Follows ACE-FCA pattern: Returns compressed findings to prevent context pollution.

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import sys
import os
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ace_subagents import BaseSubagent, SubagentResult
    from sub_agents.notebooklm_wrapper import NotebookLMAgent
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False
    # Fallback definitions
    class SubagentResult:
        def __init__(self, agent_type, compressed_output, raw_output, metadata, success=True, error=None):
            self.agent_type = agent_type
            self.compressed_output = compressed_output
            self.raw_output = raw_output
            self.metadata = metadata
            self.success = success
            self.error = error


class NotebookLMSubagent(BaseSubagent if ACE_AVAILABLE else object):
    """
    NotebookLM-powered research subagent for ACE-FCA workflows

    Use cases:
    - Research phase: Ingest documentation, research papers, specs
    - Analysis phase: Compare multiple sources, synthesize findings
    - Planning phase: Generate briefings and summaries for decision-making
    - Verification phase: Create study guides and FAQs for knowledge validation
    """

    def __init__(self, ai_engine=None, max_tokens: int = 8192):
        """
        Initialize NotebookLM subagent

        Args:
            ai_engine: AI engine (not used, NotebookLM has its own)
            max_tokens: Context window (NotebookLM supports up to 2M)
        """
        if ACE_AVAILABLE and ai_engine:
            super().__init__(ai_engine, max_tokens)

        self.notebooklm = NotebookLMAgent()
        self.agent_type = "NotebookLMSubagent"

    def execute(self, task: Dict) -> SubagentResult:
        """
        Execute NotebookLM research task

        Task format:
        {
            "action": "add_source" | "query" | "summarize" | "faq" | "study_guide" | "synthesis" | "briefing",
            "content": str (for add_source),
            "file_path": str (for add_source),
            "prompt": str (for query),
            "source_ids": List[str] (optional),
            "mode": str (qa, summarize, etc.)
        }

        Returns:
            SubagentResult with compressed findings
        """
        if not self.notebooklm.is_available():
            return SubagentResult(
                agent_type=self.agent_type,
                compressed_output="NotebookLM unavailable - GOOGLE_API_KEY not configured",
                raw_output="NotebookLM agent is not available. Please configure GOOGLE_API_KEY environment variable.",
                metadata={"available": False},
                success=False,
                error="NotebookLM not configured"
            )

        action = task.get("action", "query")
        metadata = {"action": action}

        try:
            # Execute action
            if action == "add_source":
                result = self.notebooklm.add_source(
                    content=task.get("content"),
                    file_path=task.get("file_path"),
                    title=task.get("title"),
                    metadata=task.get("metadata")
                )
                raw_output = f"Source added: {result.get('title', 'Unknown')}\n"
                raw_output += f"Source ID: {result.get('source_id', 'N/A')}\n"
                raw_output += f"Content length: {result.get('content_length', 0)} chars\n"
                compressed = f"✅ Added source '{result.get('title')}' ({result.get('content_length', 0)} chars)"
                metadata.update(result)

            elif action == "query":
                result = self.notebooklm.query(
                    prompt=task.get("prompt", ""),
                    source_ids=task.get("source_ids"),
                    notebook_id=task.get("notebook_id"),
                    mode=task.get("mode", "qa")
                )
                raw_output = result.get("response", "")
                # Compress using ACE-FCA pattern
                if ACE_AVAILABLE:
                    compressed = self._compress_output(raw_output, max_tokens=500)
                else:
                    # Simple compression: first 500 words
                    words = raw_output.split()[:500]
                    compressed = ' '.join(words)
                metadata.update({
                    "mode": result.get("mode"),
                    "sources_used": result.get("sources_used", 0)
                })

            elif action == "summarize":
                result = self.notebooklm.summarize_sources(
                    source_ids=task.get("source_ids")
                )
                raw_output = result.get("response", "")
                if ACE_AVAILABLE:
                    compressed = self._compress_output(raw_output, max_tokens=500)
                else:
                    words = raw_output.split()[:500]
                    compressed = ' '.join(words)
                metadata["sources_used"] = result.get("sources_used", 0)

            elif action == "faq":
                result = self.notebooklm.create_faq(
                    source_ids=task.get("source_ids")
                )
                raw_output = result.get("response", "")
                if ACE_AVAILABLE:
                    compressed = self._compress_output(raw_output, max_tokens=700)
                else:
                    words = raw_output.split()[:700]
                    compressed = ' '.join(words)
                metadata["sources_used"] = result.get("sources_used", 0)

            elif action == "study_guide":
                result = self.notebooklm.create_study_guide(
                    source_ids=task.get("source_ids")
                )
                raw_output = result.get("response", "")
                if ACE_AVAILABLE:
                    compressed = self._compress_output(raw_output, max_tokens=700)
                else:
                    words = raw_output.split()[:700]
                    compressed = ' '.join(words)
                metadata["sources_used"] = result.get("sources_used", 0)

            elif action == "synthesis":
                result = self.notebooklm.synthesize(
                    source_ids=task.get("source_ids")
                )
                raw_output = result.get("response", "")
                if ACE_AVAILABLE:
                    compressed = self._compress_output(raw_output, max_tokens=600)
                else:
                    words = raw_output.split()[:600]
                    compressed = ' '.join(words)
                metadata["sources_used"] = result.get("sources_used", 0)

            elif action == "briefing":
                result = self.notebooklm.create_briefing(
                    source_ids=task.get("source_ids")
                )
                raw_output = result.get("response", "")
                if ACE_AVAILABLE:
                    compressed = self._compress_output(raw_output, max_tokens=400)
                else:
                    words = raw_output.split()[:400]
                    compressed = ' '.join(words)
                metadata["sources_used"] = result.get("sources_used", 0)

            elif action == "list_sources":
                result = self.notebooklm.list_sources()
                sources = result.get("sources", [])
                raw_output = "Available sources:\n"
                for src in sources:
                    raw_output += f"- {src['title']} ({src['source_id'][:8]}..., {src['content_length']} chars)\n"
                compressed = f"{len(sources)} sources available: " + ", ".join([s['title'] for s in sources[:5]])
                if len(sources) > 5:
                    compressed += f" + {len(sources) - 5} more"
                metadata["source_count"] = len(sources)

            else:
                return SubagentResult(
                    agent_type=self.agent_type,
                    compressed_output=f"Unknown action: {action}",
                    raw_output=f"Unknown action: {action}",
                    metadata=metadata,
                    success=False,
                    error=f"Unknown action: {action}"
                )

            return SubagentResult(
                agent_type=self.agent_type,
                compressed_output=compressed,
                raw_output=raw_output,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return SubagentResult(
                agent_type=self.agent_type,
                compressed_output=f"NotebookLM {action} failed: {str(e)}",
                raw_output=str(e),
                metadata=metadata,
                success=False,
                error=str(e)
            )

    def get_capabilities(self) -> List[str]:
        """Get list of NotebookLM capabilities"""
        return [
            "Document ingestion (text, PDF, markdown)",
            "Multi-source Q&A with grounding",
            "Summary generation",
            "FAQ creation",
            "Study guide generation",
            "Source synthesis and comparison",
            "Executive briefing generation"
        ]


# Convenience function for creating NotebookLM subagent
def create_notebooklm_subagent(ai_engine=None) -> NotebookLMSubagent:
    """Create NotebookLM subagent instance"""
    return NotebookLMSubagent(ai_engine=ai_engine)


# CLI for testing
if __name__ == "__main__":
    import json

    agent = NotebookLMSubagent()

    if len(sys.argv) < 2:
        print("\nNotebookLM Subagent - Usage:")
        print("  python3 notebooklm_subagent.py add-source 'content here'")
        print("  python3 notebooklm_subagent.py add-file /path/to/file.pdf")
        print("  python3 notebooklm_subagent.py query 'your question'")
        print("  python3 notebooklm_subagent.py summarize")
        print("  python3 notebooklm_subagent.py faq")
        print("  python3 notebooklm_subagent.py list-sources")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "add-source" and len(sys.argv) > 2:
        task = {
            "action": "add_source",
            "content": sys.argv[2],
            "title": sys.argv[3] if len(sys.argv) > 3 else "Test Source"
        }
        result = agent.execute(task)
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "add-file" and len(sys.argv) > 2:
        task = {
            "action": "add_source",
            "file_path": sys.argv[2]
        }
        result = agent.execute(task)
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "query" and len(sys.argv) > 2:
        task = {
            "action": "query",
            "prompt": sys.argv[2]
        }
        result = agent.execute(task)
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "summarize":
        task = {"action": "summarize"}
        result = agent.execute(task)
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "faq":
        task = {"action": "faq"}
        result = agent.execute(task)
        print(json.dumps(result.to_dict(), indent=2))

    elif cmd == "list-sources":
        task = {"action": "list_sources"}
        result = agent.execute(task)
        print(json.dumps(result.to_dict(), indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
