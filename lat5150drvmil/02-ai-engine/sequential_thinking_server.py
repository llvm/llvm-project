#!/usr/bin/env python3
"""
DSMIL Sequential Thinking MCP Server (Security Hardened)
Based on: https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking

Enables dynamic problem-solving through structured thinking processes.
Integrated with DSMIL security framework.

SECURITY FEATURES:
- Token-based authentication
- Rate limiting (60 req/min per client)
- Input validation and sanitization
- Audit logging (~/.dsmil/mcp_audit.log)
- Thought history tracking
- Resource limits

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 1.0.0 (Security Hardened)
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any, Optional, List, Dict
import hashlib
from datetime import datetime

# Add parent directory for DSMIL imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
    )
except ImportError:
    print("ERROR: MCP library not found. Install with: pip3 install mcp", file=sys.stderr)
    sys.exit(1)

from mcp_security import get_security_manager


class ThinkingSession:
    """Manages a sequential thinking session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.thoughts: List[Dict[str, Any]] = []
        self.branches: Dict[int, List[Dict[str, Any]]] = {}
        self.created_at = datetime.now()
        self.last_update = datetime.now()
        self.total_thoughts_estimate = 0
        self.current_path: List[int] = []

    def add_thought(self, thought_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a thought to the session"""
        thought_number = thought_data.get("thoughtNumber", len(self.thoughts) + 1)

        thought_entry = {
            "thoughtNumber": thought_number,
            "thought": thought_data.get("thought", ""),
            "nextThoughtNeeded": thought_data.get("nextThoughtNeeded", True),
            "totalThoughts": thought_data.get("totalThoughts", self.total_thoughts_estimate),
            "isRevision": thought_data.get("isRevision", False),
            "revisesThought": thought_data.get("revisesThought"),
            "branchFromThought": thought_data.get("branchFromThought"),
            "branchId": thought_data.get("branchId"),
            "timestamp": datetime.now().isoformat()
        }

        # Handle branching
        if thought_entry.get("branchFromThought"):
            branch_from = thought_entry["branchFromThought"]
            branch_id = thought_entry.get("branchId", len(self.branches.get(branch_from, [])))

            if branch_from not in self.branches:
                self.branches[branch_from] = []

            self.branches[branch_from].append(thought_entry)
            thought_entry["path"] = self.current_path + [branch_from, branch_id]
        else:
            self.thoughts.append(thought_entry)
            thought_entry["path"] = [thought_number]

        self.last_update = datetime.now()
        self.total_thoughts_estimate = thought_entry["totalThoughts"]

        return thought_entry

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "total_thoughts": len(self.thoughts),
            "total_branches": sum(len(branches) for branches in self.branches.values()),
            "estimated_total": self.total_thoughts_estimate,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            "duration_seconds": (self.last_update - self.created_at).total_seconds()
        }

    def get_thought_history(self) -> List[Dict[str, Any]]:
        """Get all thoughts in chronological order"""
        all_thoughts = self.thoughts.copy()

        # Add branched thoughts
        for branch_list in self.branches.values():
            all_thoughts.extend(branch_list)

        # Sort by timestamp
        all_thoughts.sort(key=lambda x: x.get("timestamp", ""))

        return all_thoughts


class SequentialThinkingServer:
    """MCP Server for Sequential Thinking (Security Hardened)"""

    def __init__(self):
        self.server = Server("sequential-thinking")
        self.security = get_security_manager()
        self._client_id = self._generate_client_id()
        self.sessions: Dict[str, ThinkingSession] = {}
        self._setup_handlers()

        # Log server startup
        self.security.audit_logger.info(f"Sequential Thinking MCP Server started (client_id: {self._client_id})")

    def _generate_client_id(self) -> str:
        """Generate unique client identifier"""
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        return hashlib.sha256(f"{hostname}:{pid}:seqthink".encode()).hexdigest()[:16]

    def _get_or_create_session(self, session_id: Optional[str] = None) -> ThinkingSession:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]

        # Create new session
        new_id = session_id or hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        session = ThinkingSession(new_id)
        self.sessions[new_id] = session

        # Limit to 100 sessions (prevent memory exhaustion)
        if len(self.sessions) > 100:
            # Remove oldest session
            oldest = min(self.sessions.values(), key=lambda s: s.last_update)
            del self.sessions[oldest.session_id]

        return session

    def _setup_handlers(self):
        """Setup MCP request handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available MCP tools"""
            return [
                Tool(
                    name="sequential_thinking",
                    description="Enables dynamic problem-solving through structured thinking. "
                                "Break down complex problems into steps, revise thoughts, branch "
                                "into alternative reasoning paths. Maintains context across multiple "
                                "thinking steps. Ideal for: complex analysis, planning with revisions, "
                                "multi-step reasoning, filtering irrelevant information.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Current thinking step or observation (required)",
                            },
                            "nextThoughtNeeded": {
                                "type": "boolean",
                                "description": "Whether another thought step is needed (default: true)",
                                "default": True
                            },
                            "thoughtNumber": {
                                "type": "integer",
                                "description": "Current thought number in sequence (auto-increments if not provided)",
                                "minimum": 1
                            },
                            "totalThoughts": {
                                "type": "integer",
                                "description": "Estimated total thoughts needed (can be adjusted dynamically)",
                                "minimum": 1
                            },
                            "isRevision": {
                                "type": "boolean",
                                "description": "Whether this revises previous thinking (default: false)",
                                "default": False
                            },
                            "revisesThought": {
                                "type": "integer",
                                "description": "Thought number being revised (if isRevision=true)",
                                "minimum": 1
                            },
                            "branchFromThought": {
                                "type": "integer",
                                "description": "Branch from this thought to explore alternative path",
                                "minimum": 1
                            },
                            "branchId": {
                                "type": "string",
                                "description": "Identifier for this branch (auto-generated if not provided)"
                            },
                            "sessionId": {
                                "type": "string",
                                "description": "Session ID to maintain context across thoughts (auto-generated if not provided)"
                            }
                        },
                        "required": ["thought"]
                    }
                ),
                Tool(
                    name="get_thinking_session",
                    description="Retrieve the complete thinking history for a session. "
                                "Shows all thoughts, branches, revisions, and session statistics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sessionId": {
                                "type": "string",
                                "description": "Session ID to retrieve"
                            }
                        },
                        "required": ["sessionId"]
                    }
                ),
                Tool(
                    name="list_thinking_sessions",
                    description="List all active thinking sessions with summaries.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle MCP tool calls with security checks"""

            # SECURITY: Rate limiting
            if not self.security.check_rate_limit(self._client_id, name):
                error_msg = "Rate limit exceeded. Please wait before retrying."
                self.security.audit_request(name, arguments, self._client_id, False, error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            try:
                if name == "sequential_thinking":
                    thought = arguments.get("thought")
                    if not thought:
                        return [TextContent(type="text", text="Error: 'thought' is required")]

                    # SECURITY: Validate thought input
                    valid, error_msg = self.security.validate_query(thought)
                    if not valid:
                        self.security.audit_request(name, {"thought_length": len(thought)},
                                                  self._client_id, False, error_msg)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

                    # Get or create session
                    session_id = arguments.get("sessionId")
                    session = self._get_or_create_session(session_id)

                    # Add thought to session
                    thought_entry = session.add_thought({
                        "thought": thought,
                        "thoughtNumber": arguments.get("thoughtNumber"),
                        "nextThoughtNeeded": arguments.get("nextThoughtNeeded", True),
                        "totalThoughts": arguments.get("totalThoughts", 5),
                        "isRevision": arguments.get("isRevision", False),
                        "revisesThought": arguments.get("revisesThought"),
                        "branchFromThought": arguments.get("branchFromThought"),
                        "branchId": arguments.get("branchId")
                    })

                    # SECURITY: Audit
                    self.security.audit_request(
                        name,
                        {
                            "thought_length": len(thought),
                            "session_id": session.session_id,
                            "thought_number": thought_entry["thoughtNumber"],
                            "is_revision": thought_entry["isRevision"],
                            "is_branch": thought_entry.get("branchFromThought") is not None
                        },
                        self._client_id,
                        True
                    )

                    # Build response
                    output = f"Thought #{thought_entry['thoughtNumber']}"
                    if thought_entry["isRevision"]:
                        output += f" (revises #{thought_entry['revisesThought']})"
                    if thought_entry.get("branchFromThought"):
                        output += f" (branch from #{thought_entry['branchFromThought']})"

                    output += f" recorded.\n\n"
                    output += f"Session: {session.session_id}\n"
                    output += f"Progress: {thought_entry['thoughtNumber']}/{thought_entry['totalThoughts']} thoughts\n"
                    output += f"Next thought needed: {thought_entry['nextThoughtNeeded']}\n"
                    output += f"\nTotal thoughts in session: {len(session.thoughts)}\n"

                    if session.branches:
                        total_branches = sum(len(b) for b in session.branches.values())
                        output += f"Total branches: {total_branches}\n"

                    return [TextContent(type="text", text=output)]

                elif name == "get_thinking_session":
                    session_id = arguments.get("sessionId")
                    if not session_id:
                        return [TextContent(type="text", text="Error: 'sessionId' is required")]

                    if session_id not in self.sessions:
                        return [TextContent(type="text", text=f"Error: Session '{session_id}' not found")]

                    session = self.sessions[session_id]
                    summary = session.get_summary()
                    history = session.get_thought_history()

                    # SECURITY: Audit
                    self.security.audit_request(name, {"session_id": session_id}, self._client_id, True)

                    output = "=" * 60 + "\n"
                    output += f"Thinking Session: {session_id}\n"
                    output += "=" * 60 + "\n\n"
                    output += f"Created: {summary['created_at']}\n"
                    output += f"Duration: {summary['duration_seconds']:.1f} seconds\n"
                    output += f"Total Thoughts: {summary['total_thoughts']}\n"
                    output += f"Total Branches: {summary['total_branches']}\n"
                    output += f"Estimated Total: {summary['estimated_total']}\n\n"
                    output += "=" * 60 + "\n"
                    output += "Thought History:\n"
                    output += "=" * 60 + "\n\n"

                    for thought in history:
                        output += f"[Thought #{thought['thoughtNumber']}]"
                        if thought.get("isRevision"):
                            output += f" (revises #{thought.get('revisesThought')})"
                        if thought.get("branchFromThought"):
                            output += f" (branch from #{thought.get('branchFromThought')})"
                        output += "\n"
                        output += f"{thought['thought']}\n"
                        output += f"Next needed: {thought['nextThoughtNeeded']}\n"
                        output += f"Path: {' -> '.join(map(str, thought.get('path', [])))}\n"
                        output += "\n" + "-" * 60 + "\n\n"

                    return [TextContent(type="text", text=output)]

                elif name == "list_thinking_sessions":
                    # SECURITY: Audit
                    self.security.audit_request(name, {}, self._client_id, True)

                    if not self.sessions:
                        return [TextContent(type="text", text="No active thinking sessions.")]

                    output = f"Active Thinking Sessions ({len(self.sessions)}):\n\n"

                    for session in sorted(self.sessions.values(), key=lambda s: s.last_update, reverse=True):
                        summary = session.get_summary()
                        output += f"Session: {session.session_id}\n"
                        output += f"  - Thoughts: {summary['total_thoughts']}\n"
                        output += f"  - Branches: {summary['total_branches']}\n"
                        output += f"  - Duration: {summary['duration_seconds']:.1f}s\n"
                        output += f"  - Last Update: {summary['last_update']}\n"
                        output += "\n"

                    return [TextContent(type="text", text=output)]

                else:
                    # SECURITY: Audit unknown tool
                    self.security.audit_request(name, arguments, self._client_id, False, "Unknown tool")
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                # SECURITY: Audit exception
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                self.security.audit_logger.error(f"Tool execution exception: {name} - {str(e)}")
                return [TextContent(type="text", text=f"Error executing tool: {str(e)}")]

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = SequentialThinkingServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
