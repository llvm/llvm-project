#!/usr/bin/env python3
"""
DSMIL Memory MCP Server (Security Hardened)
Based on: https://github.com/modelcontextprotocol/servers/tree/main/src/memory

Persistent knowledge graph for AI memory across sessions.

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List
import hashlib
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: MCP library not found", file=sys.stderr)
    sys.exit(1)

from mcp_security import get_security_manager


class MemoryGraph:
    """Knowledge graph storage"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        self.load()

    def load(self):
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            self.entities = data.get("entities", {})
            self.relations = data.get("relations", [])

    def save(self):
        data = {"entities": self.entities, "relations": self.relations}
        self.storage_path.write_text(json.dumps(data, indent=2))
        os.chmod(self.storage_path, 0o600)

    def create_entity(self, name: str, entity_type: str, observations: List[str]) -> Dict:
        entity_id = hashlib.sha256(f"{name}:{entity_type}".encode()).hexdigest()[:16]
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "observations": observations,
            "created": datetime.now().isoformat()
        }
        self.save()
        return self.entities[entity_id]

    def add_observations(self, entity_id: str, observations: List[str]):
        if entity_id in self.entities:
            self.entities[entity_id]["observations"].extend(observations)
            self.save()

    def create_relation(self, from_id: str, to_id: str, relation_type: str):
        self.relations.append({"from": from_id, "to": to_id, "type": relation_type})
        self.save()

    def search(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for entity in self.entities.values():
            if (query_lower in entity["name"].lower() or
                query_lower in entity["type"].lower() or
                any(query_lower in obs.lower() for obs in entity["observations"])):
                results.append(entity)
        return results


class MemoryServer:
    """MCP Server for Persistent Memory (Security Hardened)"""

    def __init__(self):
        self.server = Server("memory")
        self.security = get_security_manager()
        self._client_id = hashlib.sha256(f"{os.getpid()}:memory".encode()).hexdigest()[:16]
        self.graph = MemoryGraph(Path.home() / ".dsmil" / "memory_graph.json")
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name="create_entity", description="Create entity with observations",
                     inputSchema={"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}, "observations": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "type", "observations"]}),
                Tool(name="add_observations", description="Add observations to entity",
                     inputSchema={"type": "object", "properties": {"entity_id": {"type": "string"}, "observations": {"type": "array", "items": {"type": "string"}}}, "required": ["entity_id", "observations"]}),
                Tool(name="create_relation", description="Create relation between entities",
                     inputSchema={"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "type": {"type": "string"}}, "required": ["from", "to", "type"]}),
                Tool(name="search_entities", description="Search entities by query",
                     inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
                Tool(name="read_graph", description="Read entire knowledge graph",
                     inputSchema={"type": "object", "properties": {}}),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            if not self.security.check_rate_limit(self._client_id, name):
                return [TextContent(type="text", text="Error: Rate limit exceeded")]

            try:
                if name == "create_entity":
                    entity = self.graph.create_entity(
                        arguments["name"],
                        arguments["type"],
                        arguments["observations"]
                    )
                    self.security.audit_request(name, {"entity_id": entity["id"]}, self._client_id, True)
                    return [TextContent(type="text", text=json.dumps(entity, indent=2))]

                elif name == "add_observations":
                    self.graph.add_observations(arguments["entity_id"], arguments["observations"])
                    self.security.audit_request(name, {"entity_id": arguments["entity_id"]}, self._client_id, True)
                    return [TextContent(type="text", text="Observations added")]

                elif name == "create_relation":
                    self.graph.create_relation(arguments["from"], arguments["to"], arguments["type"])
                    self.security.audit_request(name, {}, self._client_id, True)
                    return [TextContent(type="text", text="Relation created")]

                elif name == "search_entities":
                    results = self.graph.search(arguments["query"])
                    self.security.audit_request(name, {"results": len(results)}, self._client_id, True)
                    return [TextContent(type="text", text=json.dumps(results, indent=2))]

                elif name == "read_graph":
                    data = {"entities": len(self.graph.entities), "relations": len(self.graph.relations)}
                    self.security.audit_request(name, {}, self._client_id, True)
                    return [TextContent(type="text", text=json.dumps(data, indent=2))]

                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(MemoryServer().run())
