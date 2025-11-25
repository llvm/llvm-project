#!/usr/bin/env python3
"""
Atomic Red Team API Integration

Provides Natural Language interface to Atomic Red Team security testing framework.
Integrates with DSMIL TEMPEST dashboard for querying, validating, and managing
MITRE ATT&CK security test cases.

Features:
- Query atomics by technique ID, name, description, platform
- Validate atomic test YAML structure
- Refresh tests from GitHub repository
- List available tests by category
- Natural language query processing
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AtomicTest:
    """Represents an Atomic Red Team test"""
    technique_id: str
    name: str
    description: str
    platform: List[str]
    executor: str
    input_arguments: Dict[str, Any]
    dependency_executor_name: Optional[str] = None
    dependencies: Optional[List[Dict]] = None


@dataclass
class QueryResult:
    """Result of an atomic test query"""
    success: bool
    tests: List[Dict[str, Any]]
    count: int
    query: str
    timestamp: str
    error: Optional[str] = None


class AtomicRedTeamAPI:
    """
    Natural Language API for Atomic Red Team MCP server

    Provides high-level interface for security testing operations
    integrated with DSMIL TEMPEST dashboard.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize Atomic Red Team API

        Args:
            data_dir: Directory for storing atomic test data
        """
        self.data_dir = data_dir or "/home/user/LAT5150DRVMIL/03-mcp-servers/atomic-red-team-data"
        self.config_path = "/home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json"

        # Ensure data directory exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # Load MCP configuration
        self.mcp_config = self._load_mcp_config()

        logger.info(f"Atomic Red Team API initialized (data_dir: {self.data_dir})")

    def _load_mcp_config(self) -> Dict:
        """Load MCP server configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}

    def _execute_mcp_command(self, tool: str, arguments: Dict) -> Dict:
        """
        Execute MCP server command

        Args:
            tool: Tool name (query_atomics, refresh_atomics, etc.)
            arguments: Tool arguments

        Returns:
            Command result as dictionary
        """
        try:
            # Build MCP request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": arguments
                }
            }

            # Execute via uvx
            cmd = ["uvx", "atomic-red-team-mcp"]

            # Set environment variables
            env = os.environ.copy()
            env.update({
                "ART_MCP_TRANSPORT": "stdio",
                "ART_DATA_DIR": self.data_dir,
                "ART_EXECUTION_ENABLED": "false"
            })

            # Run command
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )

            # Send request and get response
            stdout, stderr = process.communicate(input=json.dumps(request))

            if stderr:
                logger.warning(f"MCP stderr: {stderr}")

            # Parse response
            if stdout:
                response = json.loads(stdout)
                return response.get("result", {})

            return {"error": "No response from MCP server"}

        except Exception as e:
            logger.error(f"MCP command execution failed: {e}")
            return {"error": str(e)}

    def query_atomics(self,
                     query: Optional[str] = None,
                     technique_id: Optional[str] = None,
                     platform: Optional[str] = None,
                     name_filter: Optional[str] = None) -> QueryResult:
        """
        Query atomic tests with natural language or filters

        Args:
            query: Natural language query (e.g., "mshta atomics for windows")
            technique_id: MITRE ATT&CK technique ID (e.g., "T1059.002")
            platform: Filter by platform (windows, linux, macos)
            name_filter: Filter by test name substring

        Returns:
            QueryResult with matching tests
        """
        try:
            # Build arguments
            arguments = {}

            if technique_id:
                arguments["technique_id"] = technique_id
            if platform:
                arguments["platform"] = platform
            if name_filter:
                arguments["name"] = name_filter
            if query and not (technique_id or platform or name_filter):
                # Parse natural language query
                arguments = self._parse_nl_query(query)

            # Execute query
            result = self._execute_mcp_command("query_atomics", arguments)

            if "error" in result:
                return QueryResult(
                    success=False,
                    tests=[],
                    count=0,
                    query=query or str(arguments),
                    timestamp=datetime.now().isoformat(),
                    error=result["error"]
                )

            # Parse tests
            tests = result.get("tests", [])

            return QueryResult(
                success=True,
                tests=tests,
                count=len(tests),
                query=query or str(arguments),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResult(
                success=False,
                tests=[],
                count=0,
                query=query or str(arguments),
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

    def _parse_nl_query(self, query: str) -> Dict:
        """
        Parse natural language query into structured filters

        Args:
            query: Natural language query string

        Returns:
            Dictionary of filters
        """
        query_lower = query.lower()
        filters = {}

        # Extract platform
        if "windows" in query_lower:
            filters["platform"] = "windows"
        elif "linux" in query_lower:
            filters["platform"] = "linux"
        elif "macos" in query_lower or "mac os" in query_lower:
            filters["platform"] = "macos"

        # Extract technique ID if present (e.g., T1059.002)
        import re
        technique_match = re.search(r'T\d{4}(?:\.\d{3})?', query, re.IGNORECASE)
        if technique_match:
            filters["technique_id"] = technique_match.group(0).upper()

        # Extract name filter (remaining words)
        # Remove platform and technique mentions
        name_parts = query_lower
        for keyword in ["windows", "linux", "macos", "mac os", "for", "on", "atomics", "tests"]:
            name_parts = name_parts.replace(keyword, "")
        if technique_match:
            name_parts = name_parts.replace(technique_match.group(0).lower(), "")

        name_parts = name_parts.strip()
        if name_parts:
            filters["name"] = name_parts

        return filters

    def refresh_atomics(self) -> Dict:
        """
        Refresh atomic tests from GitHub repository

        Returns:
            Status dictionary with success flag and message
        """
        try:
            result = self._execute_mcp_command("refresh_atomics", {})

            if "error" in result:
                return {
                    "success": False,
                    "message": result["error"],
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "success": True,
                "message": "Atomic tests refreshed successfully",
                "timestamp": datetime.now().isoformat(),
                "details": result
            }

        except Exception as e:
            logger.error(f"Refresh failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def validate_atomic(self, yaml_content: str) -> Dict:
        """
        Validate atomic test YAML structure

        Args:
            yaml_content: YAML content to validate

        Returns:
            Validation result dictionary
        """
        try:
            arguments = {"yaml_content": yaml_content}
            result = self._execute_mcp_command("validate_atomic", arguments)

            if "error" in result:
                return {
                    "valid": False,
                    "message": result["error"],
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "valid": result.get("valid", False),
                "message": result.get("message", "Validation complete"),
                "errors": result.get("errors", []),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "valid": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_validation_schema(self) -> Dict:
        """
        Get atomic test validation schema

        Returns:
            Schema dictionary
        """
        try:
            result = self._execute_mcp_command("get_validation_schema", {})

            if "error" in result:
                return {
                    "success": False,
                    "message": result["error"],
                    "schema": None
                }

            return {
                "success": True,
                "schema": result.get("schema"),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Get schema failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "schema": None
            }

    def list_techniques(self) -> Dict:
        """
        List all available MITRE ATT&CK techniques

        Returns:
            Dictionary with technique list
        """
        try:
            # Query all atomics
            result = self.query_atomics(query="")

            if not result.success:
                return {
                    "success": False,
                    "techniques": [],
                    "count": 0,
                    "error": result.error
                }

            # Extract unique techniques
            techniques = {}
            for test in result.tests:
                tech_id = test.get("technique_id", "UNKNOWN")
                if tech_id not in techniques:
                    techniques[tech_id] = {
                        "id": tech_id,
                        "name": test.get("technique_name", ""),
                        "test_count": 0
                    }
                techniques[tech_id]["test_count"] += 1

            return {
                "success": True,
                "techniques": list(techniques.values()),
                "count": len(techniques),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"List techniques failed: {e}")
            return {
                "success": False,
                "techniques": [],
                "count": 0,
                "error": str(e)
            }

    def get_status(self) -> Dict:
        """
        Get Atomic Red Team API status

        Returns:
            Status dictionary
        """
        return {
            "available": True,
            "data_dir": self.data_dir,
            "data_dir_exists": Path(self.data_dir).exists(),
            "execution_enabled": False,  # Always disabled for safety
            "mcp_configured": "atomic-red-team" in self.mcp_config.get("mcpServers", {}),
            "timestamp": datetime.now().isoformat()
        }


# CLI interface for testing
if __name__ == "__main__":
    import sys

    api = AtomicRedTeamAPI()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 atomic_red_team_api.py query <query>")
        print("  python3 atomic_red_team_api.py refresh")
        print("  python3 atomic_red_team_api.py status")
        print("  python3 atomic_red_team_api.py list-techniques")
        sys.exit(1)

    command = sys.argv[1]

    if command == "query":
        if len(sys.argv) < 3:
            print("Error: Query string required")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        result = api.query_atomics(query=query)
        print(json.dumps(asdict(result), indent=2))

    elif command == "refresh":
        result = api.refresh_atomics()
        print(json.dumps(result, indent=2))

    elif command == "status":
        result = api.get_status()
        print(json.dumps(result, indent=2))

    elif command == "list-techniques":
        result = api.list_techniques()
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
