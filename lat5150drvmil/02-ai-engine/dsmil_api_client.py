#!/usr/bin/env python3
"""
DSMIL API Client
================
Python client library for connecting to DSMIL Terminal API server.

Usage:
    from dsmil_api_client import DSMILClient

    client = DSMILClient()
    session = client.create_session("/path/to/project")
    response = client.query(session['session_id'], "Generate a Python function")
    print(response['response'])

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import os
import json
import socket
from typing import Dict, Optional, List
from pathlib import Path


class DSMILClient:
    """
    Client for DSMIL Terminal API

    Provides convenient methods for interacting with the API server.
    """

    def __init__(self, socket_path: Optional[str] = None):
        """
        Initialize DSMIL API client

        Args:
            socket_path: Path to Unix domain socket (default: /tmp/dsmil-ai-{uid}.sock)
        """
        self.uid = os.getuid()
        self.socket_path = socket_path or f"/tmp/dsmil-ai-{self.uid}.sock"
        self.request_id = 0

    def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send JSON-RPC request to server"""
        self.request_id += 1

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }

        # Connect to server
        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(self.socket_path)
        except (FileNotFoundError, ConnectionRefusedError):
            raise ConnectionError(
                f"Cannot connect to DSMIL API server at {self.socket_path}\n"
                f"Start the server with: dsmil-terminal-api --daemon"
            )

        try:
            # Send request
            request_data = json.dumps(request).encode('utf-8')
            client_socket.sendall(request_data)

            # Receive response
            response_data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk

                # Check if we have complete JSON
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    break
                except json.JSONDecodeError:
                    continue

            if not response_data:
                raise ConnectionError("No response from server")

            response = json.loads(response_data.decode('utf-8'))

            # Check for errors
            if "error" in response and response["error"]:
                error = response["error"]
                raise RuntimeError(f"API Error: {error.get('message', 'Unknown error')}")

            return response.get("result", {})

        finally:
            client_socket.close()

    def ping(self) -> Dict:
        """Ping server to check if it's alive"""
        return self._send_request("ping")

    def create_session(self, working_dir: str, project_type: Optional[str] = None) -> Dict:
        """
        Create a new coding session

        Args:
            working_dir: Working directory for the session
            project_type: Optional project type (python, rust, c, etc.)

        Returns:
            Dictionary with session_id, auth_token, etc.
        """
        return self._send_request("create_session", {
            "working_dir": working_dir,
            "project_type": project_type
        })

    def close_session(self, session_id: str) -> Dict:
        """Close a coding session"""
        return self._send_request("close_session", {"session_id": session_id})

    def list_sessions(self) -> List[Dict]:
        """List all active sessions"""
        result = self._send_request("list_sessions")
        return result.get("sessions", [])

    def query(self, session_id: str, prompt: str) -> Dict:
        """
        Execute AI query with session context

        Args:
            session_id: Session ID
            prompt: Query/prompt for the AI

        Returns:
            Dictionary with response and metadata
        """
        return self._send_request("query", {
            "session_id": session_id,
            "prompt": prompt
        })

    def generate_code(self, session_id: str, prompt: str, language: str = "python") -> Dict:
        """
        Generate code based on prompt

        Args:
            session_id: Session ID
            prompt: Code generation prompt
            language: Programming language

        Returns:
            Dictionary with generated code
        """
        return self._send_request("generate_code", {
            "session_id": session_id,
            "prompt": prompt,
            "language": language
        })

    def review_code(self, session_id: str, code: str, filename: str = "code.py") -> Dict:
        """
        Review code for issues and improvements

        Args:
            session_id: Session ID
            code: Code to review
            filename: Filename for context

        Returns:
            Dictionary with review results
        """
        return self._send_request("review_code", {
            "session_id": session_id,
            "code": code,
            "filename": filename
        })

    def execute_code(self, session_id: str, code: str, language: str = "python") -> Dict:
        """
        Execute code safely

        Args:
            session_id: Session ID
            code: Code to execute
            language: Programming language

        Returns:
            Dictionary with execution results
        """
        return self._send_request("execute_code", {
            "session_id": session_id,
            "code": code,
            "language": language
        })

    def analyze_project(self, session_id: str) -> Dict:
        """
        Analyze project structure and context

        Args:
            session_id: Session ID

        Returns:
            Dictionary with project information
        """
        return self._send_request("analyze_project", {"session_id": session_id})

    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> Dict:
        """
        Get conversation history for a session

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return

        Returns:
            Dictionary with conversation history
        """
        params = {"session_id": session_id}
        if limit:
            params["limit"] = limit

        return self._send_request("get_session_history", params)


def main():
    """CLI for API client"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL API Client")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Ping
    subparsers.add_parser('ping', help='Ping server')

    # Create session
    create_parser = subparsers.add_parser('create', help='Create session')
    create_parser.add_argument('directory', help='Working directory')

    # List sessions
    subparsers.add_parser('list', help='List sessions')

    # Query
    query_parser = subparsers.add_parser('query', help='Execute query')
    query_parser.add_argument('session_id', help='Session ID')
    query_parser.add_argument('prompt', nargs='+', help='Prompt')

    # Analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyze project')
    analyze_parser.add_argument('session_id', help='Session ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    client = DSMILClient()

    try:
        if args.command == 'ping':
            result = client.ping()
            print(f"Server is alive: {result}")

        elif args.command == 'create':
            result = client.create_session(args.directory)
            print(f"Session created: {result['session_id']}")
            print(f"Auth token: {result['auth_token']}")

        elif args.command == 'list':
            sessions = client.list_sessions()
            print(f"Active sessions: {len(sessions)}")
            for session in sessions:
                print(f"  {session['session_id']}: {session['working_dir']}")

        elif args.command == 'query':
            prompt = ' '.join(args.prompt)
            result = client.query(args.session_id, prompt)
            print(result['response'])

        elif args.command == 'analyze':
            result = client.analyze_project(args.session_id)
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
