#!/usr/bin/env python3
"""
DSMIL Terminal API Server
==========================
Unix domain socket server providing terminal interface for local system integration.

Enables:
- Right-click "Open DSMIL AI" functionality in file managers
- External process communication via JSON-RPC
- Session management per directory/project
- Context-aware code assistance

Protocol: JSON-RPC 2.0 over Unix domain socket
Socket: /tmp/dsmil-ai-{uid}.sock (user-specific)

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import os
import sys
import json
import socket
import asyncio
import threading
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import DSMIL components
try:
    from code_specialist import CodeSpecialist
    from autonomous_self_improvement import AutonomousSelfImprovement
    CODE_SPECIALIST_AVAILABLE = True
except ImportError:
    CODE_SPECIALIST_AVAILABLE = False

try:
    from rag_system.code_assistant import CodeAssistant
    CODE_ASSISTANT_AVAILABLE = True
except ImportError:
    CODE_ASSISTANT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Coding session for a directory/project"""
    session_id: str
    working_dir: Path
    project_type: Optional[str]
    created_at: datetime
    last_active: datetime
    conversation_history: List[Dict]
    context_loaded: bool = False


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict = None
    id: Optional[int] = None


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response"""
    jsonrpc: str = "2.0"
    result: Any = None
    error: Optional[Dict] = None
    id: Optional[int] = None


class DSMILTerminalAPI:
    """
    Terminal API server for DSMIL AI system integration

    Provides IPC mechanism for external processes to interact with
    the DSMIL AI system via Unix domain socket.
    """

    def __init__(self, socket_path: Optional[str] = None):
        """
        Initialize terminal API server

        Args:
            socket_path: Path to Unix domain socket (default: /tmp/dsmil-ai-{uid}.sock)
        """
        self.uid = os.getuid()
        self.socket_path = socket_path or f"/tmp/dsmil-ai-{self.uid}.sock"

        self.sessions: Dict[str, Session] = {}
        self.auth_tokens: Dict[str, str] = {}  # token -> session_id

        # Initialize components
        if CODE_SPECIALIST_AVAILABLE:
            self.code_specialist = CodeSpecialist()
            logger.info("✓ Code Specialist initialized")
        else:
            self.code_specialist = None
            logger.warning("✗ Code Specialist not available")

        if CODE_ASSISTANT_AVAILABLE:
            self.code_assistant = None  # Initialized per-session
            logger.info("✓ Code Assistant available")
        else:
            logger.warning("✗ Code Assistant not available")

        self.running = False
        self.server_socket = None

        logger.info(f"DSMIL Terminal API initialized (socket: {self.socket_path})")

    def start(self):
        """Start the API server"""
        # Remove existing socket if it exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix domain socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)

        # Set socket permissions (user-only: 0600)
        os.chmod(self.socket_path, 0o600)

        self.server_socket.listen(5)
        self.running = True

        logger.info(f"Server started, listening on {self.socket_path}")

        try:
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    logger.info("Client connected")

                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket,),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")

        finally:
            self.stop()

    def stop(self):
        """Stop the API server"""
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        logger.info("Server stopped")

    def _handle_client(self, client_socket: socket.socket):
        """Handle client connection"""
        try:
            # Receive data
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk

                # Check if we have complete JSON message
                try:
                    json.loads(data.decode('utf-8'))
                    break  # Complete message received
                except json.JSONDecodeError:
                    continue  # Need more data

            if not data:
                return

            # Parse JSON-RPC request
            try:
                request_data = json.loads(data.decode('utf-8'))
                request = JSONRPCRequest(**request_data)
            except Exception as e:
                logger.error(f"Invalid JSON-RPC request: {e}")
                error_response = JSONRPCResponse(
                    error={"code": -32700, "message": "Parse error"},
                    id=None
                )
                client_socket.sendall(json.dumps(asdict(error_response)).encode('utf-8'))
                return

            # Handle request
            response = self._handle_request(request)

            # Send response
            response_data = json.dumps(asdict(response)).encode('utf-8')
            client_socket.sendall(response_data)

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def _handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle JSON-RPC request"""
        try:
            method = request.method
            params = request.params or {}

            # Method dispatch
            if method == "create_session":
                result = self._create_session(params)
            elif method == "close_session":
                result = self._close_session(params)
            elif method == "list_sessions":
                result = self._list_sessions(params)
            elif method == "query":
                result = self._query(params)
            elif method == "generate_code":
                result = self._generate_code(params)
            elif method == "review_code":
                result = self._review_code(params)
            elif method == "execute_code":
                result = self._execute_code(params)
            elif method == "analyze_project":
                result = self._analyze_project(params)
            elif method == "get_session_history":
                result = self._get_session_history(params)
            elif method == "ping":
                result = {"status": "ok", "timestamp": time.time()}
            else:
                return JSONRPCResponse(
                    error={"code": -32601, "message": f"Method not found: {method}"},
                    id=request.id
                )

            return JSONRPCResponse(result=result, id=request.id)

        except Exception as e:
            logger.error(f"Error handling request {request.method}: {e}")
            return JSONRPCResponse(
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
                id=request.id
            )

    def _create_session(self, params: Dict) -> Dict:
        """Create a new coding session"""
        working_dir = Path(params.get('working_dir', '.')).resolve()
        project_type = params.get('project_type')

        # Generate session ID
        session_id = hashlib.sha256(
            f"{working_dir}{time.time()}".encode()
        ).hexdigest()[:16]

        # Create session
        session = Session(
            session_id=session_id,
            working_dir=working_dir,
            project_type=project_type,
            created_at=datetime.now(),
            last_active=datetime.now(),
            conversation_history=[]
        )

        self.sessions[session_id] = session

        # Generate auth token
        auth_token = hashlib.sha256(
            f"{session_id}{os.urandom(16).hex()}".encode()
        ).hexdigest()
        self.auth_tokens[auth_token] = session_id

        logger.info(f"Created session {session_id} for {working_dir}")

        return {
            "session_id": session_id,
            "auth_token": auth_token,
            "working_dir": str(working_dir),
            "project_type": project_type,
            "created_at": session.created_at.isoformat()
        }

    def _close_session(self, params: Dict) -> Dict:
        """Close a coding session"""
        session_id = params.get('session_id')

        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        # Remove session
        del self.sessions[session_id]

        # Remove auth tokens
        self.auth_tokens = {
            token: sid for token, sid in self.auth_tokens.items()
            if sid != session_id
        }

        logger.info(f"Closed session {session_id}")

        return {"status": "ok", "session_id": session_id}

    def _list_sessions(self, params: Dict) -> Dict:
        """List all active sessions"""
        sessions = []
        for session_id, session in self.sessions.items():
            sessions.append({
                "session_id": session_id,
                "working_dir": str(session.working_dir),
                "project_type": session.project_type,
                "created_at": session.created_at.isoformat(),
                "last_active": session.last_active.isoformat(),
                "message_count": len(session.conversation_history)
            })

        return {"sessions": sessions, "count": len(sessions)}

    def _query(self, params: Dict) -> Dict:
        """Execute AI query with session context"""
        session_id = params.get('session_id')
        prompt = params.get('prompt')

        if not prompt:
            raise ValueError("prompt is required")

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Update last active
        session.last_active = datetime.now()

        # Add to conversation history
        session.conversation_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Generate response using code specialist
        if self.code_specialist:
            is_code, task_type, complexity = self.code_specialist.detect_code_task(prompt)

            if is_code:
                # Use code specialist
                model = self.code_specialist.select_model(task_type, complexity)
                response = f"[Using {model} for {task_type} task]\n\n"
                response += f"This would execute the code task: {prompt}"
            else:
                response = f"Non-code query: {prompt}"
        else:
            response = f"Query received: {prompt}"

        # Add response to history
        session.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "response": response,
            "session_id": session_id,
            "message_count": len(session.conversation_history)
        }

    def _generate_code(self, params: Dict) -> Dict:
        """Generate code based on prompt"""
        session_id = params.get('session_id')
        prompt = params.get('prompt')
        language = params.get('language', 'python')

        if not prompt:
            raise ValueError("prompt is required")

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.last_active = datetime.now()

        # TODO: Integrate with actual code generation
        code = f"# Generated code for: {prompt}\n# Language: {language}\n\n"
        code += "def example():\n    pass\n"

        return {
            "code": code,
            "language": language,
            "session_id": session_id
        }

    def _review_code(self, params: Dict) -> Dict:
        """Review code"""
        session_id = params.get('session_id')
        code = params.get('code')
        filename = params.get('filename', 'code.py')

        if not code:
            raise ValueError("code is required")

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.last_active = datetime.now()

        # TODO: Integrate with actual code review
        review = {
            "issues": [],
            "suggestions": ["Add type hints", "Add docstrings"],
            "security": "No issues found",
            "complexity": "Low"
        }

        return review

    def _execute_code(self, params: Dict) -> Dict:
        """Execute code safely"""
        session_id = params.get('session_id')
        code = params.get('code')
        language = params.get('language', 'python')

        if not code:
            raise ValueError("code is required")

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.last_active = datetime.now()

        # TODO: Implement safe code execution
        return {
            "exit_code": 0,
            "stdout": "Execution would happen here",
            "stderr": "",
            "session_id": session_id
        }

    def _analyze_project(self, params: Dict) -> Dict:
        """Analyze project structure and context"""
        session_id = params.get('session_id')

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        working_dir = session.working_dir

        # Detect project type
        project_info = {
            "path": str(working_dir),
            "name": working_dir.name,
            "type": "unknown",
            "files": [],
            "languages": [],
            "git_repo": False
        }

        # Check for git repo
        if (working_dir / ".git").exists():
            project_info["git_repo"] = True

        # Detect project type
        if (working_dir / "setup.py").exists() or (working_dir / "pyproject.toml").exists():
            project_info["type"] = "python"
            project_info["languages"].append("python")

        if (working_dir / "Cargo.toml").exists():
            project_info["type"] = "rust"
            project_info["languages"].append("rust")

        if (working_dir / "package.json").exists():
            project_info["type"] = "javascript"
            project_info["languages"].append("javascript")

        if (working_dir / "CMakeLists.txt").exists() or (working_dir / "Makefile").exists():
            project_info["type"] = "c/c++"
            project_info["languages"].append("c/c++")

        # List files
        try:
            files = list(working_dir.glob("**/*"))
            project_info["files"] = [str(f.relative_to(working_dir)) for f in files[:50]]  # Limit to 50
            project_info["file_count"] = len(files)
        except Exception as e:
            logger.error(f"Error listing files: {e}")

        session.project_type = project_info["type"]
        session.context_loaded = True

        return project_info

    def _get_session_history(self, params: Dict) -> Dict:
        """Get conversation history for a session"""
        session_id = params.get('session_id')
        limit = params.get('limit', 50)

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        history = session.conversation_history[-limit:] if limit else session.conversation_history

        return {
            "session_id": session_id,
            "history": history,
            "total_messages": len(session.conversation_history)
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL Terminal API Server")
    parser.add_argument('--socket', type=str, help="Socket path")
    parser.add_argument('--daemon', action='store_true', help="Run as daemon")

    args = parser.parse_args()

    server = DSMILTerminalAPI(socket_path=args.socket)

    if args.daemon:
        # TODO: Implement proper daemonization
        print(f"Starting daemon on {server.socket_path}")
    else:
        print(f"Starting server on {server.socket_path}")
        print("Press Ctrl+C to stop")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
