#!/usr/bin/env python3
"""
Secured Self-Coding Web API with APT-Grade Hardening
Implements defense-in-depth security for localhost-only deployment
"""

import sys
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sock import Sock
from functools import wraps

# Import security hardening
sys.path.insert(0, str(Path(__file__).parent.parent / "02-ai-engine"))
from apt_security_hardening import (
    APTGradeSecurityHardening,
    SecurityConfig,
    SecurityLevel,
    create_security_config,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    SandboxViolation,
    RateLimitExceeded
)
from natural_language_interface import NaturalLanguageInterface

logger = logging.getLogger(__name__)


class SecuredSelfCodingAPI:
    """
    Secured self-coding API with APT-grade hardening

    Security Features:
    - Localhost-only access enforcement
    - Token-based authentication
    - Input validation and sanitization
    - Command execution sandboxing
    - File system protections
    - Rate limiting
    - Intrusion detection
    - Complete audit logging
    """

    def __init__(
        self,
        workspace_root: str = ".",
        port: int = 5001,
        security_level: str = SecurityLevel.HIGH,
        enable_rag: bool = True,
        enable_int8: bool = True,
        enable_learning: bool = True
    ):
        """Initialize secured API"""
        self.workspace_root = workspace_root
        self.port = port

        # Initialize security
        security_config = create_security_config(security_level)
        security_config.workspace_root = workspace_root
        security_config.bind_address = "127.0.0.1"  # Localhost only

        self.security = APTGradeSecurityHardening(security_config)

        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app, origins=[f"http://127.0.0.1:{port}", f"http://localhost:{port}"])
        self.sock = Sock(self.app)

        # Natural language interface
        self.interface = NaturalLanguageInterface(
            workspace_root=workspace_root,
            enable_rag=enable_rag,
            enable_int8=enable_int8,
            enable_learning=enable_learning
        )

        # Setup routes
        self._setup_secured_routes()

        logger.info(f"Secured API initialized (level: {security_level})")

    def _verify_request_security(self):
        """Verify request meets security requirements"""
        # Get client IP
        client_ip = request.remote_addr

        # Verify localhost access
        self.security.verify_localhost_access(client_ip)

        # Check rate limit
        self.security.check_rate_limit(client_ip)

        # Validate token if auth required
        if self.security.config.require_auth:
            token = request.headers.get('Authorization', '').replace('Bearer ', '')

            if not token:
                token = request.args.get('token') or request.json.get('token') if request.json else None

            if not token:
                raise AuthenticationError("No authentication token provided")

            self.security.validate_token(token)

    def secured_endpoint(self, func):
        """Decorator for secured endpoints"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Verify security
                self._verify_request_security()

                # Execute function
                return func(*args, **kwargs)

            except AuthenticationError as e:
                return jsonify({"error": "Authentication failed", "message": str(e)}), 401
            except AuthorizationError as e:
                return jsonify({"error": "Authorization failed", "message": str(e)}), 403
            except ValidationError as e:
                return jsonify({"error": "Validation failed", "message": str(e)}), 400
            except SandboxViolation as e:
                return jsonify({"error": "Sandbox violation", "message": str(e)}), 403
            except RateLimitExceeded as e:
                return jsonify({"error": "Rate limit exceeded", "message": str(e)}), 429
            except Exception as e:
                logger.error(f"Endpoint error: {e}")
                return jsonify({"error": "Internal server error"}), 500

        return wrapper

    def _setup_secured_routes(self):
        """Setup secured routes"""

        @self.app.before_request
        def enforce_localhost():
            """Enforce localhost access on all requests"""
            client_ip = request.remote_addr

            # Log all requests
            self.security._audit_log("REQUEST", {
                "ip": client_ip,
                "method": request.method,
                "path": request.path,
                "user_agent": request.headers.get('User-Agent', 'Unknown')
            })

            # Check if external access attempt
            if client_ip not in ['127.0.0.1', '::1', 'localhost']:
                self.security._audit_log("EXTERNAL_ACCESS_BLOCKED", {
                    "ip": client_ip,
                    "path": request.path
                })
                return jsonify({
                    "error": "Access denied",
                    "message": "Only localhost access allowed. Use SSH tunneling for remote access."
                }), 403

        @self.app.route('/api/health', methods=['GET'])
        def health():
            """Health check (no auth required)"""
            return jsonify({
                "status": "ok",
                "service": "secured-self-coding-api",
                "security": {
                    "localhost_only": self.security.config.localhost_only,
                    "authentication_required": self.security.config.require_auth,
                    "intrusion_detection": self.security.config.enable_intrusion_detection
                }
            })

        @self.app.route('/api/auth/token', methods=['POST'])
        @self.secured_endpoint
        def generate_token():
            """Generate authentication token"""
            user_id = request.json.get('user_id', 'localhost') if request.json else 'localhost'

            token = self.security.generate_token(user_id)

            return jsonify({
                "status": "success",
                "token": token,
                "expires_in": self.security.config.token_expiry_minutes * 60
            })

        @self.app.route('/api/chat', methods=['POST'])
        @self.secured_endpoint
        def chat():
            """Secured chat endpoint"""
            data = request.json
            message = data.get('message', '')

            # Validate message
            message = self.security.validate_message(message)

            # Execute without streaming
            result = None
            for event in self.interface.chat(message, stream=False):
                result = event

            return jsonify({
                "status": "success",
                "result": result.data if result else {},
                "message": result.message if result else "Completed"
            })

        @self.app.route('/api/chat/stream', methods=['POST'])
        @self.secured_endpoint
        def chat_stream():
            """Secured streaming chat endpoint"""
            data = request.json
            message = data.get('message', '')

            # Validate message
            message = self.security.validate_message(message)

            def generate():
                """Generator for SSE"""
                for event in self.interface.chat(message, stream=True):
                    yield f"data: {event.to_json()}\n\n"
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"

            return Response(generate(), mimetype='text/event-stream')

        @self.app.route('/api/task/execute', methods=['POST'])
        @self.secured_endpoint
        def execute_task():
            """Secured task execution"""
            data = request.json
            task = data.get('task', '')

            # Validate task
            task = self.security.validate_message(task)

            result = self.interface.system.execute_task(
                task,
                dry_run=data.get('dry_run', False),
                interactive=False  # Disable interactive for API
            )

            return jsonify({"status": "success", "result": result})

        @self.app.route('/api/self-code', methods=['POST'])
        @self.secured_endpoint
        def self_code():
            """Secured self-coding endpoint"""
            data = request.json
            improvement = data.get('improvement', '')
            target_file = data.get('target_file')

            # Validate inputs
            improvement = self.security.validate_message(improvement)

            if target_file:
                target_file = str(self.security.validate_filepath(target_file))

            result = self.interface.system.code_itself(
                improvement,
                target_file=target_file
            )

            return jsonify({"status": "success", "result": result})

        @self.app.route('/api/learn', methods=['POST'])
        @self.secured_endpoint
        def learn():
            """Secured learning endpoint"""
            data = request.json or {}
            path = data.get('path')

            if path:
                path = str(self.security.validate_filepath(path))

            result = self.interface.system.learn_from_codebase(
                path=path,
                file_pattern=data.get('file_pattern', '**/*.py'),
                max_files=min(data.get('max_files', 100), 500)  # Limit to 500
            )

            return jsonify({"status": "success", "result": result})

        @self.app.route('/api/patterns/search', methods=['GET'])
        @self.secured_endpoint
        def search_patterns():
            """Secured pattern search"""
            query = request.args.get('q', '')

            # Validate query
            query = self.security.validate_message(query)

            patterns = self.interface.system.pattern_db.search_patterns(
                query,
                limit=min(int(request.args.get('limit', 10)), 100)  # Max 100
            )

            return jsonify({
                "status": "success",
                "patterns": [
                    {
                        "name": p.name,
                        "category": p.category,
                        "description": p.description,
                        "quality": p.quality
                    }
                    for p in patterns
                ]
            })

        @self.app.route('/api/security/audit', methods=['GET'])
        @self.secured_endpoint
        def security_audit():
            """Security audit endpoint"""
            report = self.security.audit_system_security()
            return jsonify({"status": "success", "audit": report})

        @self.app.route('/api/security/log', methods=['GET'])
        @self.secured_endpoint
        def security_log():
            """Get security audit log"""
            last_n = min(int(request.args.get('limit', 100)), 1000)
            log = self.security.get_audit_log(last_n)
            return jsonify({"status": "success", "log": log})

        @self.app.route('/api/stats', methods=['GET'])
        @self.secured_endpoint
        def stats():
            """System statistics"""
            stats = self.interface.system.get_stats()
            return jsonify({"status": "success", "stats": stats})

        # WebSocket endpoint (secured)
        @self.sock.route('/ws/chat')
        def chat_websocket(ws):
            """Secured WebSocket chat"""
            try:
                # Verify localhost
                client_ip = request.remote_addr
                self.security.verify_localhost_access(client_ip)

                while True:
                    message = ws.receive()
                    if message is None:
                        break

                    try:
                        data = json.loads(message)

                        if data.get('type') == 'ping':
                            ws.send(json.dumps({"type": "pong"}))
                            continue

                        if data.get('type') == 'chat':
                            # Verify token
                            if self.security.config.require_auth:
                                token = data.get('token')
                                self.security.validate_token(token)

                            # Check rate limit
                            self.security.check_rate_limit(client_ip)

                            # Validate message
                            user_message = self.security.validate_message(
                                data.get('message', '')
                            )

                            # Stream events
                            for event in self.interface.chat(user_message, stream=True):
                                ws.send(event.to_json())

                            ws.send(json.dumps({"type": "done"}))

                    except (AuthenticationError, ValidationError, RateLimitExceeded) as e:
                        ws.send(json.dumps({"type": "error", "message": str(e)}))
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        ws.send(json.dumps({"type": "error", "message": "Internal error"}))

            except AuthorizationError as e:
                ws.send(json.dumps({"type": "error", "message": str(e)}))
                ws.close()

    def run(self, debug: bool = False):
        """Run secured server"""
        logger.info(f"Starting Secured Self-Coding API")
        logger.info(f"Binding to: {self.security.config.bind_address}:{self.port}")
        logger.info(f"Security level: {self.security.config}")

        self.app.run(
            host=self.security.config.bind_address,  # Localhost only
            port=self.port,
            debug=debug,
            threaded=True
        )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Secured Self-Coding Web API")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--security-level", default=SecurityLevel.HIGH,
                       choices=[SecurityLevel.PARANOID, SecurityLevel.HIGH, SecurityLevel.MEDIUM, SecurityLevel.LOW],
                       help="Security level")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8")
    parser.add_argument("--no-learning", action="store_true", help="Disable learning")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run secured API
    api = SecuredSelfCodingAPI(
        workspace_root=args.workspace,
        port=args.port,
        security_level=args.security_level,
        enable_rag=not args.no_rag,
        enable_int8=not args.no_int8,
        enable_learning=not args.no_learning
    )

    print(f"""
    ═══════════════════════════════════════════════════════════
    SECURED SELF-CODING WEB API
    APT-Grade Security Hardening
    ═══════════════════════════════════════════════════════════

    Security Level: {args.security_level}

    Network Security:
      ✅ Localhost-only access (127.0.0.1)
      ✅ External access blocked
      ✅ Firewall-level protection recommended

    Authentication:
      ✅ Token-based authentication
      ✅ Token expiration enabled
      ✅ Session timeout enabled

    Input Validation:
      ✅ Message sanitization
      ✅ Path traversal prevention
      ✅ Command injection prevention
      ✅ SQL injection prevention

    Command Sandboxing:
      ✅ Whitelist/blacklist enforcement
      ✅ Dangerous command blocking
      ✅ Execution timeout protection

    Monitoring:
      ✅ Intrusion detection
      ✅ Complete audit logging
      ✅ Rate limiting
      ✅ Suspicious activity tracking

    API Endpoints:
      http://127.0.0.1:{args.port}/api/health
      http://127.0.0.1:{args.port}/api/auth/token (generate token)
      http://127.0.0.1:{args.port}/api/chat
      http://127.0.0.1:{args.port}/api/security/audit

    WebSocket:
      ws://127.0.0.1:{args.port}/ws/chat

    ⚠️  SECURITY NOTICE:
    - Only localhost (127.0.0.1) access allowed
    - All external connections are blocked
    - Use SSH tunneling for remote access:
      ssh -L {args.port}:127.0.0.1:{args.port} user@host

    Audit Log: {args.workspace}/.security/audit.log
    ═══════════════════════════════════════════════════════════
    """)

    api.run(debug=args.debug)


if __name__ == "__main__":
    main()
