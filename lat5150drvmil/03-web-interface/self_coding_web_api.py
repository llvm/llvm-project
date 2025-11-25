#!/usr/bin/env python3
"""
Web API for Self-Coding System
Provides HTTP and WebSocket endpoints for natural language interface
"""

import sys
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sock import Sock
import threading
import queue

# Import natural language interface
sys.path.insert(0, str(Path(__file__).parent.parent / "02-ai-engine"))
from natural_language_interface import (
    NaturalLanguageInterface,
    DisplayEvent,
    DisplayEventType,
    MessageType
)

logger = logging.getLogger(__name__)


class SelfCodingWebAPI:
    """
    Web API for self-coding system

    Features:
    - HTTP endpoints for tasks
    - WebSocket for streaming
    - Session management
    - Multi-client support
    """

    def __init__(
        self,
        workspace_root: str = ".",
        enable_rag: bool = True,
        enable_int8: bool = True,
        enable_learning: bool = True,
        port: int = 5001
    ):
        """
        Initialize web API

        Args:
            workspace_root: Project root
            enable_rag: Enable RAG
            enable_int8: Enable INT8
            enable_learning: Enable learning
            port: Server port
        """
        self.workspace_root = workspace_root
        self.port = port

        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self.sock = Sock(self.app)

        # Natural language interface
        self.interface = NaturalLanguageInterface(
            workspace_root=workspace_root,
            enable_rag=enable_rag,
            enable_int8=enable_int8,
            enable_learning=enable_learning
        )

        # Session management
        self.sessions: Dict[str, Any] = {}
        self.event_queues: Dict[str, queue.Queue] = {}

        # Setup routes
        self._setup_routes()

        logger.info(f"SelfCodingWebAPI initialized on port {port}")

    def _setup_routes(self):
        """Setup HTTP and WebSocket routes"""

        @self.app.route('/api/health', methods=['GET'])
        def health():
            """Health check"""
            return jsonify({"status": "ok", "service": "self-coding-api"})

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """Chat endpoint (non-streaming)"""
            try:
                data = request.json
                message = data.get('message', '')
                session_id = data.get('session_id', 'default')

                if not message:
                    return jsonify({"error": "No message provided"}), 400

                # Execute without streaming
                result = None
                for event in self.interface.chat(message, stream=False):
                    result = event

                return jsonify({
                    "status": "success",
                    "result": result.data if result else {},
                    "message": result.message if result else "Completed"
                })

            except Exception as e:
                logger.error(f"Chat error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/chat/stream', methods=['POST'])
        def chat_stream():
            """Chat endpoint with Server-Sent Events streaming"""
            try:
                data = request.json
                message = data.get('message', '')
                session_id = data.get('session_id', 'default')

                if not message:
                    return jsonify({"error": "No message provided"}), 400

                def generate():
                    """Generator for SSE"""
                    for event in self.interface.chat(message, stream=True):
                        # Send as SSE
                        yield f"data: {event.to_json()}\n\n"

                    # Send completion event
                    yield "data: " + json.dumps({"type": "done"}) + "\n\n"

                return Response(generate(), mimetype='text/event-stream')

            except Exception as e:
                logger.error(f"Stream chat error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/task/execute', methods=['POST'])
        def execute_task():
            """Execute a coding task"""
            try:
                data = request.json
                task = data.get('task', '')
                dry_run = data.get('dry_run', False)
                interactive = data.get('interactive', False)

                if not task:
                    return jsonify({"error": "No task provided"}), 400

                result = self.interface.system.execute_task(
                    task,
                    dry_run=dry_run,
                    interactive=interactive
                )

                return jsonify({
                    "status": "success",
                    "result": result
                })

            except Exception as e:
                logger.error(f"Task execution error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/self-code', methods=['POST'])
        def self_code():
            """Self-coding endpoint"""
            try:
                data = request.json
                improvement = data.get('improvement', '')
                target_file = data.get('target_file')

                if not improvement:
                    return jsonify({"error": "No improvement description provided"}), 400

                result = self.interface.system.code_itself(
                    improvement,
                    target_file=target_file
                )

                return jsonify({
                    "status": "success",
                    "result": result
                })

            except Exception as e:
                logger.error(f"Self-coding error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/learn', methods=['POST'])
        def learn():
            """Learn from codebase"""
            try:
                data = request.json or {}
                path = data.get('path')
                file_pattern = data.get('file_pattern', '**/*.py')
                max_files = data.get('max_files', 100)

                result = self.interface.system.learn_from_codebase(
                    path=path,
                    file_pattern=file_pattern,
                    max_files=max_files
                )

                return jsonify({
                    "status": "success",
                    "result": result
                })

            except Exception as e:
                logger.error(f"Learning error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/patterns/search', methods=['GET'])
        def search_patterns():
            """Search patterns"""
            try:
                query = request.args.get('q', '')
                limit = int(request.args.get('limit', 10))

                if not query:
                    return jsonify({"error": "No query provided"}), 400

                patterns = self.interface.system.pattern_db.search_patterns(
                    query,
                    limit=limit
                )

                return jsonify({
                    "status": "success",
                    "patterns": [
                        {
                            "name": p.name,
                            "category": p.category,
                            "description": p.description,
                            "code_example": p.code_example[:200],
                            "quality": p.quality,
                            "usage_count": p.usage_count
                        }
                        for p in patterns
                    ]
                })

            except Exception as e:
                logger.error(f"Pattern search error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/stats', methods=['GET'])
        def stats():
            """Get system statistics"""
            try:
                stats = self.interface.system.get_stats()

                return jsonify({
                    "status": "success",
                    "stats": stats
                })

            except Exception as e:
                logger.error(f"Stats error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/history', methods=['GET'])
        def history():
            """Get conversation history"""
            try:
                history = self.interface.get_conversation_history()

                return jsonify({
                    "status": "success",
                    "history": history
                })

            except Exception as e:
                logger.error(f"History error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/history/clear', methods=['POST'])
        def clear_history():
            """Clear conversation history"""
            try:
                self.interface.clear_history()

                return jsonify({
                    "status": "success",
                    "message": "History cleared"
                })

            except Exception as e:
                logger.error(f"Clear history error: {e}")
                return jsonify({"error": str(e)}), 500

        # WebSocket endpoint for real-time streaming
        @self.sock.route('/ws/chat')
        def chat_websocket(ws):
            """WebSocket endpoint for chat with streaming"""
            session_id = None

            try:
                while True:
                    # Receive message
                    message = ws.receive()

                    if message is None:
                        break

                    try:
                        data = json.loads(message)
                        msg_type = data.get('type', 'chat')

                        if msg_type == 'ping':
                            ws.send(json.dumps({"type": "pong"}))
                            continue

                        if msg_type == 'chat':
                            user_message = data.get('message', '')
                            session_id = data.get('session_id', 'default')

                            if not user_message:
                                ws.send(json.dumps({
                                    "type": "error",
                                    "message": "No message provided"
                                }))
                                continue

                            # Stream events to WebSocket
                            for event in self.interface.chat(user_message, stream=True):
                                ws.send(event.to_json())

                            # Send done event
                            ws.send(json.dumps({"type": "done"}))

                    except json.JSONDecodeError:
                        ws.send(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON"
                        }))
                    except Exception as e:
                        logger.error(f"WebSocket processing error: {e}")
                        ws.send(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                logger.info(f"WebSocket closed for session: {session_id}")

    def run(self, debug: bool = False):
        """Run the web server"""
        logger.info(f"Starting Self-Coding Web API on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Self-Coding Web API")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
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

    # Create and run API
    api = SelfCodingWebAPI(
        workspace_root=args.workspace,
        enable_rag=not args.no_rag,
        enable_int8=not args.no_int8,
        enable_learning=not args.no_learning,
        port=args.port
    )

    print(f"""
    ═══════════════════════════════════════════════════════════
    Self-Coding Web API Server
    ═══════════════════════════════════════════════════════════

    API Endpoints:
      Health Check:        http://localhost:{args.port}/api/health
      Chat (sync):         POST http://localhost:{args.port}/api/chat
      Chat (streaming):    POST http://localhost:{args.port}/api/chat/stream
      Execute Task:        POST http://localhost:{args.port}/api/task/execute
      Self-Code:           POST http://localhost:{args.port}/api/self-code
      Learn Codebase:      POST http://localhost:{args.port}/api/learn
      Search Patterns:     GET  http://localhost:{args.port}/api/patterns/search
      Statistics:          GET  http://localhost:{args.port}/api/stats
      History:             GET  http://localhost:{args.port}/api/history

    WebSocket:
      Chat Streaming:      ws://localhost:{args.port}/ws/chat

    Features:
      ✅ Natural language interface
      ✅ Real-time streaming
      ✅ Self-coding capabilities
      ✅ Incremental learning
      {'✅' if not args.no_rag else '❌'} RAG integration
      {'✅' if not args.no_int8 else '❌'} INT8 optimization
      {'✅' if not args.no_learning else '❌'} Codebase learning

    Server running at: http://localhost:{args.port}
    ═══════════════════════════════════════════════════════════
    """)

    api.run(debug=args.debug)


if __name__ == "__main__":
    main()
