#!/usr/bin/env python3
"""
LSP Client for Pyright Language Server

- Implements JSON-RPC 2.0 over stdin/stdout
- Uses asyncio subprocess for non-blocking I/O
- Provides helpers for common LSP operations (hover, refs, definitions, symbols)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]
JsonResult = Optional[Union[JsonDict, List[JsonDict]]]


class LSPClient:
    """
    Language Server Protocol client for Pyright.

    Communicates with Pyright language server via JSON-RPC over stdin/stdout.
    """

    def __init__(self, workspace_root: str, server_command: Optional[List[str]] = None) -> None:
        self.workspace_root = str(Path(workspace_root).resolve())
        self.server_command = server_command or ["pyright-langserver", "--stdio"]

        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id: int = 0
        self.initialized: bool = False

        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    async def start(self) -> bool:
        """Start the LSP server process and perform initialize/initialized handshake."""
        try:
            logger.info("Starting LSP server: %s", " ".join(self.server_command))

            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_root,
            )

            # Background tasks: responses + stderr
            self._read_task = asyncio.create_task(self._read_responses(), name="lsp-read")
            if self.process.stderr:
                self._stderr_task = asyncio.create_task(self._read_stderr(), name="lsp-stderr")

            # Initialize
            init_result = await self._initialize()

            if init_result:
                await self._send_notification("initialized", {})
                self.initialized = True
                logger.info("✅ LSP server initialized successfully")
                return True

            logger.error("❌ LSP server initialization failed")
            return False

        except FileNotFoundError:
            logger.error("LSP server command not found: %s", self.server_command[0])
            return False
        except Exception as e:
            logger.error("Failed to start LSP server: %s", e)
            return False

    async def shutdown(self) -> None:
        """Shutdown LSP server and clean up background tasks."""
        # Try graceful LSP shutdown
        if self.initialized and self.process and self.process.stdin:
            await self._send_request("shutdown", {}, timeout=3.0)
            await self._send_notification("exit", {})

        # Stop background tasks
        for task in (self._read_task, self._stderr_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Terminate process
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("LSP process did not exit, killing")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error("Error during shutdown: %s", e)

        # Cancel any outstanding requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_result(None)
        self._pending_requests.clear()

        self.initialized = False
        logger.info("LSP client shutdown complete")

    # --------------------------------------------------------------------- #
    # Core JSON-RPC plumbing
    # --------------------------------------------------------------------- #

    def _next_request_id(self) -> int:
        self.request_id += 1
        return self.request_id

    async def _send_request(
        self,
        method: str,
        params: JsonDict,
        timeout: float = 5.0,
    ) -> JsonResult:
        """
        Send JSON-RPC request and wait for response.

        Args:
            method: LSP method name (e.g., "textDocument/definition")
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            Response "result" or None on error/timeout.
        """
        if not self.process or not self.process.stdin:
            logger.error("LSP server not running, cannot send request %s", method)
            return None

        request_id = self._next_request_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[request_id] = future

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # LSP uses byte-length for Content-Length, not char length
        try:
            message_bytes = json.dumps(request, separators=(",", ":")).encode("utf-8")
            header = f"Content-Length: {len(message_bytes)}\r\n\r\n".encode("ascii")
            payload = header + message_bytes

            logger.debug("Sending request %s (id=%s)", method, request_id)
            self.process.stdin.write(payload)
            await self.process.stdin.drain()

            result = await asyncio.wait_for(future, timeout=timeout)
            logger.debug("Received result for %s (id=%s)", method, request_id)
            return result

        except asyncio.TimeoutError:
            logger.warning("Request %s (id=%s) timed out after %.1fs", method, request_id, timeout)
            self._pending_requests.pop(request_id, None)
            return None
        except Exception as e:
            logger.error("Error sending request %s: %s", method, e)
            self._pending_requests.pop(request_id, None)
            return None

    async def _send_notification(self, method: str, params: JsonDict) -> None:
        """Send notification (no response expected)."""
        if not self.process or not self.process.stdin:
            logger.debug("LSP server not running, skipping notification %s", method)
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            message_bytes = json.dumps(notification, separators=(",", ":")).encode("utf-8")
            header = f"Content-Length: {len(message_bytes)}\r\n\r\n".encode("ascii")
            payload = header + message_bytes

            logger.debug("Sending notification %s", method)
            self.process.stdin.write(payload)
            await self.process.stdin.drain()
        except Exception as e:
            logger.error("Error sending notification %s: %s", method, e)

    async def _read_responses(self) -> None:
        """Read responses/notifications from LSP server (background task)."""
        if not self.process or not self.process.stdout:
            return

        buffer = b""

        try:
            while True:
                chunk = await self.process.stdout.read(4096)
                if not chunk:
                    # EOF
                    logger.debug("LSP stdout closed")
                    break

                buffer += chunk

                # Process all complete messages in the buffer
                while True:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        break

                    headers = buffer[:header_end].decode("ascii", errors="replace")
                    content_length = 0

                    for line in headers.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            try:
                                content_length = int(line.split(":", 1)[1].strip())
                            except ValueError:
                                logger.error("Invalid Content-Length header: %r", line)
                                content_length = 0
                            break

                    message_start = header_end + 4
                    message_end = message_start + content_length

                    if content_length <= 0 or len(buffer) < message_end:
                        # Incomplete body
                        break

                    message_bytes = buffer[message_start:message_end]
                    buffer = buffer[message_end:]

                    try:
                        message = json.loads(message_bytes.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse LSP JSON: %s", e)
                        logger.debug("Raw message bytes: %r", message_bytes)
                        continue

                    await self._handle_message(message)

        except asyncio.CancelledError:
            logger.debug("LSP read task cancelled")
        except Exception as e:
            logger.error("Error reading LSP responses: %s", e)

    async def _read_stderr(self) -> None:
        """Stream LSP server stderr into logger for debugging."""
        assert self.process is not None
        assert self.process.stderr is not None

        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                logger.debug("LSP STDERR: %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            logger.debug("LSP stderr task cancelled")
        except Exception as e:
            logger.error("Error reading LSP stderr: %s", e)

    async def _handle_message(self, message: JsonDict) -> None:
        """Handle incoming message from LSP server."""
        # Response to a request
        if "id" in message:
            request_id = message.get("id")
            future = self._pending_requests.pop(request_id, None)

            if future is None:
                logger.debug("Received response for unknown request id=%s", request_id)
                return

            if "result" in message:
                future.set_result(message["result"])
            elif "error" in message:
                logger.error("LSP error for id=%s: %s", request_id, message["error"])
                future.set_result(None)
            else:
                future.set_result(None)
            return

        # Notification / server request
        method = message.get("method")
        params = message.get("params", {}) or {}

        if method == "window/logMessage":
            msg = params.get("message", "")
            logger.debug("LSP logMessage: %s", msg)
        elif method == "textDocument/publishDiagnostics":
            # Could be hooked into diagnostics UI; for now just debug log
            uri = params.get("uri", "")
            diagnostics = params.get("diagnostics", [])
            logger.debug("Diagnostics for %s: %d item(s)", uri, len(diagnostics))
        else:
            logger.debug("Unhandled LSP notification/request: %s", method)

    # --------------------------------------------------------------------- #
    # Initialize
    # --------------------------------------------------------------------- #

    async def _initialize(self) -> bool:
        """Send initialize request to LSP server."""
        params: JsonDict = {
            "processId": os.getpid(),
            "rootUri": self._path_to_uri(self.workspace_root),
            "capabilities": {
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "references": {"dynamicRegistration": True},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                },
                "workspace": {
                    "symbol": {"dynamicRegistration": True},
                    "workspaceFolders": True,
                },
            },
            "workspaceFolders": [
                {
                    "uri": self._path_to_uri(self.workspace_root),
                    "name": Path(self.workspace_root).name,
                }
            ],
        }

        try:
            result = await self._send_request("initialize", params, timeout=10.0)
            return result is not None
        except Exception as e:
            logger.error("Initialize request failed: %s", e)
            return False

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _path_to_uri(path: str) -> str:
        """Convert filesystem path to file:// URI."""
        return Path(path).resolve().as_uri()

    async def _notify_did_open(self, file_path: str) -> None:
        """Notify server that a document is opened (simple one-shot version)."""
        try:
            file_path = str(Path(file_path).resolve())
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            params: JsonDict = {
                "textDocument": {
                    "uri": self._path_to_uri(file_path),
                    "languageId": "python",
                    "version": 1,
                    "text": text,
                }
            }

            await self._send_notification("textDocument/didOpen", params)
        except Exception as e:
            logger.error("Error opening document %s: %s", file_path, e)

    # --------------------------------------------------------------------- #
    # High-level LSP operations
    # --------------------------------------------------------------------- #

    async def text_document_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[List[JsonDict]]:
        """
        Get definition of symbol at position.

        Returns:
            List of Location or LocationLink objects, or None.
        """
        if not self.initialized:
            logger.debug("LSP not initialized, skipping definition request")
            return None

        await self._notify_did_open(file_path)

        params: JsonDict = {
            "textDocument": {"uri": self._path_to_uri(file_path)},
            "position": {"line": line - 1, "character": column},
        }

        result = await self._send_request("textDocument/definition", params)
        if not result:
            return None

        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return None

    async def text_document_references(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
    ) -> Optional[List[JsonDict]]:
        """
        Find all references to symbol at position.

        Returns:
            List of Location objects or None.
        """
        if not self.initialized:
            logger.debug("LSP not initialized, skipping references request")
            return None

        await self._notify_did_open(file_path)

        params: JsonDict = {
            "textDocument": {"uri": self._path_to_uri(file_path)},
            "position": {"line": line - 1, "character": column},
            "context": {"includeDeclaration": include_declaration},
        }

        result = await self._send_request("textDocument/references", params)
        return result if isinstance(result, list) else None

    async def workspace_symbol(self, query: str) -> Optional[List[JsonDict]]:
        """
        Search for symbols in workspace.

        Returns:
            List of SymbolInformation objects or None.
        """
        if not self.initialized:
            logger.debug("LSP not initialized, skipping workspace/symbol request")
            return None

        params: JsonDict = {"query": query}
        result = await self._send_request("workspace/symbol", params, timeout=10.0)
        return result if isinstance(result, list) else None

    async def text_document_hover(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[JsonDict]:
        """
        Get hover information at position.

        Returns:
            Hover object ({contents, range}) or None.
        """
        if not self.initialized:
            logger.debug("LSP not initialized, skipping hover request")
            return None

        await self._notify_did_open(file_path)

        params: JsonDict = {
            "textDocument": {"uri": self._path_to_uri(file_path)},
            "position": {"line": line - 1, "character": column},
        }

        result = await self._send_request("textDocument/hover", params)
        return result if isinstance(result, dict) else None


# ------------------------------------------------------------------------- #
# Simple test harness
# ------------------------------------------------------------------------- #

async def test_lsp_client() -> None:
    """Minimal test of the LSP client with Pyright on this directory."""

    here = Path(__file__).resolve()
    workspace_root = str(here.parent)
    test_file = str(here)

    print(f"Workspace root: {workspace_root}")
    client = LSPClient(workspace_root=workspace_root)

    if not await client.start():
        print("Failed to start LSP server")
        return

    print("✅ LSP server started")
    print("⏳ Waiting for Pyright to index workspace...")
    await asyncio.sleep(2)

    # Test workspace symbol search
    print("\n=== Testing workspace/symbol ===")
    symbols = await client.workspace_symbol("LSPClient")
    if symbols:
        print(f"Found {len(symbols)} symbols (showing up to 5):")
        for sym in symbols[:5]:
            loc = sym.get("location", {})
            print(f"  - {sym.get('name')} ({sym.get('kind')}) in {loc.get('uri', '')}")
    else:
        print("No symbols found (Pyright may still be indexing)")

    # Test definition on this file (line/column arbitrary example)
    print("\n=== Testing textDocument/definition ===")
    if os.path.exists(test_file):
        definition = await client.text_document_definition(test_file, 20, 10)
        if definition:
            print(f"Definition found at: {definition[0].get('uri')}")
        else:
            print("No definition found")
    else:
        print(f"Test file not found: {test_file}")

    await client.shutdown()
    print("\n✅ Test complete")


if __name__ == "__main__":
    # Basic default logging; caller can override
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(test_lsp_client())
