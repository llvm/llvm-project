#!/usr/bin/env python3
"""
DSMIL Fetch MCP Server (Security Hardened)
Based on: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

Web content fetching with SSRF protection.

SECURITY: Blocks internal IPs, validates URLs, rate limits requests.

Author: DSMIL Integration Framework
Version: 1.0.0
"""

import asyncio
import sys
import os
import hashlib
import ipaddress
import urllib.parse
import socket
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import requests
    from bs4 import BeautifulSoup
    import html2text
except ImportError:
    print("ERROR: Missing dependencies. Install: pip3 install mcp requests beautifulsoup4 html2text", file=sys.stderr)
    sys.exit(1)

from mcp_security import get_security_manager


class FetchServer:
    """MCP Server for Web Fetching (Security Hardened with SSRF Protection)"""

    # SSRF Protection: Blocked IP ranges
    BLOCKED_IP_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),      # Private
        ipaddress.ip_network("172.16.0.0/12"),   # Private
        ipaddress.ip_network("192.168.0.0/16"),  # Private
        ipaddress.ip_network("127.0.0.0/8"),     # Loopback
        ipaddress.ip_network("169.254.0.0/16"),  # Link-local
        ipaddress.ip_network("::1/128"),         # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),        # IPv6 private
    ]

    MAX_CONTENT_LENGTH = 1024 * 1024  # 1MB

    def __init__(self):
        self.server = Server("fetch")
        self.security = get_security_manager()
        self._client_id = hashlib.sha256(f"{os.getpid()}:fetch".encode()).hexdigest()[:16]
        self._setup_handlers()

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate URL and check for SSRF"""
        try:
            parsed = urllib.parse.urlparse(url)

            # Must be HTTP/HTTPS
            if parsed.scheme not in ["http", "https"]:
                return False, "Only HTTP/HTTPS URLs allowed"

            # Resolve hostname to IP
            hostname = parsed.hostname
            if not hostname:
                return False, "Invalid hostname"

            # Check if hostname is an IP address
            try:
                ip = ipaddress.ip_address(hostname)
                # Check against blocked ranges
                for blocked_range in self.BLOCKED_IP_RANGES:
                    if ip in blocked_range:
                        return False, f"Access to {ip} is blocked (SSRF protection)"
            except ValueError:
                # Not a direct IP, resolve it
                try:
                    resolved_ip = socket.gethostbyname(hostname)
                    ip = ipaddress.ip_address(resolved_ip)
                    for blocked_range in self.BLOCKED_IP_RANGES:
                        if ip in blocked_range:
                            return False, f"Hostname resolves to blocked IP {resolved_ip} (SSRF protection)"
                except socket.gaierror:
                    return False, "Cannot resolve hostname"

            return True, ""

        except Exception as e:
            return False, f"Invalid URL: {str(e)}"

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="fetch_url",
                    description="Fetch web content and convert to markdown (SSRF protected)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"},
                            "max_length": {"type": "integer", "description": "Max content length (default: 5000)", "default": 5000},
                            "raw": {"type": "boolean", "description": "Return raw HTML (default: false)", "default": False}
                        },
                        "required": ["url"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            if not self.security.check_rate_limit(self._client_id, name):
                return [TextContent(type="text", text="Error: Rate limit exceeded")]

            try:
                if name == "fetch_url":
                    url = arguments.get("url")
                    max_length = arguments.get("max_length", 5000)
                    raw = arguments.get("raw", False)

                    # SECURITY: Validate URL and check for SSRF
                    valid, error = self._validate_url(url)
                    if not valid:
                        self.security.audit_request(name, {"url": url}, self._client_id, False, error)
                        return [TextContent(type="text", text=f"Error: {error}")]

                    # Fetch with timeout and size limit
                    response = requests.get(
                        url,
                        timeout=10,
                        stream=True,
                        headers={"User-Agent": "DSMIL-MCP-Fetch/1.0"}
                    )
                    response.raise_for_status()

                    # Check content length
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > self.MAX_CONTENT_LENGTH:
                        return [TextContent(type="text", text=f"Error: Content too large (max {self.MAX_CONTENT_LENGTH} bytes)")]

                    # Read content with limit
                    content = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        content += chunk
                        if len(content) > self.MAX_CONTENT_LENGTH:
                            return [TextContent(type="text", text="Error: Content exceeds size limit")]

                    # Process content
                    html_content = content.decode('utf-8', errors='ignore')

                    if raw:
                        result = html_content
                    else:
                        # Convert HTML to Markdown
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.ignore_images = False
                        result = h.handle(html_content)

                    # Truncate to max_length
                    if len(result) > max_length:
                        result = result[:max_length] + "\n\n[Content truncated...]"

                    self.security.audit_request(name, {"url": url, "length": len(result)}, self._client_id, True)
                    return [TextContent(type="text", text=result)]

                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except requests.exceptions.RequestException as e:
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                return [TextContent(type="text", text=f"Error fetching URL: {str(e)}")]
            except Exception as e:
                self.security.audit_request(name, arguments, self._client_id, False, str(e))
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(FetchServer().run())
