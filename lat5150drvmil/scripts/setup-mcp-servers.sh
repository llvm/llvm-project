#!/bin/bash
#
# LAT5150DRVMIL - MCP Servers Setup
#
# Automatically installs all external MCP servers:
# - search-tools-mcp (code search)
# - docs-mcp-server (documentation)
# - MetasploitMCP (security testing)
# - mcp-maigret (OSINT)
# - mcp-for-security (23 security tools)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_DIR="$SCRIPT_DIR/03-mcp-servers"

echo "========================================"
echo "  LAT5150DRVMIL MCP SERVERS SETUP"
echo "========================================"
echo ""
echo "This will install 5 external MCP servers:"
echo "  1. search-tools-mcp"
echo "  2. docs-mcp-server"
echo "  3. MetasploitMCP"
echo "  4. mcp-maigret"
echo "  5. mcp-for-security"
echo ""
echo "Plus 6 core Python MCP servers (already ready)"
echo "Plus CVE Scraper (automated security monitoring)"
echo ""

cd "$MCP_DIR"

# Check if setup script exists
if [ ! -f "setup_mcp_servers.sh" ]; then
    echo "❌ Error: setup_mcp_servers.sh not found"
    echo "   Expected location: $MCP_DIR/setup_mcp_servers.sh"
    exit 1
fi

# Run setup
bash setup_mcp_servers.sh

echo ""
echo "========================================"
echo "  MCP SERVERS SETUP COMPLETE"
echo "========================================"
echo ""
echo "All 11 MCP servers are now ready:"
echo ""
echo "Core Python Servers (Ready):"
echo "  ✅ dsmil-ai"
echo "  ✅ sequential-thinking"
echo "  ✅ filesystem"
echo "  ✅ memory"
echo "  ✅ fetch"
echo "  ✅ git"
echo ""
echo "External Servers (Installed):"
echo "  ✅ search-tools-mcp"
echo "  ✅ docs-mcp-server"
echo "  ✅ metasploit"
echo "  ✅ maigret"
echo "  ✅ security-tools"
echo ""
echo "Security Monitoring:"
echo "  ✅ CVE Scraper (automated Telegram monitoring)"
echo "     - Monitors: @cveNotify on Telegram"
echo "     - Schedule: Every 5 minutes + daily at 3 AM"
echo "     - Auto-updates RAG knowledge base"
echo ""
echo "Configuration: 02-ai-engine/mcp_servers_config.json"
echo ""
echo "Next step: Start the dashboard"
echo "  ./start-dashboard.sh"
echo ""
