#!/bin/bash
#
# DSMIL AI MCP Server Installation Script
#
# This script installs the MCP server and configures it for Claude Desktop
#
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
# Version: 1.0.0

set -e

echo "==============================================="
echo "  DSMIL AI MCP Server Installation"
echo "==============================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[1/4] Installing MCP Python library..."
pip3 install mcp

echo ""
echo "[2/4] Verifying installation..."
python3 -c "import mcp; print(f'MCP version: {mcp.__version__}')"

echo ""
echo "[3/4] Making MCP server executable..."
chmod +x "$SCRIPT_DIR/dsmil_mcp_server.py"

echo ""
echo "[4/4] Generating configuration..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    CONFIG_DIR="$HOME/Library/Application Support/Claude"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    CONFIG_DIR="$APPDATA/Claude"
else
    CONFIG_DIR="$HOME/.config/Claude"
fi

CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"

echo ""
echo "Configuration file location: $CONFIG_FILE"
echo ""
echo "To enable DSMIL AI in Claude Desktop, add this to your configuration:"
echo ""
echo "=============================================="
cat <<EOF
{
  "mcpServers": {
    "dsmil-ai": {
      "command": "python3",
      "args": [
        "$SCRIPT_DIR/dsmil_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "$PROJECT_ROOT"
      }
    }
  }
}
EOF
echo "=============================================="
echo ""

# Offer to create config directory
if [ ! -d "$CONFIG_DIR" ]; then
    read -p "Claude config directory doesn't exist. Create it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$CONFIG_DIR"
        echo "Created: $CONFIG_DIR"
    fi
fi

# Offer to create/update config file
if [ -f "$CONFIG_FILE" ]; then
    echo "⚠️  Config file already exists: $CONFIG_FILE"
    echo "Please manually merge the configuration above."
else
    read -p "Create Claude Desktop config file? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cat > "$CONFIG_FILE" <<EOF
{
  "mcpServers": {
    "dsmil-ai": {
      "command": "python3",
      "args": [
        "$SCRIPT_DIR/dsmil_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "$PROJECT_ROOT"
      }
    }
  }
}
EOF
        echo "✓ Created: $CONFIG_FILE"
    fi
fi

echo ""
echo "==============================================="
echo "  Installation Complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. Restart Claude Desktop (if running)"
echo "2. Check available tools in Claude"
echo "3. Try: 'Use dsmil_get_status to check system status'"
echo ""
echo "Documentation: $SCRIPT_DIR/MCP_SERVER_GUIDE.md"
echo ""
echo "Available tools:"
echo "  - dsmil_ai_query          Query AI with RAG"
echo "  - dsmil_rag_add_file      Add file to knowledge base"
echo "  - dsmil_rag_add_folder    Add folder to knowledge base"
echo "  - dsmil_rag_search        Search knowledge base"
echo "  - dsmil_get_status        Get system status"
echo "  - dsmil_list_models       List AI models"
echo "  - dsmil_rag_list_documents List indexed docs"
echo "  - dsmil_rag_stats         RAG statistics"
echo "  - dsmil_pqc_status        PQC status"
echo "  - dsmil_device_info       Device information"
echo ""
