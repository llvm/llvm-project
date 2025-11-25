#!/bin/bash
#
# LAT5150DRVMIL - Dashboard Launcher
#
# Single entry point for the entire AI platform
# Access at: http://localhost:5050
#
# Features:
# - 84 DSMIL devices (79 usable, 5 quarantined)
# - CSNA 2.0 quantum encryption
# - TPM 2.0 hardware crypto (88 algorithms on MIL-SPEC)
# - 22 comprehensive test tasks
# - 7 DSMIL API endpoints
# - TPM status and benchmarking
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_ENGINE_DIR="$SCRIPT_DIR/02-ai-engine"

echo "========================================"
echo "  LAT5150DRVMIL AI TACTICAL PLATFORM"
echo "========================================"
echo ""
echo "Starting unified dashboard..."
echo ""
echo "Features:"
echo "  • 84 DSMIL devices (79 usable)"
echo "  • Quantum encryption (CSNA 2.0)"
echo "  • TPM 2.0 hardware crypto"
echo "  • 22 test tasks integrated"
echo "  • 14 MCP servers configured"
echo "  • Atomic Red Team (MITRE ATT&CK)"
echo "  • Heretic abliteration (5% threshold)"
echo ""
echo "Dashboard URL: http://localhost:5050"
echo ""

cd "$AI_ENGINE_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "   Please install Python 3.10+"
    exit 1
fi

# Launch dashboard
echo "🚀 Launching dashboard..."
echo ""
python3 ai_gui_dashboard.py
