#!/bin/bash

# DSMIL Documentation Viewer Launcher
# Quick access to all 72-device documentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo "  DSMIL 72-Device Documentation Browser"
echo "  Dell Latitude 5450 MIL-SPEC Military Subsystem Documentation"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Launching documentation browser with AI-enhanced analysis..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Launch the documentation browser
# Default to project root, but allow custom directory
if [ -n "$1" ]; then
    DOC_DIR="$1"
else
    DOC_DIR="$SCRIPT_DIR"
fi

echo "Browsing documentation in: $DOC_DIR"
echo ""
echo "Features:"
echo "  • AI-powered document classification"
echo "  • Automatic PDF text extraction"
echo "  • Intelligent overview generation"
echo "  • Markdown preview with syntax highlighting"
echo ""
echo "Key Documents:"
echo "  • DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md - Full discovery report"
echo "  • DSMIL-AGENT-COORDINATION-PLAN.md - 27-agent strategic plan"
echo "  • docs/DSMIL_ARCHITECTURE_ANALYSIS.md - 72-device architecture"
echo "  • docs/DSMIL_SAFE_PROBING_METHODOLOGY.md - Safe exploration guide"
echo ""

# Launch the browser
cd "$SCRIPT_DIR"
python3 docs/universal_docs_browser_enhanced.py "$DOC_DIR"

echo ""
echo "Documentation browser closed."