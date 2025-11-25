#!/bin/bash
# Quick Start Script for Local Opus Interface

echo "======================================================"
echo "   🔐 LOCAL OPUS INTERFACE - QUICK START"
echo "======================================================"
echo ""

# Check if server is already running
if lsof -i :8080 >/dev/null 2>&1; then
    echo "✅ Server already running on port 8080"
else
    echo "🚀 Starting Opus interface server..."
    cd /home/john
    python3 opus_server.py > /tmp/opus_server.log 2>&1 &
    sleep 2

    if lsof -i :8080 >/dev/null 2>&1; then
        echo "✅ Server started successfully!"
    else
        echo "❌ Failed to start server. Check /tmp/opus_server.log"
        exit 1
    fi
fi

echo ""
echo "======================================================"
echo "   INTERFACE ACCESS"
echo "======================================================"
echo ""
echo "🌐 Web Interface: http://localhost:8080"
echo ""

# Try to open in browser if available
if command -v xdg-open >/dev/null 2>&1; then
    echo "🔓 Opening in browser..."
    xdg-open http://localhost:8080 2>/dev/null &
elif command -v open >/dev/null 2>&1; then
    echo "🔓 Opening in browser..."
    open http://localhost:8080 2>/dev/null &
else
    echo "📌 Open http://localhost:8080 in your browser"
fi

echo ""
echo "======================================================"
echo "   QUICK REFERENCE"
echo "======================================================"
echo ""
echo "KEYBOARD SHORTCUTS:"
echo "  Ctrl+Enter  - Send message"
echo "  Ctrl+E      - Export chat"
echo "  Ctrl+L      - Clear chat"
echo "  Ctrl+K      - Copy all commands"
echo "  Ctrl+1-8    - Quick actions"
echo "  Up/Down     - Command history"
echo ""
echo "SIDEBAR BUTTONS:"
echo "  📝 Install Commands"
echo "  📄 Full Handoff Document"
echo "  🔍 Opus Context"
echo "  🛡️ APT Defenses"
echo "  ⚠️ Mode 5 Warnings"
echo "  ✅ Build Status"
echo "  🔧 DSMIL Details"
echo "  💻 Hardware Specs"
echo ""
echo "======================================================"
echo "   PROJECT STATUS"
echo "======================================================"
echo ""
echo "✅ Kernel: BUILT at /home/john/linux-6.16.9/arch/x86/boot/bzImage"
echo "✅ DSMIL: 84 devices ready, 584KB driver"
echo "✅ Mode 5: STANDARD (safe, reversible)"
echo "⚠️ WARNING: Never enable PARANOID_PLUS mode!"
echo ""
echo "NEXT STEPS:"
echo "  1. Open the web interface"
echo "  2. Click 'Install Commands' button"
echo "  3. Follow the installation steps"
echo ""
echo "======================================================"
echo "   DOCUMENTATION"
echo "======================================================"
echo ""
echo "Quick reference: cat /home/john/INTERFACE_README.md"
echo "Full handoff:    cat /home/john/COMPLETE_MILITARY_SPEC_HANDOFF.md"
echo "APT defenses:    cat /home/john/APT_ADVANCED_SECURITY_FEATURES.md"
echo "Mode 5 warnings: cat /home/john/MODE5_SECURITY_LEVELS_WARNING.md"
echo ""
echo "======================================================"
echo ""

# Server status
PID=$(lsof -t -i :8080 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Server PID: $PID"
    echo "To stop: kill $PID"
fi

echo ""
echo "Interface ready! Access at http://localhost:8080"
echo ""