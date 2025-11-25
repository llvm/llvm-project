#!/bin/bash
echo "🔄 Stopping all Python servers..."
killall -9 python3 2>/dev/null
sleep 2

echo "🚀 Starting Full-Featured Opus Server..."
cd /home/john
python3 opus_server_full.py > /tmp/opus_server.log 2>&1 &
SERVER_PID=$!
sleep 4

if lsof -i :9876 >/dev/null 2>&1; then
    echo "✅ Full-featured server started (PID: $SERVER_PID)"
    echo ""
    echo "📍 URL: http://localhost:9876/WORKING_INTERFACE_FINAL.html"
    echo ""
    echo "Features:"
    echo "  ✅ Text input + 13 buttons"
    echo "  ✅ PDF upload & RAG indexing"
    echo "  ✅ Command execution (no guardrails)"
    echo "  ✅ NPU module testing (all 6 modules)"
    echo "  ✅ File browser & reader"
    echo "  ✅ System monitoring & logs"
    echo "  ✅ Web archiving support"
    echo ""
    echo "🔄 Click red RELOAD button in page after opening!"
    echo "💡 Token usage: 351K / 1M (35.1%)"
else
    echo "❌ Server failed to start - check /tmp/opus_server.log"
    tail -20 /tmp/opus_server.log
fi
