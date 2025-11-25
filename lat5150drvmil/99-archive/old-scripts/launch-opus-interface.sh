#!/bin/bash
# Standalone Opus Interface Launcher
# Can be run anytime, independent of Claude Code

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Local Opus Interface - Standalone Launcher            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Kill any existing servers
echo -e "${YELLOW}Cleaning up old servers...${NC}"
pkill -9 -f opus_server 2>/dev/null || true
pkill -9 -f "python3.*http.server.*8080" 2>/dev/null || true
pkill -9 -f "python3.*8080" 2>/dev/null || true

# Kill by port if still running (check both 8080 and 9876)
for port in 8080 9876; do
    for pid in $(lsof -t -i:$port 2>/dev/null); do
        echo -e "${YELLOW}Killing process on port $port: $pid${NC}"
        kill -9 $pid 2>/dev/null || true
    done
done

sleep 2

# Verify port 9876 is free
if lsof -i :9876 >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 9876 still in use, waiting...${NC}"
    sleep 3
    # Force kill again
    for pid in $(lsof -t -i:9876 2>/dev/null); do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 1
fi

# Check for full-featured server (best)
if [ -f "/home/john/opus_server_full.py" ]; then
    echo -e "${GREEN}Starting full-featured server...${NC}"
    cd /home/john
    nohup python3 opus_server_full.py > /tmp/opus_server.log 2>&1 &
    SERVER_PID=$!
    sleep 3

    if lsof -i :9876 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Full-featured server started (PID: $SERVER_PID)${NC}"
        echo ""
        echo -e "${BLUE}Features enabled:${NC}"
        echo "  ✅ Document upload (PDF, TXT, MD, code)"
        echo "  ✅ Command execution with safety checks"
        echo "  ✅ File browsing and reading"
        echo "  ✅ Log viewing (kernel, system, dmesg)"
        echo "  ✅ NPU module execution"
        echo "  ✅ System diagnostics"
        echo "  ✅ Kernel status monitoring"
        echo "  ✅ Real-time information"
    else
        echo -e "${YELLOW}⚠️  Full server failed, trying enhanced...${NC}"
        nohup python3 opus_server_enhanced.py > /tmp/opus_server.log 2>&1 &
        sleep 2

        if ! lsof -i :9876 >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  Enhanced failed, using basic...${NC}"
            nohup python3 opus_server.py > /tmp/opus_server.log 2>&1 &
            sleep 2
        fi
    fi
elif [ -f "/home/john/opus_server_enhanced.py" ]; then
    echo -e "${GREEN}Starting enhanced server...${NC}"
    cd /home/john
    nohup python3 opus_server_enhanced.py > /tmp/opus_server.log 2>&1 &
    sleep 3
else
    echo -e "${YELLOW}Using basic server...${NC}"
    cd /home/john
    nohup python3 opus_server.py > /tmp/opus_server.log 2>&1 &
    sleep 2
fi

# Verify server is running
if lsof -i :9876 >/dev/null 2>&1; then
    PID=$(lsof -t -i :9876)
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   Interface Ready!                                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Access at:${NC} http://localhost:9876"
    echo -e "${BLUE}Server PID:${NC} $PID"
    echo -e "${BLUE}Logs:${NC} /tmp/opus_server.log"
    echo ""
    echo -e "${GREEN}Commands:${NC}"
    echo "  Stop server:  kill $PID"
    echo "  View logs:    tail -f /tmp/opus_server.log"
    echo "  Relaunch:     $0"
    echo ""

    # Try to open in browser
    if command -v xdg-open >/dev/null 2>&1; then
        echo -e "${YELLOW}Opening browser...${NC}"
        xdg-open http://localhost:9876 2>/dev/null &
    fi

    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Server running independently of Claude Code            ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}❌ Failed to start server${NC}"
    echo "Check logs: cat /tmp/opus_server.log"
    exit 1
fi
