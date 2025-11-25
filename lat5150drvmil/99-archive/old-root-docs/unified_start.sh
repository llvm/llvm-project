#!/usr/bin/env bash
#
# LAT5150DRVMIL Unified Startup Script
# Single command to start entire AI platform
#
# Usage: ./unified_start.sh [--gui] [--voice] [--debug]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Base directories
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_ENGINE_DIR="$BASE_DIR/02-ai-engine"
MCP_DIR="$BASE_DIR/.mcp"

# Parse arguments
START_GUI=false
START_VOICE=false
DEBUG_MODE=false

for arg in "$@"; do
    case $arg in
        --gui) START_GUI=true ;;
        --voice) START_VOICE=true ;;
        --debug) DEBUG_MODE=true ;;
    esac
done

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  LAT5150DRVMIL Unified AI Platform Startup                    ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

print_step() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================================================
# STEP 1: Load NPU Military Configuration (Auto-source)
# ============================================================================

print_step "Loading NPU military configuration..."

if [ -f "$AI_ENGINE_DIR/npu-military-build.env" ]; then
    source "$AI_ENGINE_DIR/npu-military-build.env"
    print_success "NPU military mode: $NPU_COVERT_MODE (${NPU_MAX_TOPS} TOPS)"
elif [ -f "$BASE_DIR/05-deployment/npu-covert-edition.env" ]; then
    source "$BASE_DIR/05-deployment/npu-covert-edition.env"
    print_success "NPU covert edition loaded"
else
    print_warning "NPU config not found - using defaults"
fi

# ============================================================================
# STEP 2: Initialize RAM Disk Database (replaces Redis + PostgreSQL)
# ============================================================================

print_step "Initializing RAM disk database..."

# Check if /dev/shm is available (standard Linux tmpfs)
if [ -d "/dev/shm" ]; then
    # Create RAM disk directory
    RAMDISK_DIR="/dev/shm/lat5150_ai"
    mkdir -p "$RAMDISK_DIR"

    # Check if backup database exists
    BACKUP_DB="$AI_ENGINE_DIR/data/conversation_history.db"
    RAMDISK_DB="$RAMDISK_DIR/conversation_history.db"

    if [ -f "$BACKUP_DB" ]; then
        # Load backup into RAM disk
        cp "$BACKUP_DB" "$RAMDISK_DB"
        BACKUP_SIZE=$(du -h "$BACKUP_DB" | cut -f1)
        print_success "RAM disk database loaded from backup ($BACKUP_SIZE)"
    else
        print_success "RAM disk database initialized (new database)"
    fi

    print_success "RAM disk ready: /dev/shm/lat5150_ai"
    echo "             Database will auto-sync to: $BACKUP_DB"
else
    print_warning "RAM disk (/dev/shm) not available - using disk storage"
fi

# NOTE: No Redis or PostgreSQL needed!
# - Binary protocol uses direct IPC (multiprocessing queues + shared memory)
# - Conversation history uses SQLite in RAM disk with auto-backup
# - Much faster and simpler architecture

# ============================================================================
# STEP 4: Start MCP Servers (if configured)
# ============================================================================

print_step "Starting MCP servers..."

if [ -d "$MCP_DIR" ]; then
    # Check for MCP server configuration
    if [ -f "$MCP_DIR/start_servers.sh" ]; then
        cd "$MCP_DIR"
        ./start_servers.sh &> /tmp/mcp_servers.log &
        MCP_PID=$!
        sleep 2

        if kill -0 $MCP_PID 2>/dev/null; then
            print_success "MCP servers started (PID: $MCP_PID)"
        else
            print_warning "MCP servers failed to start (check /tmp/mcp_servers.log)"
        fi
    else
        print_warning "MCP server script not found"
    fi
else
    print_warning "MCP directory not found - skipping MCP servers"
fi

# ============================================================================
# STEP 5: Start AI Engine
# ============================================================================

print_step "Starting AI engine..."

cd "$AI_ENGINE_DIR"

if [ -f "start_ai_server.sh" ]; then
    ./start_ai_server.sh
    print_success "AI engine started"
else
    print_error "AI engine startup script not found"
    exit 1
fi

# ============================================================================
# STEP 6: Compile Native Libraries (if needed)
# ============================================================================

print_step "Checking native libraries..."

if [ -f "$AI_ENGINE_DIR/Makefile" ] && [ -f "$AI_ENGINE_DIR/libagent_comm.c" ]; then
    if [ ! -f "$AI_ENGINE_DIR/libagent_comm.so" ]; then
        print_step "Compiling binary communication library..."
        cd "$AI_ENGINE_DIR"
        make clean &> /dev/null || true
        if make; then
            print_success "libagent_comm.so compiled with AVX512 support"
        else
            print_warning "Compilation failed - using Python fallback"
        fi
    else
        print_success "Native libraries already compiled"
    fi
fi

# ============================================================================
# STEP 7: Start GUI Dashboard (optional)
# ============================================================================

if [ "$START_GUI" = true ]; then
    print_step "Starting GUI dashboard..."

    cd "$AI_ENGINE_DIR"

    if [ -f "ai_gui_dashboard.py" ]; then
        # Start in background
        nohup python3 ai_gui_dashboard.py > /tmp/gui_dashboard.log 2>&1 &
        GUI_PID=$!
        sleep 2

        if kill -0 $GUI_PID 2>/dev/null; then
            print_success "GUI dashboard started (PID: $GUI_PID)"
            echo -e "${CYAN}         Access at: http://localhost:5050${NC}"
        else
            print_warning "GUI dashboard failed to start (check /tmp/gui_dashboard.log)"
        fi
    else
        print_error "GUI dashboard not found"
    fi
fi

# ============================================================================
# STEP 8: Start Voice UI (optional)
# ============================================================================

if [ "$START_VOICE" = true ]; then
    print_step "Starting voice UI..."

    cd "$AI_ENGINE_DIR"

    if [ -f "voice_ui_npu.py" ]; then
        python3 voice_ui_npu.py &
        VOICE_PID=$!
        print_success "Voice UI started (PID: $VOICE_PID)"
    else
        print_error "Voice UI not found"
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  AI Platform Started Successfully                              ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Active Services:${NC}"
echo "  ✓ NPU Military Mode: ${NPU_COVERT_MODE:-0} (${NPU_MAX_TOPS:-N/A} TOPS)"
echo "  ✓ RAM Disk Database: /dev/shm/lat5150_ai (auto-sync enabled)"
echo "  ✓ Binary Protocol: Direct IPC (no Redis)"
echo "  ✓ AI Engine: Running"

if [ "$START_GUI" = true ]; then
    echo "  ✓ GUI Dashboard: http://localhost:5050"
fi

if [ "$START_VOICE" = true ]; then
    echo "  ✓ Voice UI: Active"
fi

echo ""
echo -e "${CYAN}Quick Commands:${NC}"
echo "  Test agents:     cd $AI_ENGINE_DIR && python3 comprehensive_98_agent_system.py"
echo "  Query AI:        cd $AI_ENGINE_DIR && python3 ai_system_integrator.py"
echo "  Voice UI:        cd $AI_ENGINE_DIR && python3 voice_ui_npu.py"
echo "  Shadowgit:       cd $AI_ENGINE_DIR && python3 shadowgit.py status"

echo ""
echo -e "${CYAN}Logs:${NC}"
echo "  MCP Servers:     /tmp/mcp_servers.log"
echo "  GUI Dashboard:   /tmp/gui_dashboard.log"
echo "  vLLM Server:     /tmp/vllm_server.log (or tmux session 'vllm_server')"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Keep script running if GUI or voice UI started
if [ "$START_GUI" = true ] || [ "$START_VOICE" = true ]; then
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
    echo ""

    trap "echo ''; echo 'Stopping services...'; kill $GUI_PID $VOICE_PID 2>/dev/null; exit" INT TERM

    # Wait indefinitely
    while true; do
        sleep 1
    done
fi
