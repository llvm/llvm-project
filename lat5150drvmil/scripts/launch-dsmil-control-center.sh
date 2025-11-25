#!/bin/bash
################################################################################
# DSMIL Control Center Launcher
################################################################################
# Integrated activation and monitoring system for all 104 DSMIL devices
# Opens multi-window tmux session with:
#   - Device activation interface (104 core + 24 expansion devices)
#   - Operation monitor (800+ operations)
#   - Live system logs
#   - Hardware status monitoring
#   - Control console
#
# Author: LAT5150DRVMIL AI Platform
# Classification: DSMIL Subsystem Management
# Version: 1.0.0
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SESSION_NAME="dsmil-control-center"
# Dynamically detect project root from script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="/tmp/dsmil-logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Banner
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                    DSMIL CONTROL CENTER LAUNCHER                          ║
║                                                                           ║
║              Dell System Military Integration Layer (DSMIL)               ║
║                  LAT5150 MIL-SPEC AI Platform v2.0                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BOLD}System Overview:${NC}"
echo -e "  • ${GREEN}104 Core DSMIL Devices${NC} (0x8000-0x8087)"
echo -e "  • ${CYAN}24 Expansion Devices${NC} (0x8088-0x809F)"
echo -e "  • ${GREEN}800+ Operations${NC} across all implemented devices"
echo -e "  • ${GREEN}300+ Hardware Registers${NC} mapped"
echo -e "  • ${YELLOW}5 Quarantined Devices${NC} (safety enforced)"
echo ""
echo -e "${BOLD}Project Location:${NC} ${CYAN}$PROJECT_ROOT${NC}"
echo ""

# Ensure we're in project root
cd "$PROJECT_ROOT" || {
    echo -e "${RED}ERROR: Project root not found: $PROJECT_ROOT${NC}"
    exit 1
}

# Check if running as root/sudo (required for hardware access)
if [[ $EUID -ne 0 ]]; then
    echo -e "${YELLOW}WARNING: Not running as root. Hardware access may be limited.${NC}"
    echo -e "${YELLOW}Recommend running with: sudo $0${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create log directory
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓${NC} Log directory: $LOG_DIR"

# Build and install DSMIL driver
echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}  DSMIL KERNEL DRIVER - Building & Installing${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ -f "$PROJECT_ROOT/01-source/kernel/build-and-install.sh" ]]; then
    echo -e "${BOLD}Building DSMIL v5.2 driver (104-device architecture)...${NC}"
    echo "  • 104 core devices (Groups 0-8)"
    echo "  • 24 expansion device slots"
    echo "  • 800+ total operations"
    echo "  • 5 quarantined devices (safety enforced)"
    echo "  • Rust memory safety layer"
    echo ""

    # Run the build script
    if bash "$PROJECT_ROOT/01-source/kernel/build-and-install.sh"; then
        echo -e "\n${GREEN}${BOLD}✓ DSMIL driver ready!${NC}"
        DRIVER_LOADED=true
    else
        echo -e "\n${YELLOW}⚠ Driver build/install failed${NC}"
        echo -e "${YELLOW}  Control Center will work in simulation mode${NC}"
        DRIVER_LOADED=false
    fi
else
    echo -e "${YELLOW}⚠ Driver build script not found${NC}"
    echo -e "${YELLOW}  Control Center will work in simulation mode${NC}"
    DRIVER_LOADED=false
fi

echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check for required tools
echo -e "\n${BOLD}Checking prerequisites...${NC}"

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}✗ tmux not found${NC}"
    echo -e "  Install with: ${CYAN}sudo apt install tmux${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} tmux installed"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} python3 installed"

# Check for required files
REQUIRED_FILES=(
    "02-ai-engine/dsmil_guided_activation.py"
    "02-ai-engine/dsmil_operation_monitor.py"
    "DSMIL_DEVICE_CAPABILITIES.json"
    "00-documentation/00-root-docs/DSMIL_CURRENT_REFERENCE.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}✗ Required file missing: $file${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓${NC} All required files present"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "\n${YELLOW}Session '$SESSION_NAME' already exists!${NC}"
    read -p "Kill existing session and restart? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "${GREEN}✓${NC} Killed existing session"
    else
        echo -e "${CYAN}Attaching to existing session...${NC}"
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
fi

# Create helper scripts for each pane
echo -e "\n${BOLD}Creating session components...${NC}"

# 1. Control console script
cat > "$LOG_DIR/control_console.sh" << 'SCRIPT_EOF'
#!/bin/bash
clear
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                        DSMIL CONTROL CONSOLE                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Welcome to the DSMIL Control Center!"
echo ""
echo "ACTIVE WINDOWS:"
echo "  1. Device Activation (Bottom Left)   - 84 device enumeration and activation"
echo "  2. Operation Monitor (Bottom Right)  - 656 operation browsing and execution"
echo "  3. System Logs (Top Right)           - Real-time log monitoring"
echo "  4. Hardware Status (This pane)       - System health and metrics"
echo ""
echo "NAVIGATION:"
echo "  Ctrl+B then Arrow Keys  - Switch between panes"
echo "  Ctrl+B then D           - Detach (keeps running in background)"
echo "  Ctrl+B then [           - Enter scroll mode (Q to exit)"
echo "  Ctrl+B then ?           - Show all tmux keybindings"
echo ""
echo "QUICK ACTIONS:"
echo "  - Device Activation: Use arrow keys to navigate, ENTER to activate"
echo "  - Operation Monitor: Navigate devices/operations, press E to execute"
echo "  - View logs in real-time in the Logs pane"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Show system information
echo "HARDWARE STATUS:"
echo "────────────────────────────────────────────────────────────────────────────"

# Memory
if command -v free &> /dev/null; then
    echo "Memory:"
    free -h | grep -E "Mem|Swap" | sed 's/^/  /'
    echo ""
fi

# CPU
if [[ -f /proc/cpuinfo ]]; then
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    CPU_COUNT=$(grep -c "processor" /proc/cpuinfo)
    echo "CPU: $CPU_MODEL (${CPU_COUNT} cores)"
    echo ""
fi

# Thermal (if available)
if command -v sensors &> /dev/null; then
    echo "Temperature:"
    sensors 2>/dev/null | grep -E "Core|Package" | sed 's/^/  /' || echo "  sensors not configured"
    echo ""
fi

# Check for DSMIL driver
echo "DSMIL DRIVER STATUS:"
MODULE_PATTERN='^dsmil[_-]72dev'
if lsmod | grep -Eq "$MODULE_PATTERN"; then
    echo -e "  ✓ DSMIL driver loaded"
    lsmod | grep -E "$MODULE_PATTERN" | sed 's/^/    /'
else
    echo -e "  ⚠ DSMIL driver NOT loaded"
    echo -e "    Load with: sudo modprobe dsmil_72dev"
fi
echo ""

# Check for device nodes
echo "DSMIL DEVICE NODES:"
if ls /dev/dsmil* &> /dev/null; then
    echo -e "  ✓ Device nodes present:"
    ls -la /dev/dsmil* 2>/dev/null | sed 's/^/    /'
else
    echo -e "  ⚠ No /dev/dsmil* nodes found"
    echo -e "    Hardware access may be limited"
fi
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "LIVE MONITORING:"
echo "────────────────────────────────────────────────────────────────────────────"

# Live monitoring loop
while true; do
    echo -ne "\r$(date '+%Y-%m-%d %H:%M:%S') | "

    # CPU usage
    if command -v mpstat &> /dev/null; then
        CPU_USAGE=$(mpstat 1 1 | awk '/Average/ {print 100-$NF"%"}')
        echo -ne "CPU: $CPU_USAGE | "
    fi

    # Memory usage
    MEM_USAGE=$(free | awk '/Mem/{printf "%.1f%%", $3/$2*100}')
    echo -ne "MEM: $MEM_USAGE | "

    # Load average
    LOAD=$(cat /proc/loadavg | cut -d' ' -f1-3)
    echo -ne "LOAD: $LOAD"

    sleep 2
done
SCRIPT_EOF
chmod +x "$LOG_DIR/control_console.sh"

# 2. Log monitor script (FIXED - robust error handling)
cat > "$LOG_DIR/log_monitor.sh" << 'SCRIPT_EOF'
#!/bin/bash
# DSMIL Log Monitor - Enhanced with robust error handling
set -euo pipefail

LOG_DIR="/tmp/dsmil-logs"
GUIDED_LOG="/tmp/dsmil_guided_activation.log"
OPER_LOG="/tmp/dsmil_operation_monitor.log"
INTEGRATED_LOG="/tmp/dsmil_integrated_activation.log"
ML_LOG="/tmp/dsmil_ml_discovery.log"
KERNEL_LOG="$LOG_DIR/dsmil_kernel.log"
DMESG_PID=""

start_kernel_stream() {
    local args=("$@")
    if ! command -v dmesg &> /dev/null; then
        return 1
    fi
    dmesg "${args[@]}" >> "$KERNEL_LOG" 2>/dev/null &
    DMESG_PID=$!
    sleep 0.2
    if kill -0 "$DMESG_PID" 2>/dev/null; then
        return 0
    fi
    wait "$DMESG_PID" 2>/dev/null || true
    DMESG_PID=""
    return 1
}

cleanup() {
    if [[ -n "${DMESG_PID:-}" ]] && [[ "$DMESG_PID" != "" ]]; then
        kill "$DMESG_PID" 2>/dev/null || true
        wait "$DMESG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Initialize log directory and files
mkdir -p "$LOG_DIR" || true
touch "$GUIDED_LOG" "$OPER_LOG" "$INTEGRATED_LOG" "$ML_LOG" "$KERNEL_LOG" 2>/dev/null || true

clear
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                          DSMIL SYSTEM LOGS                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Monitoring logs from:"
echo "  • $GUIDED_LOG"
echo "  • $OPER_LOG"
echo "  • $INTEGRATED_LOG (ML-enhanced)"
echo "  • $ML_LOG (Discovery)"

# Build log sources array safely
LOG_SOURCES=()
[[ -f "$GUIDED_LOG" ]] && LOG_SOURCES+=("$GUIDED_LOG")
[[ -f "$OPER_LOG" ]] && LOG_SOURCES+=("$OPER_LOG")
[[ -f "$INTEGRATED_LOG" ]] && LOG_SOURCES+=("$INTEGRATED_LOG")
[[ -f "$ML_LOG" ]] && LOG_SOURCES+=("$ML_LOG")

if command -v dmesg &> /dev/null; then
    if start_kernel_stream --follow --human 2>/dev/null || start_kernel_stream --follow 2>/dev/null; then
        echo "  • Kernel log (live dmesg feed)"
        [[ -f "$KERNEL_LOG" ]] && LOG_SOURCES+=("$KERNEL_LOG")
    else
        echo "  ⚠ Kernel log unavailable (insufficient privileges)"
    fi
else
    echo "  ⚠ dmesg tool not found; kernel log disabled"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Multi-tail logs with color coding (robust version)
if [[ ${#LOG_SOURCES[@]} -gt 0 ]]; then
    tail -n0 -F "${LOG_SOURCES[@]}" 2>/dev/null | while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^==\> ]]; then
            echo -e "\033[0;34m$line\033[0m"  # Blue for file headers
        elif [[ "$line" =~ ERROR|CRITICAL|FAIL ]]; then
            echo -e "\033[0;31m$line\033[0m"  # Red
        elif [[ "$line" =~ WARNING|WARN ]]; then
            echo -e "\033[1;33m$line\033[0m"  # Yellow
        elif [[ "$line" =~ SUCCESS|COMPLETE ]]; then
            echo -e "\033[0;32m$line\033[0m"  # Green
        elif [[ "$line" =~ INFO ]]; then
            echo -e "\033[0;36m$line\033[0m"  # Cyan
        else
            echo "$line"
        fi
    done
else
    echo "⚠ No log files available to monitor"
    sleep infinity
fi
SCRIPT_EOF
chmod +x "$LOG_DIR/log_monitor.sh"

echo -e "${GREEN}✓${NC} Session components created"

# Create the tmux session with layout
echo -e "\n${BOLD}Launching Control Center...${NC}"

# Create session with first window
tmux new-session -d -s "$SESSION_NAME" -n "DSMIL-Control-Center"

# Set tmux options for better experience
tmux set-option -g mouse on
tmux set-option -g history-limit 10000

# Split into quad layout (TL, TR, BL, BR)
tmux split-window -h -t "$SESSION_NAME":0        # create right pane
tmux select-pane -t "$SESSION_NAME":0.0          # select left pane
tmux split-window -v -t "$SESSION_NAME":0.0      # split left into top/bottom
tmux select-pane -t "$SESSION_NAME":0.1          # select right pane
tmux split-window -v -t "$SESSION_NAME":0.1      # split right into top/bottom

# Now we have 4 panes:
#   0: Top-Left (Control Console)
#   1: Top-Right (System Logs)
#   2: Bottom-Left (Device Activation)
#   3: Bottom-Right (Operation Monitor)

# Configure pane 0: Control Console
tmux send-keys -t "$SESSION_NAME":0.0 "bash \"$LOG_DIR/control_console.sh\"" C-m

# Configure pane 1: System Logs
tmux send-keys -t "$SESSION_NAME":0.1 "bash \"$LOG_DIR/log_monitor.sh\"" C-m

# Configure pane 2: Device Activation Interface (ML-Enhanced)
tmux send-keys -t "$SESSION_NAME":0.2 "cd \"$PROJECT_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "clear" C-m
cat > "$LOG_DIR/activation_launcher.sh" << 'ACTIV_EOF'
#!/bin/bash
clear
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║              DSMIL DEVICE ACTIVATION INTERFACE                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Select activation mode:"
echo ""
echo "  1. ML-Enhanced Integrated Activation (RECOMMENDED)"
echo "     • Automatic hardware discovery"
echo "     • Machine learning device classification"
echo "     • Intelligent activation sequencing"
echo "     • Real-time safety monitoring"
echo ""
echo "  2. Guided Manual Activation (Classic)"
echo "     • Manual device-by-device activation"
echo "     • Full control over activation process"
echo "     • Browse 84 devices interactively"
echo ""
echo "  3. Discovery Only (No Activation)"
echo "     • Scan hardware for DSMIL devices"
echo "     • ML classification and analysis"
echo "     • Generate discovery report"
echo ""
read -p "Enter choice (1-3): " choice
echo ""

case $choice in
    1)
        echo "Launching ML-Enhanced Integrated Activation..."
        python3 "$PROJECT_ROOT/02-ai-engine/dsmil_integrated_activation.py"
        ;;
    2)
        echo "Launching Guided Manual Activation..."
        python3 "$PROJECT_ROOT/02-ai-engine/dsmil_guided_activation.py"
        ;;
    3)
        echo "Launching Discovery-Only Mode..."
        python3 "$PROJECT_ROOT/02-ai-engine/dsmil_integrated_activation.py" --no-activation
        ;;
    *)
        echo "Invalid choice. Defaulting to ML-Enhanced mode..."
        sleep 2
        python3 "$PROJECT_ROOT/02-ai-engine/dsmil_integrated_activation.py"
        ;;
esac

echo ""
echo "Press ENTER to return to menu..."
read
exec bash "$0"
ACTIV_EOF
chmod +x "$LOG_DIR/activation_launcher.sh"
tmux send-keys -t "$SESSION_NAME":0.2 "bash \"$LOG_DIR/activation_launcher.sh\"" C-m

# Configure pane 3: Operation Monitor
tmux send-keys -t "$SESSION_NAME":0.3 "cd \"$PROJECT_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "clear" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo ''" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'DSMIL Operation Monitor (656 operations)'" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Press ENTER when ready to launch...'" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo ''" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "read && python3 \"$PROJECT_ROOT/02-ai-engine/dsmil_operation_monitor.py\"" C-m

# Select the control console pane by default
tmux select-pane -t "$SESSION_NAME":0.0

# Create a startup instructions file
cat > "$LOG_DIR/session_info.txt" << EOF
DSMIL Control Center Session
Started: $(date)
Session Name: $SESSION_NAME

To reattach to this session later:
  tmux attach-session -t $SESSION_NAME

To detach from session (keeps running):
  Press: Ctrl+B then D

To navigate between panes:
  Press: Ctrl+B then Arrow Keys

To kill this session:
  tmux kill-session -t $SESSION_NAME

Logs saved to: $LOG_DIR
EOF

echo -e "${GREEN}✓${NC} Control Center launched!"
echo ""
echo -e "${BOLD}${CYAN}SESSION READY!${NC}"
echo ""
echo -e "Attaching to session in 2 seconds..."
echo -e "  ${YELLOW}Ctrl+B then D${NC} to detach (keeps running)"
echo -e "  ${YELLOW}Ctrl+B then Arrow Keys${NC} to switch panes"
echo -e "  ${YELLOW}Ctrl+B then ?${NC} for help"
echo ""

sleep 2

# Attach to the session
tmux attach-session -t "$SESSION_NAME"

# After detach/exit
echo ""
echo -e "${GREEN}DSMIL Control Center session detached.${NC}"
echo ""
echo -e "Session is still running in the background."
echo -e "To reattach: ${CYAN}tmux attach-session -t $SESSION_NAME${NC}"
echo -e "To kill: ${CYAN}tmux kill-session -t $SESSION_NAME${NC}"
echo ""
echo -e "Logs available at: ${CYAN}$LOG_DIR${NC}"
echo ""
