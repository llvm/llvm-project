#!/bin/bash

# DSMIL Multi-Terminal Monitoring Launcher
# Creates multiple terminal windows for comprehensive monitoring

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MONITOR="$SCRIPT_DIR/dsmil_comprehensive_monitor.py"
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Terminal configurations
TERMINALS=(
    "Dashboard:dashboard:DSMIL Dashboard"
    "Resources:resources:System Resources"
    "Tokens:tokens:Token Monitor" 
    "Alerts:alerts:Alert Monitor"
    "Kernel:kernel:Kernel Messages"
)

# Color scheme for terminals
COLORS=("#FF6B6B" "#4ECDC4" "#45B7D1" "#96CEB4" "#FFEAA7")

echo "🚀 Starting DSMIL Multi-Terminal Monitoring System"
echo "=================================================="
echo "This will launch 5 monitoring terminals:"
echo "  1. 📊 Dashboard - Main monitoring overview"
echo "  2. 🖥️  Resources - CPU, Memory, Temperature"
echo "  3. 🔍 Tokens - SMBIOS token monitoring"
echo "  4. 🚨 Alerts - Real-time alert system"
echo "  5. 📝 Kernel - Kernel message monitoring"
echo ""

# Check if python monitor exists
if [[ ! -f "$PYTHON_MONITOR" ]]; then
    echo "❌ Error: Monitor script not found at $PYTHON_MONITOR"
    exit 1
fi

# Check for required tools
command -v gnome-terminal >/dev/null 2>&1 || {
    echo "❌ Error: gnome-terminal not found. This script requires GNOME Terminal."
    echo "   Alternative: Run each monitor manually in separate terminals:"
    echo "   python3 $PYTHON_MONITOR --mode dashboard"
    echo "   python3 $PYTHON_MONITOR --mode resources"
    echo "   python3 $PYTHON_MONITOR --mode tokens"
    echo "   python3 $PYTHON_MONITOR --mode alerts"
    exit 1
}

# Function to launch terminal
launch_terminal() {
    local name="$1"
    local mode="$2"
    local title="$3"
    local color="$4"
    local log_file="$LOG_DIR/${mode}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🖥️  Launching $title terminal..."
    
    if [[ "$mode" == "kernel" ]]; then
        # Special kernel monitoring terminal
        gnome-terminal --title="$title" --geometry=120x30 -- bash -c "
            echo '🔍 DSMIL Kernel Message Monitor';
            echo '==============================';
            echo 'Monitoring kernel messages for DSMIL activity...';
            echo '';
            while true; do
                echo '--- \$(date) ---';
                echo '1786' | sudo -S dmesg -T | grep -i -E '(dsmil|dell|smbios)' | tail -5 2>/dev/null || echo 'No recent messages';
                echo '';
                sleep 5;
            done | tee '$log_file'
        " &
    else
        # Python monitoring terminals
        gnome-terminal --title="$title" --geometry=120x30 -- bash -c "
            cd '$SCRIPT_DIR/..';
            echo '🚀 Starting $title...';
            echo 'Press Ctrl+C to stop';
            echo '';
            python3 '$PYTHON_MONITOR' --mode '$mode' 2>&1 | tee '$log_file';
            echo '';
            echo '✅ $title stopped. Press Enter to close...';
            read;
        " &
    fi
    
    # Store PID for cleanup
    MONITOR_PIDS+=($!)
    sleep 1
}

# Array to store monitor PIDs
MONITOR_PIDS=()

# Launch all terminals
for i in "${!TERMINALS[@]}"; do
    IFS=':' read -r name mode title <<< "${TERMINALS[$i]}"
    color="${COLORS[$i]}"
    launch_terminal "$name" "$mode" "$title" "$color"
done

echo ""
echo "✅ All monitoring terminals launched!"
echo "📁 Logs are being saved to: $LOG_DIR"
echo ""
echo "🛑 To stop all monitors, press Ctrl+C or run:"
echo "   pkill -f dsmil_comprehensive_monitor"
echo ""
echo "💡 Usage tips:"
echo "   • Dashboard terminal shows overall system status"
echo "   • Watch the Alerts terminal during token testing"
echo "   • Resources terminal shows thermal/CPU warnings"
echo "   • Tokens terminal tracks SMBIOS token changes"
echo ""

# Wait for user interrupt
trap 'echo -e "\\n🛑 Stopping all monitors..."; pkill -f dsmil_comprehensive_monitor; exit 0' INT

echo "⏳ Monitoring active. Press Ctrl+C to stop all terminals."
while true; do
    sleep 1
done