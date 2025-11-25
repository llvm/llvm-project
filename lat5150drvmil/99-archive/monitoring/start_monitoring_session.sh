#!/bin/bash

# DSMIL Monitoring Session Starter
# Comprehensive setup for safe DSMIL token testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔬 DSMIL SAFE MONITORING SESSION"
echo "================================="
echo "Dell Latitude 5450 MIL-SPEC SMBIOS Token Testing"
echo "Project: $PROJECT_DIR"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python3 not found"
    exit 1
fi

# Check required modules
python3 -c "import psutil, threading, subprocess" 2>/dev/null || {
    echo "⚠️  Warning: Some Python modules may be missing"
    echo "   Installing basic requirements..."
    pip3 install psutil --user 2>/dev/null || echo "   Could not install psutil"
}

# Check permissions
if ! sudo -n true 2>/dev/null; then
    echo "🔐 Testing sudo access (password may be required)..."
    echo "1786" | sudo -S echo "✅ Sudo access confirmed" || {
        echo "❌ Sudo access required for system monitoring"
        exit 1
    }
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$PROJECT_DIR/monitoring/logs"

# Check system status
echo "🖥️  Checking system status..."
TEMP=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | cut -c1-2 2>/dev/null)
if [[ -n "$TEMP" ]]; then
    TEMP_C=$((TEMP))
    echo "   Temperature: ${TEMP_C}°C"
    if [[ $TEMP_C -gt 85 ]]; then
        echo "   ⚠️  Warning: High temperature detected"
        echo "   Consider cooling system before testing"
    fi
else
    echo "   Temperature: Unknown"
fi

MEM_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
echo "   Memory usage: ${MEM_USAGE}%"

if (( $(echo "$MEM_USAGE > 80" | bc -l) )); then
    echo "   ⚠️  Warning: High memory usage"
fi

echo ""

# Menu system
while true; do
    echo "🎯 MONITORING OPTIONS"
    echo "===================="
    echo "1. 🖥️  Launch Multi-Terminal Dashboard"
    echo "2. 🔍 Single Terminal - Dashboard Mode"
    echo "3. 📊 Single Terminal - Resource Monitor"
    echo "4. 🔢 Single Terminal - Token Monitor"
    echo "5. 🚨 Single Terminal - Alert Monitor"
    echo "6. 🧪 Run Safe Token Test (Dry Run)"
    echo "7. ⚠️  Run Safe Token Test (LIVE - Dangerous!)"
    echo "8. 📝 View Recent Logs"
    echo "9. 🛑 Emergency Stop All"
    echo "0. ❌ Exit"
    echo ""
    read -p "Select option [0-9]: " choice
    
    case $choice in
        1)
            echo "🚀 Launching multi-terminal dashboard..."
            "$SCRIPT_DIR/multi_terminal_launcher.sh"
            ;;
        2)
            echo "🖥️  Starting dashboard mode..."
            cd "$PROJECT_DIR"
            python3 "$SCRIPT_DIR/dsmil_comprehensive_monitor.py" --mode dashboard
            ;;
        3)
            echo "📊 Starting resource monitor..."
            cd "$PROJECT_DIR"
            python3 "$SCRIPT_DIR/dsmil_comprehensive_monitor.py" --mode resources
            ;;
        4)
            echo "🔢 Starting token monitor..."
            cd "$PROJECT_DIR"
            python3 "$SCRIPT_DIR/dsmil_comprehensive_monitor.py" --mode tokens
            ;;
        5)
            echo "🚨 Starting alert monitor..."
            cd "$PROJECT_DIR"
            python3 "$SCRIPT_DIR/dsmil_comprehensive_monitor.py" --mode alerts
            ;;
        6)
            echo "🧪 Starting safe token test (dry run)..."
            cd "$PROJECT_DIR"
            python3 "$SCRIPT_DIR/safe_token_tester.py" --range Range_0480
            echo ""
            echo "✅ Dry run complete. Check logs for results."
            read -p "Press Enter to continue..."
            ;;
        7)
            echo ""
            echo "⚠️  WARNING: LIVE TOKEN TESTING"
            echo "================================"
            echo "This will attempt to modify SMBIOS tokens!"
            echo "This could potentially:"
            echo "- Change system behavior"
            echo "- Trigger unknown responses"
            echo "- Require system restart"
            echo ""
            read -p "Are you absolutely sure? (type 'YES' to confirm): " confirm
            
            if [[ "$confirm" == "YES" ]]; then
                echo "🚨 Starting LIVE token testing..."
                cd "$PROJECT_DIR"
                python3 "$SCRIPT_DIR/safe_token_tester.py" --range Range_0480 --live
                echo ""
                echo "⚠️  LIVE test complete. System may need restart."
                read -p "Press Enter to continue..."
            else
                echo "❌ Live testing cancelled"
            fi
            ;;
        8)
            echo "📝 Recent log files:"
            find "$SCRIPT_DIR/logs" -name "*.log" -mtime -1 2>/dev/null | head -10 | while read log; do
                echo "   $(basename "$log") - $(stat -c %y "$log" | cut -d. -f1)"
            done
            echo ""
            read -p "Enter log filename to view (or Enter to skip): " logfile
            if [[ -n "$logfile" && -f "$SCRIPT_DIR/logs/$logfile" ]]; then
                echo "--- $logfile ---"
                tail -50 "$SCRIPT_DIR/logs/$logfile"
                echo "--- End of log ---"
            fi
            read -p "Press Enter to continue..."
            ;;
        9)
            echo "🛑 Executing emergency stop..."
            "$SCRIPT_DIR/emergency_stop.sh"
            read -p "Press Enter to continue..."
            ;;
        0)
            echo "✅ Exiting monitoring session"
            break
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac
    echo ""
done

echo ""
echo "📋 SESSION SUMMARY"
echo "=================="
echo "Log directory: $SCRIPT_DIR/logs"
echo "Project directory: $PROJECT_DIR"
echo "Emergency stop script: $SCRIPT_DIR/emergency_stop.sh"
echo ""
echo "💡 IMPORTANT REMINDERS:"
echo "• Always monitor system temperature during testing"
echo "• Use dry run mode first to validate token ranges"
echo "• Keep emergency stop script readily available"
echo "• Live testing may require system restart"
echo ""
echo "✅ Monitoring session ended safely"