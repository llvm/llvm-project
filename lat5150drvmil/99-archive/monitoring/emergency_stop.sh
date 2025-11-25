#!/bin/bash

# DSMIL Emergency Stop Script
# Immediately stops all DSMIL testing and monitoring

echo "🚨 DSMIL EMERGENCY STOP INITIATED"
echo "=================================="
echo "Timestamp: $(date)"

# Stop all DSMIL monitoring processes
echo "🛑 Stopping monitoring processes..."
pkill -f dsmil_comprehensive_monitor
pkill -f dsmil-monitor
pkill -f multi_terminal_launcher

# Stop any DSMIL kernel operations
echo "🔧 Stopping DSMIL kernel operations..."
echo "1786" | sudo -S pkill -f dsmil-72dev 2>/dev/null
echo "1786" | sudo -S rmmod dsmil-72dev 2>/dev/null || echo "   Module not loaded"

# Kill any SMBIOS testing scripts
echo "⚠️  Stopping SMBIOS testing scripts..."
pkill -f discover_dsmil_tokens
pkill -f dell-smbios-token

# Check thermal status
echo "🌡️  Checking thermal status..."
TEMP=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | cut -c1-2)
if [[ -n "$TEMP" && "$TEMP" -gt 85 ]]; then
    echo "   ⚠️  High temperature detected: ${TEMP}°C"
    echo "   💨 Consider cooling system before restart"
else
    echo "   ✅ Temperature normal: ${TEMP}°C"
fi

# Check memory usage
echo "💾 Checking memory status..."
MEM_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
echo "   Memory usage: ${MEM_USAGE}%"

# Save emergency log
LOG_FILE="/tmp/dsmil_emergency_$(date +%Y%m%d_%H%M%S).log"
{
    echo "DSMIL Emergency Stop Log"
    echo "Timestamp: $(date)"
    echo "Thermal status:"
    cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | while read temp; do
        echo "  Zone: $((temp/1000))°C"
    done
    echo "Memory status:"
    free -h
    echo "Load average:"
    uptime
    echo "DSMIL processes:"
    ps aux | grep -E "(dsmil|smbios)" | grep -v grep
    echo "Kernel messages (last 10):"
    dmesg | tail -10
} > "$LOG_FILE"

echo "📝 Emergency log saved: $LOG_FILE"

# Final status
echo ""
echo "✅ EMERGENCY STOP COMPLETE"
echo "   All DSMIL operations stopped"
echo "   System monitoring halted"
echo "   Emergency log created"
echo ""
echo "💡 To restart monitoring:"
echo "   ./monitoring/multi_terminal_launcher.sh"
echo ""
echo "🔍 To check system status:"
echo "   python3 monitoring/dsmil_comprehensive_monitor.py --mode dashboard --json-output"