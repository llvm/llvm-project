#!/bin/bash
###############################################################################
# LAT5150 DRVMIL Tactical AI Sub-Engine - Production Monitoring Script
# Version: 1.0.0
# Purpose: Real-time monitoring of all system components
###############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
REFRESH_INTERVAL=5
API_URL="http://127.0.0.1:5001"
LOG_LINES=20

###############################################################################
# Helper Functions
###############################################################################

get_service_status() {
    if systemctl is-active lat5150-tactical.service &>/dev/null; then
        echo -e "${GREEN}● RUNNING${NC}"
    else
        echo -e "${RED}● STOPPED${NC}"
    fi
}

get_service_uptime() {
    systemctl show lat5150-tactical.service --property=ActiveEnterTimestamp --value 2>/dev/null || echo "N/A"
}

get_api_health() {
    if command -v curl &>/dev/null; then
        RESPONSE=$(curl -s --max-time 2 "$API_URL/health" 2>/dev/null || echo "")
        if [ -n "$RESPONSE" ]; then
            echo -e "${GREEN}HEALTHY${NC}"
        else
            echo -e "${RED}UNREACHABLE${NC}"
        fi
    else
        echo "N/A"
    fi
}

get_api_response_time() {
    if command -v curl &>/dev/null; then
        RESPONSE_TIME=$(curl -s -o /dev/null -w '%{time_total}' --max-time 2 "$API_URL/health" 2>/dev/null || echo "999")
        RESPONSE_MS=$(echo "$RESPONSE_TIME * 1000" | bc 2>/dev/null || echo "999")
        printf "%.0f ms" "$RESPONSE_MS"
    else
        echo "N/A"
    fi
}

get_memory_usage() {
    PID=$(pgrep -f "secured_self_coding_api.py" | head -1)
    if [ -n "$PID" ]; then
        ps -p "$PID" -o %mem,rss 2>/dev/null | tail -1 | awk '{printf "%s%% (%d MB)", $1, $2/1024}'
    else
        echo "N/A"
    fi
}

get_cpu_usage() {
    PID=$(pgrep -f "secured_self_coding_api.py" | head -1)
    if [ -n "$PID" ]; then
        ps -p "$PID" -o %cpu 2>/dev/null | tail -1 | awk '{printf "%s%%", $1}'
    else
        echo "N/A"
    fi
}

get_active_connections() {
    if command -v netstat &>/dev/null; then
        netstat -tn 2>/dev/null | grep ":5001" | grep ESTABLISHED | wc -l
    elif command -v ss &>/dev/null; then
        ss -tn 2>/dev/null | grep ":5001" | grep ESTAB | wc -l
    else
        echo "N/A"
    fi
}

get_vm_tunnels() {
    if command -v netstat &>/dev/null; then
        netstat -tn 2>/dev/null | grep ":22" | grep "192.168.100" | wc -l
    elif command -v ss &>/dev/null; then
        ss -tn 2>/dev/null | grep ":22" | grep "192.168.100" | wc -l
    else
        echo "0"
    fi
}

get_disk_usage() {
    df -h /home/user/LAT5150DRVMIL 2>/dev/null | tail -1 | awk '{printf "%s / %s (%s)", $3, $2, $5}'
}

get_system_load() {
    uptime | awk -F'load average:' '{print $2}' | sed 's/^[ \t]*//' | cut -d',' -f1-3
}

get_dsmil_status() {
    if [ -e /dev/dsmil ]; then
        echo -e "${GREEN}AVAILABLE${NC}"
    else
        echo -e "${YELLOW}NOT FOUND${NC}"
    fi
}

###############################################################################
# Display Functions
###############################################################################

display_header() {
    clear
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          LAT5150 DRVMIL Tactical AI Sub-Engine - Production Monitor          ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${CYAN}Last Update: $(date '+%Y-%m-%d %H:%M:%S')${NC}                     Refresh: ${REFRESH_INTERVAL}s"
    echo ""
}

display_service_status() {
    echo -e "${MAGENTA}━━━ SERVICE STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    STATUS=$(get_service_status)
    UPTIME=$(get_service_uptime)

    echo -e "Service:            $STATUS"
    echo -e "Uptime:             ${CYAN}$UPTIME${NC}"
    echo ""
}

display_api_health() {
    echo -e "${MAGENTA}━━━ API HEALTH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    HEALTH=$(get_api_health)
    RESPONSE_TIME=$(get_api_response_time)

    echo -e "Health Status:      $HEALTH"
    echo -e "Response Time:      ${CYAN}$RESPONSE_TIME${NC}"

    # Get detailed health info if available
    if command -v curl &>/dev/null && command -v jq &>/dev/null; then
        HEALTH_DATA=$(curl -s --max-time 2 "$API_URL/health" 2>/dev/null || echo "{}")

        if [ "$HEALTH_DATA" != "{}" ]; then
            RAG=$(echo "$HEALTH_DATA" | jq -r '.rag_enabled // "unknown"')
            INT8=$(echo "$HEALTH_DATA" | jq -r '.int8_enabled // "unknown"')
            LEARNING=$(echo "$HEALTH_DATA" | jq -r '.learning_enabled // "unknown"')
            SECURITY=$(echo "$HEALTH_DATA" | jq -r '.security_level // "unknown"')

            [ "$RAG" = "true" ] && RAG_COLOR="${GREEN}" || RAG_COLOR="${YELLOW}"
            [ "$INT8" = "true" ] && INT8_COLOR="${GREEN}" || INT8_COLOR="${YELLOW}"
            [ "$LEARNING" = "true" ] && LEARN_COLOR="${GREEN}" || LEARN_COLOR="${YELLOW}"

            echo -e "RAG System:         ${RAG_COLOR}$(echo $RAG | tr '[:lower:]' '[:upper:]')${NC}"
            echo -e "INT8 Optimization:  ${INT8_COLOR}$(echo $INT8 | tr '[:lower:]' '[:upper:]')${NC}"
            echo -e "Learning:           ${LEARN_COLOR}$(echo $LEARNING | tr '[:lower:]' '[:upper:]')${NC}"
            echo -e "Security Level:     ${CYAN}$SECURITY${NC}"
        fi
    fi
    echo ""
}

display_resource_usage() {
    echo -e "${MAGENTA}━━━ RESOURCE USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    CPU=$(get_cpu_usage)
    MEMORY=$(get_memory_usage)
    DISK=$(get_disk_usage)
    LOAD=$(get_system_load)

    echo -e "CPU Usage:          ${CYAN}$CPU${NC}"
    echo -e "Memory Usage:       ${CYAN}$MEMORY${NC}"
    echo -e "Disk Usage:         ${CYAN}$DISK${NC}"
    echo -e "System Load:        ${CYAN}$LOAD${NC}"
    echo ""
}

display_network_status() {
    echo -e "${MAGENTA}━━━ NETWORK STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    CONNECTIONS=$(get_active_connections)
    VM_TUNNELS=$(get_vm_tunnels)

    echo -e "Active Connections: ${CYAN}$CONNECTIONS${NC}"
    echo -e "VM SSH Tunnels:     ${CYAN}$VM_TUNNELS${NC}"

    # Check if listening on correct port
    if netstat -tln 2>/dev/null | grep -q "127.0.0.1:5001"; then
        echo -e "Binding:            ${GREEN}127.0.0.1:5001 (SECURE)${NC}"
    elif netstat -tln 2>/dev/null | grep -q ":5001"; then
        echo -e "Binding:            ${RED}0.0.0.0:5001 (INSECURE!)${NC}"
    else
        echo -e "Binding:            ${YELLOW}NOT LISTENING${NC}"
    fi
    echo ""
}

display_dsmil_status() {
    echo -e "${MAGENTA}━━━ DSMIL HARDWARE SYSTEM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    DSMIL=$(get_dsmil_status)

    echo -e "Device Node:        $DSMIL"

    if [ -e /dev/dsmil ]; then
        PERMS=$(stat -c '%a' /dev/dsmil 2>/dev/null || echo "N/A")
        OWNER=$(stat -c '%U:%G' /dev/dsmil 2>/dev/null || echo "N/A")
        echo -e "Permissions:        ${CYAN}$PERMS${NC}"
        echo -e "Ownership:          ${CYAN}$OWNER${NC}"
    fi

    # Check for recent reconnaissance runs
    RECON_LOG="/home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced.log"
    if [ -f "$RECON_LOG" ]; then
        LAST_SCAN=$(stat -c '%y' "$RECON_LOG" 2>/dev/null | cut -d'.' -f1)
        echo -e "Last Scan:          ${CYAN}$LAST_SCAN${NC}"
    fi
    echo ""
}

display_recent_logs() {
    echo -e "${MAGENTA}━━━ RECENT SERVICE LOGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if command -v journalctl &>/dev/null; then
        journalctl -u lat5150-tactical.service -n "$LOG_LINES" --no-pager --output=short 2>/dev/null | tail -10 | \
        while IFS= read -r line; do
            # Colorize based on log level
            if echo "$line" | grep -iq "error"; then
                echo -e "${RED}$line${NC}"
            elif echo "$line" | grep -iq "warning"; then
                echo -e "${YELLOW}$line${NC}"
            elif echo "$line" | grep -iq "info"; then
                echo -e "${CYAN}$line${NC}"
            else
                echo "$line"
            fi
        done
    else
        echo "journalctl not available"
    fi
    echo ""
}

display_quick_actions() {
    echo -e "${MAGENTA}━━━ QUICK ACTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}[r]${NC} Restart Service  ${CYAN}[s]${NC} Stop Service  ${CYAN}[l]${NC} Full Logs  ${CYAN}[v]${NC} Validate  ${CYAN}[q]${NC} Quit"
}

display_full_logs() {
    clear
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                            FULL SERVICE LOGS                                  ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if command -v journalctl &>/dev/null; then
        journalctl -u lat5150-tactical.service -n 100 --no-pager
    else
        echo "journalctl not available"
    fi

    echo ""
    read -p "Press Enter to continue..." -r
}

restart_service() {
    echo -e "\n${YELLOW}Restarting service...${NC}"
    if sudo systemctl restart lat5150-tactical.service; then
        echo -e "${GREEN}✅ Service restarted successfully${NC}"
    else
        echo -e "${RED}❌ Failed to restart service${NC}"
    fi
    sleep 2
}

stop_service() {
    echo -e "\n${YELLOW}Stopping service...${NC}"
    if sudo systemctl stop lat5150-tactical.service; then
        echo -e "${GREEN}✅ Service stopped successfully${NC}"
    else
        echo -e "${RED}❌ Failed to stop service${NC}"
    fi
    sleep 2
}

run_validation() {
    clear
    VALIDATE_SCRIPT="/home/user/LAT5150DRVMIL/deployment/validate-deployment.sh"
    if [ -f "$VALIDATE_SCRIPT" ] && [ -x "$VALIDATE_SCRIPT" ]; then
        sudo "$VALIDATE_SCRIPT"
    else
        echo -e "${RED}Validation script not found or not executable${NC}"
    fi
    echo ""
    read -p "Press Enter to continue..." -r
}

###############################################################################
# Main Monitor Loop
###############################################################################

monitor_loop() {
    while true; do
        display_header
        display_service_status
        display_api_health
        display_resource_usage
        display_network_status
        display_dsmil_status
        display_recent_logs
        display_quick_actions

        # Non-blocking read with timeout
        read -t "$REFRESH_INTERVAL" -n 1 -s KEY || true

        case "$KEY" in
            r|R)
                restart_service
                ;;
            s|S)
                stop_service
                ;;
            l|L)
                display_full_logs
                ;;
            v|V)
                run_validation
                ;;
            q|Q)
                echo -e "\n${CYAN}Exiting monitor...${NC}"
                exit 0
                ;;
        esac
    done
}

###############################################################################
# Main Execution
###############################################################################

main() {
    # Check if running with appropriate privileges
    if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}Note: Some monitoring features require root privileges${NC}"
        echo -e "${YELLOW}Run with: sudo $0${NC}"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Start monitoring loop
    monitor_loop
}

# Trap Ctrl+C to clean exit
trap "echo -e '\n${CYAN}Exiting monitor...${NC}'; exit 0" INT TERM

main "$@"
