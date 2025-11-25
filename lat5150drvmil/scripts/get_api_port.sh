#!/bin/bash
#
# LAT5150 API Port Manager
# Generates and manages a consistent API port across all components
#
# Usage:
#   ./get_api_port.sh           # Get current or generate new port
#   ./get_api_port.sh --reset   # Force generate new port
#   ./get_api_port.sh --check   # Check if current port is available
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT_FILE="${PROJECT_ROOT}/.lat5150_api_port"

# Default port range for random selection (8000-9999, avoiding common ports)
MIN_PORT=8000
MAX_PORT=9999

# Ports to avoid (common services)
AVOID_PORTS=(8080 8443 8888 9000 9090)

# Check if a port is available
is_port_available() {
    local port=$1

    # Check if port is in avoid list
    for avoid in "${AVOID_PORTS[@]}"; do
        if [ "${port}" -eq "${avoid}" ]; then
            return 1
        fi
    done

    # Check if port is in use
    if command -v netstat >/dev/null 2>&1; then
        ! netstat -tuln 2>/dev/null | grep -q ":${port} "
    elif command -v ss >/dev/null 2>&1; then
        ! ss -tuln 2>/dev/null | grep -q ":${port} "
    else
        # Fallback: try to bind to the port
        ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" 2>/dev/null
    fi
}

# Generate a random available port
generate_port() {
    local max_attempts=50
    local attempt=0

    while [ ${attempt} -lt ${max_attempts} ]; do
        local port=$((MIN_PORT + RANDOM % (MAX_PORT - MIN_PORT + 1)))

        if is_port_available "${port}"; then
            echo "${port}"
            return 0
        fi

        ((attempt++))
    done

    # Fallback to sequential search if random fails
    for port in $(seq ${MIN_PORT} ${MAX_PORT}); do
        if is_port_available "${port}"; then
            echo "${port}"
            return 0
        fi
    done

    # Last resort
    echo "8765"
    return 1
}

# Get current port from file or generate new one
get_port() {
    local force_new=${1:-0}

    # If force new or file doesn't exist, generate new port
    if [ "${force_new}" -eq 1 ] || [ ! -f "${PORT_FILE}" ]; then
        local new_port
        new_port=$(generate_port)
        echo "${new_port}" > "${PORT_FILE}"
        chmod 644 "${PORT_FILE}"
        echo "${new_port}"
        return 0
    fi

    # Read existing port
    local existing_port
    existing_port=$(cat "${PORT_FILE}" 2>/dev/null || echo "")

    # Validate existing port
    if [ -z "${existing_port}" ] || ! [[ "${existing_port}" =~ ^[0-9]+$ ]]; then
        local new_port
        new_port=$(generate_port)
        echo "${new_port}" > "${PORT_FILE}"
        echo "${new_port}"
        return 0
    fi

    echo "${existing_port}"
    return 0
}

# Main
case "${1:-}" in
    --reset)
        get_port 1
        ;;
    --check)
        current_port=$(get_port 0)
        if is_port_available "${current_port}"; then
            echo "${current_port}"
            exit 0
        else
            echo "Port ${current_port} is not available" >&2
            exit 1
        fi
        ;;
    *)
        get_port 0
        ;;
esac
