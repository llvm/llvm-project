#!/bin/bash
#
# LAT5150 DRVMIL - API Environment Variables and Helper Functions
# Source this file in .bashrc to make API accessible in every shell session
#
# Add to .bashrc:
#   source /path/to/LAT5150DRVMIL/deployment/lat5150-api-env.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# API Configuration
export LAT5150_API_URL="http://localhost:80"
export LAT5150_API_PORT="80"
export LAT5150_WORKSPACE="${PROJECT_ROOT}"

# Atomic Red Team Configuration
export ART_DATA_DIR="${PROJECT_ROOT}/03-mcp-servers/atomic-red-team-data"
export ART_EXECUTION_ENABLED="false"

# Helper function: Query the unified API with natural language
lat5150_query() {
    if [ -z "$1" ]; then
        echo "Usage: lat5150_query <natural_language_query>"
        echo ""
        echo "Examples:"
        echo "  lat5150_query 'Show me atomic tests for T1059.002'"
        echo "  lat5150_query 'Find mshta atomics for Windows'"
        echo "  lat5150_query 'List all MITRE ATT&CK techniques'"
        echo "  lat5150_query 'Check system health'"
        return 1
    fi

    curl -s -X POST "${LAT5150_API_URL}/api/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$*\"}" | python3 -m json.tool
}

# Helper function: Get API self-awareness report
lat5150_status() {
    echo "LAT5150 Unified Tactical API Status"
    echo "===================================="
    echo ""

    # Check if service is running
    if systemctl is-active --quiet lat5150-unified-api 2>/dev/null; then
        echo "Service: RUNNING"
    else
        echo "Service: NOT RUNNING"
        echo ""
        echo "Start with: sudo systemctl start lat5150-unified-api"
        return 1
    fi

    # Check API response
    if curl -s "${LAT5150_API_URL}/api/self-awareness" > /dev/null 2>&1; then
        echo "API:     RESPONDING"
    else
        echo "API:     NOT RESPONDING"
        return 1
    fi

    echo ""
    echo "Components:"
    echo "-----------"

    # Parse self-awareness report
    curl -s "${LAT5150_API_URL}/api/self-awareness" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    components = data.get('components', {})
    for name, status in components.items():
        status_str = '✓' if status else '✗'
        print(f'{status_str} {name.replace(\"_\", \" \").title()}')

    caps = data.get('legacy_capabilities', {})
    if caps:
        print(f'\nCapabilities: {caps.get(\"total_capabilities\", 0)}')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
"

    echo ""
    echo "URL: ${LAT5150_API_URL}"
}

# Helper function: List all available atomic tests for a technique
lat5150_atomic_list() {
    if [ -z "$1" ]; then
        echo "Usage: lat5150_atomic_list <technique_id>"
        echo ""
        echo "Examples:"
        echo "  lat5150_atomic_list T1059.002"
        echo "  lat5150_atomic_list T1003"
        return 1
    fi

    lat5150_query "Show me atomic tests for $1"
}

# Helper function: Search atomic tests by platform
lat5150_atomic_search() {
    local platform="$1"
    local query="${2:-}"

    if [ -z "$platform" ]; then
        echo "Usage: lat5150_atomic_search <platform> [query]"
        echo ""
        echo "Platforms: Windows, Linux, macOS"
        echo ""
        echo "Examples:"
        echo "  lat5150_atomic_search Windows mshta"
        echo "  lat5150_atomic_search Linux bash"
        echo "  lat5150_atomic_search macOS"
        return 1
    fi

    if [ -z "$query" ]; then
        lat5150_query "Find all atomic tests for ${platform}"
    else
        lat5150_query "Find ${query} atomics for ${platform}"
    fi
}

# Helper function: Refresh Atomic Red Team tests
lat5150_atomic_refresh() {
    echo "Refreshing Atomic Red Team tests from GitHub..."
    lat5150_query "Refresh atomic red team tests"
}

# Helper function: Get service logs
lat5150_logs() {
    local lines="${1:-20}"
    sudo journalctl -u lat5150-unified-api -n "$lines" --no-pager
}

# Helper function: Follow service logs
lat5150_logs_follow() {
    sudo journalctl -u lat5150-unified-api -f
}

# Helper function: Show all capabilities
lat5150_capabilities() {
    echo "LAT5150 Unified Tactical API - Available Capabilities"
    echo "====================================================="
    echo ""

    curl -s "${LAT5150_API_URL}/api/self-awareness" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    caps = data.get('legacy_capabilities', {})
    total = caps.get('total_capabilities', 0)
    by_cat = caps.get('by_category', {})

    print(f'Total Capabilities: {total}\n')

    for category, count in sorted(by_cat.items()):
        cat_name = category.replace('_', ' ').title()
        print(f'{cat_name}: {count}')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
"
}

# Helper function: Quick API test
lat5150_test() {
    echo "Testing LAT5150 Unified Tactical API..."
    echo ""

    # Test 1: API health
    echo "[1/3] Testing API health..."
    if curl -s "${LAT5150_API_URL}/api/self-awareness" > /dev/null; then
        echo "  ✓ API is responding"
    else
        echo "  ✗ API is not responding"
        return 1
    fi

    # Test 2: Atomic Red Team integration
    echo "[2/3] Testing Atomic Red Team..."
    if curl -s "${LAT5150_API_URL}/api/self-awareness" | grep -q '"atomic_red_team": true'; then
        echo "  ✓ Atomic Red Team is loaded"
    else
        echo "  ✗ Atomic Red Team is not loaded"
    fi

    # Test 3: Natural language query
    echo "[3/3] Testing natural language query..."
    response=$(lat5150_query "List all MITRE ATT&CK techniques" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  ✓ Natural language processing working"
    else
        echo "  ✗ Natural language query failed"
    fi

    echo ""
    echo "Test complete!"
}

# Show quick help on source
if [ -n "$PS1" ]; then
    # Interactive shell - show help
    cat <<EOF

LAT5150 Unified Tactical API - Helper Functions Loaded
======================================================

Available commands:
  lat5150_query <query>              Query API with natural language
  lat5150_status                     Show API status and components
  lat5150_atomic_list <technique>    List tests for MITRE technique
  lat5150_atomic_search <platform>   Search tests by platform
  lat5150_atomic_refresh             Refresh tests from GitHub
  lat5150_capabilities               Show all 20 capabilities
  lat5150_logs [lines]               Show service logs
  lat5150_logs_follow                Follow service logs in real-time
  lat5150_test                       Run API health tests

Examples:
  lat5150_query 'Show me atomic tests for T1059.002'
  lat5150_atomic_list T1003
  lat5150_atomic_search Windows powershell
  lat5150_status

API URL: ${LAT5150_API_URL}

EOF
fi

# Export functions
export -f lat5150_query
export -f lat5150_status
export -f lat5150_atomic_list
export -f lat5150_atomic_search
export -f lat5150_atomic_refresh
export -f lat5150_logs
export -f lat5150_logs_follow
export -f lat5150_capabilities
export -f lat5150_test
