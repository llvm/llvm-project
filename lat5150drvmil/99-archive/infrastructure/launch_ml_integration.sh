#!/bin/bash
#
# DSMIL ML Integration Launcher v1.0
# Launch script for DSMIL Enhanced Learning System integration
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSMIL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${HOME}/.local/share/claude/venv"
LOG_DIR="${DSMIL_ROOT}/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Docker for PostgreSQL
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required for PostgreSQL but not installed"
        exit 1
    fi
    
    # Check if PostgreSQL container is running
    if ! docker ps | grep -q claude-postgres; then
        log_warn "PostgreSQL container not running, attempting to start..."
        if ! docker start claude-postgres 2>/dev/null; then
            log_error "Failed to start PostgreSQL container. Run setup first."
            exit 1
        fi
    fi
    
    # Create log directory
    mkdir -p "${LOG_DIR}"
    
    log_info "Dependencies OK"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Activate Claude virtual environment if available
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        source "${VENV_PATH}/bin/activate"
        log_info "Activated Claude virtual environment"
    else
        log_warn "Claude virtual environment not found, using system Python"
    fi
    
    # Install required packages if needed
    log_debug "Checking Python packages..."
    python3 -c "import asyncpg, numpy" 2>/dev/null || {
        log_info "Installing required Python packages..."
        pip3 install asyncpg numpy
    }
    
    # Optional: sklearn for advanced ML
    python3 -c "import sklearn" 2>/dev/null || {
        log_warn "sklearn not available, using basic ML fallbacks"
    }
}

# Check PostgreSQL connection
check_database() {
    log_info "Checking database connection..."
    
    python3 -c "
import asyncio
import asyncpg

async def test_connection():
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5433,
            database='claude_auth',
            user='claude_auth',
            password='claude_auth_pass'
        )
        await conn.fetchval('SELECT 1')
        await conn.close()
        print('Database connection OK')
        return True
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False

result = asyncio.run(test_connection())
exit(0 if result else 1)
" || {
        log_error "Database connection failed"
        log_info "Try: docker ps | grep claude-postgres"
        exit 1
    }
}

# Show system status
show_status() {
    log_info "Getting system status..."
    cd "${SCRIPT_DIR}"
    python3 dsmil_ml_orchestrator.py --status
}

# Submit a test task
submit_test_task() {
    local task="${1:-Monitor DSMIL device thermal conditions}"
    log_info "Submitting test task: ${task}"
    cd "${SCRIPT_DIR}"
    python3 dsmil_ml_orchestrator.py --task "${task}" --priority medium
}

# Get agent recommendations
get_recommendations() {
    local task="${1:-Analyze device performance and security}"
    log_info "Getting agent recommendations for: ${task}"
    cd "${SCRIPT_DIR}"
    python3 dsmil_ml_orchestrator.py --recommend "${task}"
}

# Start full orchestrator
start_orchestrator() {
    local mode="${1:-integrated}"
    log_info "Starting DSMIL ML Orchestrator in ${mode} mode..."
    
    # Create startup log
    local log_file="${LOG_DIR}/orchestrator_$(date +%Y%m%d_%H%M%S).log"
    
    cd "${SCRIPT_DIR}"
    python3 dsmil_ml_orchestrator.py --mode "${mode}" 2>&1 | tee "${log_file}"
}

# Interactive menu
show_menu() {
    echo
    echo "=== DSMIL ML Integration Launcher ==="
    echo "1. Check system status"
    echo "2. Start full orchestrator (integrated mode)"
    echo "3. Start monitoring mode only"
    echo "4. Start analysis mode only"
    echo "5. Start coordination mode only"
    echo "6. Submit test task"
    echo "7. Get agent recommendations"
    echo "8. Run system diagnostics"
    echo "9. Exit"
    echo
}

run_diagnostics() {
    log_info "Running system diagnostics..."
    
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "PWD: $(pwd)"
    echo "Python: $(python3 --version)"
    echo
    
    echo "=== DSMIL Directory Structure ==="
    ls -la "${DSMIL_ROOT}/infrastructure/"
    echo
    
    echo "=== Docker Containers ==="
    docker ps | grep postgres || echo "No PostgreSQL container found"
    echo
    
    echo "=== Python Packages ==="
    python3 -c "
try:
    import asyncpg
    print('✓ asyncpg available')
except ImportError:
    print('✗ asyncpg not available')

try:
    import numpy
    print('✓ numpy available')
except ImportError:
    print('✗ numpy not available')
    
try:
    import sklearn
    print('✓ sklearn available (full ML)')
except ImportError:
    print('○ sklearn not available (using fallbacks)')
"
    echo
    
    echo "=== Database Connection Test ==="
    check_database
}

# Main execution
main() {
    case "${1:-menu}" in
        "check")
            check_dependencies
            check_database
            ;;
        "status")
            check_dependencies
            setup_python_env
            show_status
            ;;
        "start")
            check_dependencies
            setup_python_env
            check_database
            start_orchestrator "${2:-integrated}"
            ;;
        "task")
            check_dependencies
            setup_python_env
            submit_test_task "${2:-Monitor DSMIL device thermal conditions}"
            ;;
        "recommend")
            check_dependencies
            setup_python_env
            get_recommendations "${2:-Analyze device performance and security}"
            ;;
        "diagnostics")
            run_diagnostics
            ;;
        "menu"|*)
            # Interactive menu
            while true; do
                show_menu
                read -p "Enter your choice [1-9]: " choice
                case $choice in
                    1)
                        check_dependencies
                        setup_python_env
                        show_status
                        ;;
                    2)
                        check_dependencies
                        setup_python_env
                        check_database
                        start_orchestrator "integrated"
                        ;;
                    3)
                        check_dependencies
                        setup_python_env
                        check_database
                        start_orchestrator "monitoring"
                        ;;
                    4)
                        check_dependencies
                        setup_python_env
                        check_database
                        start_orchestrator "analysis"
                        ;;
                    5)
                        check_dependencies
                        setup_python_env
                        check_database
                        start_orchestrator "coordination"
                        ;;
                    6)
                        check_dependencies
                        setup_python_env
                        read -p "Enter task description: " task_desc
                        submit_test_task "${task_desc}"
                        ;;
                    7)
                        check_dependencies
                        setup_python_env
                        read -p "Enter task for recommendations: " rec_task
                        get_recommendations "${rec_task}"
                        ;;
                    8)
                        run_diagnostics
                        ;;
                    9)
                        log_info "Exiting..."
                        exit 0
                        ;;
                    *)
                        log_warn "Invalid option: $choice"
                        ;;
                esac
                echo
                read -p "Press Enter to continue..."
            done
            ;;
    esac
}

# Usage information
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "DSMIL ML Integration Launcher"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  menu         Show interactive menu (default)"
    echo "  check        Check dependencies and database"
    echo "  status       Show system status"
    echo "  start [mode] Start orchestrator (modes: integrated, monitoring, analysis, coordination)"
    echo "  task [desc]  Submit a test task"
    echo "  recommend    Get agent recommendations"
    echo "  diagnostics  Run system diagnostics"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Interactive menu"
    echo "  $0 status                            # Show status"
    echo "  $0 start integrated                  # Start full system"
    echo "  $0 task 'Monitor thermal sensors'    # Submit task"
    echo "  $0 recommend 'Optimize performance'  # Get recommendations"
    echo ""
    exit 0
fi

# Execute main function
main "$@"