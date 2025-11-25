#!/usr/bin/env bash
################################################################################
# Open DSMIL AI - File Manager Integration Script
################################################################################
# Universal script for opening DSMIL AI coding assistant in any directory
# Supports Nautilus, Thunar, Dolphin, Nemo, Caja
#
# Installation:
#   ./install_context_menu.sh
#
# Usage:
#   Called automatically from file manager right-click menu, or:
#     ./open_dsmil_ai.sh /path/to/project
#
# Author: LAT5150DRVMIL AI Platform
# Version: 1.1.0
################################################################################

set -o errexit
set -o nounset
set -o pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Python paths
export PYTHONPATH="$PROJECT_ROOT/02-ai-engine:$PROJECT_ROOT:${PYTHONPATH-}"

# API client/server
API_CLIENT="$PROJECT_ROOT/02-ai-engine/dsmil_api_client.py"
API_SERVER="$PROJECT_ROOT/02-ai-engine/dsmil_terminal_api.py"

# Socket path (allow override for advanced users)
SOCKET_PATH="${DSMIL_SOCKET_PATH:-/tmp/dsmil-ai-$(id -u).sock}"

###############################################################################
# Dependency checks
###############################################################################

ensure_dependencies() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "python3 not found. Install Python 3 and re-run."
        exit 1
    fi

    if [ ! -f "$API_CLIENT" ]; then
        log_error "DSMIL API client not found at: $API_CLIENT"
        log_error "Make sure you're running this from a valid LAT5150DRVMIL checkout."
        exit 1
    fi

    if [ ! -f "$API_SERVER" ]; then
        log_error "DSMIL API server not found at: $API_SERVER"
        exit 1
    fi

    if [ ! -w /tmp ]; then
        log_error "/tmp is not writable; cannot create socket or session scripts."
        exit 1
    fi
}

###############################################################################
# Directory detection (Nautilus / Nemo / Caja / Thunar / Dolphin / CLI)
###############################################################################

detect_directory() {
    local dir=""

    # Nautilus
    if [ -n "${NAUTILUS_SCRIPT_SELECTED_FILE_PATHS-}" ]; then
        # May contain multiple lines; use first
        dir="$(printf "%s" "$NAUTILUS_SCRIPT_SELECTED_FILE_PATHS" | head -n1)"
    # Nemo
    elif [ -n "${NEMO_SCRIPT_SELECTED_FILE_PATHS-}" ]; then
        dir="$(printf "%s" "$NEMO_SCRIPT_SELECTED_FILE_PATHS" | head -n1)"
    # Caja
    elif [ -n "${CAJA_SCRIPT_SELECTED_FILE_PATHS-}" ]; then
        dir="$(printf "%s" "$CAJA_SCRIPT_SELECTED_FILE_PATHS" | head -n1)"
    # Thunar / Dolphin / CLI
    elif [ -n "${1-}" ]; then
        dir="$1"
    else
        dir="$(pwd)"
    fi

    # If it's a file, use its directory
    if [ -f "$dir" ]; then
        dir="$(dirname "$dir")"
    fi

    # Hard fallback if path is bogus
    if [ ! -d "$dir" ]; then
        log_warn "Resolved directory '$dir' is not valid; falling back to current directory."
        dir="$(pwd)"
    fi

    printf "%s" "$dir"
}

###############################################################################
# Server checks
###############################################################################

check_server() {
    if [ -S "$SOCKET_PATH" ]; then
        if python3 "$API_CLIENT" ping >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

start_server() {
    log_info "Starting DSMIL API server..."

    if [ -n "${DSMIL_SOCKET_PATH-}" ]; then
        # Advanced override: explicitly set socket path
        python3 "$API_SERVER" --daemon --socket "$SOCKET_PATH" >/dev/null 2>&1 &
    else
        python3 "$API_SERVER" --daemon >/dev/null 2>&1 &
    fi
    local server_pid=$!

    # Wait for server to come up
    for _ in {1..20}; do
        sleep 0.3
        if check_server; then
            log_ok "API server is up (PID $server_pid, socket $SOCKET_PATH)"
            return 0
        fi
    done

    log_error "Failed to start DSMIL API server. Check /tmp/dsmil-terminal-api.log (if configured) for details."
    return 1
}

###############################################################################
# Terminal detection + session launcher
###############################################################################

detect_terminal() {
    # Allow explicit override
    if [ -n "${DSMIL_TERMINAL-}" ] && command -v "$DSMIL_TERMINAL" >/dev/null 2>&1; then
        echo "$DSMIL_TERMINAL"
        return 0
    fi

    # Preference order: konsole / kitty / alacritty / gnome-terminal / xfce4-terminal / xterm
    if command -v konsole >/dev/null 2>&1; then
        echo "konsole"
    elif command -v kitty >/dev/null 2>&1; then
        echo "kitty"
    elif command -v alacritty >/dev/null 2>&1; then
        echo "alacritty"
    elif command -v gnome-terminal >/dev/null 2>&1; then
        echo "gnome-terminal"
    elif command -v xfce4-terminal >/dev/null 2>&1; then
        echo "xfce4-terminal"
    elif command -v xterm >/dev/null 2>&1; then
        echo "xterm"
    else
        echo ""
        return 1
    fi
}

open_dsmil_terminal() {
    local directory="$1"

    local terminal
    terminal="$(detect_terminal || true)"

    # Create session startup script
    local session_script
    session_script="$(mktemp /tmp/dsmil_session_XXXXXX.sh)"

    cat > "$session_script" << 'SCRIPT_EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

DIRECTORY="__DIRECTORY__"
PROJECT_ROOT="__PROJECT_ROOT__"
API_CLIENT="__API_CLIENT__"

export PYTHONPATH="$PROJECT_ROOT/02-ai-engine:$PROJECT_ROOT:${PYTHONPATH-}"

cleanup() {
    if [ -n "${SESSION_ID-}" ]; then
        python3 "$API_CLIENT" close "$SESSION_ID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# Banner
clear || true
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                      DSMIL AI CODING ASSISTANT                            ║
║                                                                           ║
║              Dell System Military Integration Layer (DSMIL)               ║
║                  LAT5150 MIL-SPEC AI Platform v2.0                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BOLD}Working Directory:${NC} ${CYAN}$DIRECTORY${NC}"
echo ""

if [ ! -d "$DIRECTORY" ]; then
    echo -e "${YELLOW}Warning:${NC} Directory '$DIRECTORY' does not exist. Using \$HOME instead."
    DIRECTORY="$HOME"
fi

# Create session
echo -e "${CYAN}Initializing session...${NC}"
SESSION_INFO="$(python3 "$API_CLIENT" create "$DIRECTORY" 2>&1)" || {
    echo -e "${RED}Failed to create session:${NC}"
    echo "$SESSION_INFO"
    echo ""
    read -rp "Press ENTER to exit..." _
    exit 1
}

SESSION_ID="$(echo "$SESSION_INFO" | awk '/Session created:/ {print $3; exit}')"

if [ -z "$SESSION_ID" ]; then
    echo -e "${RED}Failed to extract session ID from response:${NC}"
    echo "$SESSION_INFO"
    echo ""
    read -rp "Press ENTER to exit..." _
    exit 1
fi

echo -e "${GREEN}✓ Session created: ${SESSION_ID}${NC}"
echo ""

# Analyze project
echo -e "${CYAN}Analyzing project structure...${NC}"
python3 "$API_CLIENT" analyze "$SESSION_ID" 2>/dev/null | python3 -m json.tool 2>/dev/null || {
    echo -e "${YELLOW}Project analysis failed or returned non-JSON; continuing anyway.${NC}"
}
echo ""

# Interactive loop
echo -e "${BOLD}${GREEN}Ready!${NC} Type your questions or commands below."
echo -e "Type ${YELLOW}help${NC} for available commands, ${YELLOW}exit${NC} to quit."
echo ""

while true; do
    echo -ne "${BOLD}${BLUE}DSMIL>${NC} "
    if ! read -r INPUT; then
        # EOF (Ctrl+D)
        echo ""
        break
    fi

    case "$INPUT" in
        "" )
            continue
            ;;
        "exit"|"quit"|"q")
            echo -e "${CYAN}Closing session...${NC}"
            break
            ;;
        "help"|"?")
            echo ""
            echo -e "${BOLD}Available Commands:${NC}"
            echo "  ${YELLOW}help${NC}                  - Show this help"
            echo "  ${YELLOW}analyze${NC}               - Re-analyze project"
            echo "  ${YELLOW}history${NC}               - Show conversation history"
            echo "  ${YELLOW}generate <lang>${NC}       - Generate code in specified language"
            echo "  ${YELLOW}review <file>${NC}         - Review code file"
            echo "  ${YELLOW}exit${NC}                  - Exit assistant"
            echo ""
            echo -e "${BOLD}Example queries:${NC}"
            echo "  - Generate a Python function that reads a JSON file"
            echo "  - How do I implement async error handling in Rust?"
            echo "  - Explain this code pattern: context managers"
            echo "  - Refactor my code to use type hints"
            echo ""
            continue
            ;;
        "analyze")
            python3 "$API_CLIENT" analyze "$SESSION_ID" 2>/dev/null | python3 -m json.tool 2>/dev/null || {
                echo -e "${YELLOW}Analysis failed; showing raw output.${NC}"
                python3 "$API_CLIENT" analyze "$SESSION_ID" 2>/dev/null || true
            }
            echo ""
            continue
            ;;
        "history")
            python3 "$API_CLIENT" history "$SESSION_ID" 2>/dev/null || echo "No history yet"
            echo ""
            continue
            ;;
    esac

    # General query
    echo ""
    RESPONSE="$(python3 "$API_CLIENT" query "$SESSION_ID" "$INPUT" 2>&1)" || {
        echo -e "${RED}Error while querying DSMIL AI:${NC}"
        echo "$RESPONSE"
        echo ""
        continue
    }

    echo -e "${CYAN}$RESPONSE${NC}"
    echo ""
done

echo -e "${GREEN}Goodbye!${NC}"
SCRIPT_EOF

    # Replace placeholders
    sed -i "s|__DIRECTORY__|$directory|g" "$session_script"
    sed -i "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" "$session_script"
    sed -i "s|__API_CLIENT__|$API_CLIENT|g" "$session_script"

    chmod +x "$session_script"

    # Launch terminal or fallback to current TTY
    if [ -n "$terminal" ]; then
        case "$terminal" in
            gnome-terminal)
                (cd "$directory" && gnome-terminal -- "$session_script") || {
                    log_warn "Failed to launch gnome-terminal; running in current terminal instead."
                    (cd "$directory" && "$session_script")
                }
                ;;
            xfce4-terminal)
                (cd "$directory" && xfce4-terminal -e "$session_script") || {
                    log_warn "Failed to launch xfce4-terminal; running in current terminal instead."
                    (cd "$directory" && "$session_script")
                }
                ;;
            konsole)
                (cd "$directory" && konsole -e "$session_script") || {
                    log_warn "Failed to launch konsole; running in current terminal instead."
                    (cd "$directory" && "$session_script")
                }
                ;;
            kitty)
                (cd "$directory" && kitty "$session_script") || {
                    log_warn "Failed to launch kitty; running in current terminal instead."
                    (cd "$directory" && "$session_script")
                }
                ;;
            alacritty)
                (cd "$directory" && alacritty -e "$session_script") || {
                    log_warn "Failed to launch alacritty; running in current terminal instead."
                    (cd "$directory" && "$session_script")
                }
                ;;
            xterm)
                (cd "$directory" && xterm -e "$session_script") || {
                    log_warn "Failed to launch xterm; running in current terminal instead."
                    (cd "$directory" && "$session_script")
                }
                ;;
            *)
                log_warn "Unknown terminal '$terminal'; running in current terminal."
                (cd "$directory" && "$session_script")
                ;;
        esac
    else
        log_warn "No supported terminal emulator found. Running DSMIL session in current terminal."
        (cd "$directory" && "$session_script")
    fi

    # Best-effort cleanup after some time
    (sleep 900; rm -f "$session_script") >/dev/null 2>&1 &
}

###############################################################################
# Main
###############################################################################

main() {
    ensure_dependencies

    DIRECTORY="$(detect_directory "${1-}")"

    echo -e "${CYAN}${BOLD}DSMIL AI - Opening in: ${NC}${DIRECTORY}"
    echo ""

    if ! check_server; then
        if ! start_server; then
            exit 1
        fi
    fi

    open_dsmil_terminal "$DIRECTORY"
}

main "$@"
exit 0
