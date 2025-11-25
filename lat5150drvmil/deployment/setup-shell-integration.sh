#!/bin/bash
#
# LAT5150 DRVMIL - Shell Integration Setup
# Adds LAT5150 API helper functions to shell profile
#
# This script adds the LAT5150 API environment to your shell
# so you can query the unified API from any command line.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_SCRIPT="${PROJECT_ROOT}/deployment/lat5150-api-env.sh"

if [ -n "${SUDO_USER:-}" ]; then
    TARGET_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    TARGET_HOME="${HOME}"
fi

BASHRC_PATH="${TARGET_HOME}/.bashrc"
MARKER="# LAT5150 Unified API Integration"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_section() {
    echo -e "${CYAN}[====]${NC} $1"
}

# Check if env script exists
if [ ! -f "$ENV_SCRIPT" ]; then
    echo "Error: Environment script not found: $ENV_SCRIPT"
    exit 1
fi

log_section "LAT5150 Shell Integration Setup"
echo ""

# Check if already installed
if grep -q "$MARKER" "$BASHRC_PATH" 2>/dev/null; then
    log_warn "LAT5150 API integration already installed in $BASHRC_PATH"
    log_info "To reinstall, remove the section and run this script again"
    echo ""
    log_info "Current integration:"
    grep -A 3 "$MARKER" "$BASHRC_PATH"
    echo ""
    exit 0
fi

# Add to .bashrc
log_info "Adding LAT5150 API integration to $BASHRC_PATH..."

cat >> "$BASHRC_PATH" <<EOF

$MARKER
# Automatically loads LAT5150 Unified Tactical API helper functions
# Provides: lat5150_query, lat5150_status, lat5150_atomic_list, etc.
if [ -f "$ENV_SCRIPT" ]; then
    source "$ENV_SCRIPT"
fi
EOF

log_info "✓ Integration added to $BASHRC_PATH"
echo ""

log_info "Helper functions available after sourcing .bashrc:"
echo "  - lat5150_query <query>        # Query API with natural language"
echo "  - lat5150_status               # Show API status"
echo "  - lat5150_atomic_list <tech>   # List atomic tests"
echo "  - lat5150_atomic_search <plat> # Search by platform"
echo "  - lat5150_test                 # Run API health test"
echo ""

log_info "To activate in current shell:"
echo "  source ~/.bashrc"
echo ""

log_info "To activate in new shells:"
echo "  Open a new terminal (functions load automatically)"
echo ""

log_section "Installation Complete"
echo ""

# Offer to source now
read -p "Source .bashrc now to activate functions? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Sourcing .bashrc..."
    source "$BASHRC_PATH"
    log_info "✓ Functions activated!"
    echo ""
    log_info "Try it now: lat5150_status"
fi
