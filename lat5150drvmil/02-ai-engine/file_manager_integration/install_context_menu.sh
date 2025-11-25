#!/usr/bin/env bash
################################################################################
# Install DSMIL AI Context Menu Integration
################################################################################
# Installs "Open DSMIL AI" option in file manager right-click menus
# Supports: Nautilus, Thunar, Dolphin, Nemo, Caja
#
# Usage:
#   ./install_context_menu.sh [--uninstall]
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
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SCRIPT="$SCRIPT_DIR/open_dsmil_ai.sh"

# Args
UNINSTALL=false
if [[ "${1-}" == "--uninstall" ]]; then
    UNINSTALL=true
elif [[ "${1-}" != "" ]]; then
    log_error "Unknown option: ${1}"
    echo "Usage: $0 [--uninstall]"
    exit 1
fi

# Banner
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              DSMIL AI Context Menu Installation                           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

if $UNINSTALL; then
    log_info "Uninstalling context menu integration..."
else
    log_info "Installing context menu integration..."
fi
echo ""

# Check script
if [ ! -f "$OPEN_SCRIPT" ]; then
    log_error "Open script not found: $OPEN_SCRIPT"
    exit 1
fi

chmod +x "$OPEN_SCRIPT"

################################################################################
# Nautilus (GNOME Files)
################################################################################

install_nautilus() {
    local scripts_dir="$HOME/.local/share/nautilus/scripts"

    if $UNINSTALL; then
        rm -f "$scripts_dir/Open DSMIL AI" || true
        log_success "Nautilus: Removed"
    else
        mkdir -p "$scripts_dir"
        ln -sf "$OPEN_SCRIPT" "$scripts_dir/Open DSMIL AI"
        log_success "Nautilus: Installed"
    fi
}

################################################################################
# Thunar (XFCE)
################################################################################

install_thunar() {
    local sendto_dir="$HOME/.local/share/Thunar/sendto"
    local desktop_file="$sendto_dir/dsmil-ai.desktop"

    if $UNINSTALL; then
        rm -f "$desktop_file" || true
        log_success "Thunar: Removed"
    else
        mkdir -p "$sendto_dir"

        cat > "$desktop_file" << EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Open DSMIL AI
Comment=Open DSMIL AI Coding Assistant in this directory
Exec=$OPEN_SCRIPT %f
Icon=utilities-terminal
Terminal=false
Categories=Utility;
MimeType=inode/directory;
EOF

        log_success "Thunar: Installed"
    fi
}

################################################################################
# Dolphin (KDE) – Plasma 5 (kservices5) & Plasma 6 (kservices6)
################################################################################

install_dolphin() {
    local services_dir5="$HOME/.local/share/kservices5/ServiceMenus"
    local services_dir6="$HOME/.local/share/kservices6/ServiceMenus"

    local installed_any=false

    if $UNINSTALL; then
        rm -f "$services_dir5/dsmil-ai.desktop" 2>/dev/null || true
        rm -f "$services_dir6/dsmil-ai.desktop" 2>/dev/null || true
        log_success "Dolphin: Removed"
        return
    fi

    if [ -d "$services_dir5" ] || command -v dolphin >/dev/null 2>&1; then
        mkdir -p "$services_dir5"
        cat > "$services_dir5/dsmil-ai.desktop" << EOF
[Desktop Entry]
Type=Service
X-KDE-ServiceTypes=KonqPopupMenu/Plugin
MimeType=inode/directory;
Actions=OpenDSMILAI;
X-KDE-Priority=TopLevel

[Desktop Action OpenDSMILAI]
Name=Open DSMIL AI
Icon=utilities-terminal
Exec=$OPEN_SCRIPT %f
EOF
        installed_any=true
    fi

    if [ -d "$services_dir6" ]; then
        mkdir -p "$services_dir6"
        cat > "$services_dir6/dsmil-ai.desktop" << EOF
[Desktop Entry]
Type=Service
X-KDE-ServiceTypes=KonqPopupMenu/Plugin
MimeType=inode/directory;
Actions=OpenDSMILAI;
X-KDE-Priority=TopLevel

[Desktop Action OpenDSMILAI]
Name=Open DSMIL AI
Icon=utilities-terminal
Exec=$OPEN_SCRIPT %f
EOF
        installed_any=true
    fi

    if $installed_any; then
        log_success "Dolphin: Installed"
    else
        log_warning "Dolphin: No suitable ServiceMenus directory found; skipped."
    fi
}

################################################################################
# Nemo (Cinnamon)
################################################################################

install_nemo() {
    local scripts_dir="$HOME/.local/share/nemo/scripts"

    if $UNINSTALL; then
        rm -f "$scripts_dir/Open DSMIL AI" || true
        log_success "Nemo: Removed"
    else
        mkdir -p "$scripts_dir"
        ln -sf "$OPEN_SCRIPT" "$scripts_dir/Open DSMIL AI"
        log_success "Nemo: Installed"
    fi
}

################################################################################
# Caja (MATE)
################################################################################

install_caja() {
    local scripts_dir="$HOME/.local/share/caja/scripts"

    if $UNINSTALL; then
        rm -f "$scripts_dir/Open DSMIL AI" || true
        log_success "Caja: Removed"
    else
        mkdir -p "$scripts_dir"
        ln -sf "$OPEN_SCRIPT" "$scripts_dir/Open DSMIL AI"
        log_success "Caja: Installed"
    fi
}

################################################################################
# Install for all detected file managers
################################################################################

INSTALLED_COUNT=0

# Nautilus
if command -v nautilus >/dev/null 2>&1 || [ -d "$HOME/.local/share/nautilus" ]; then
    install_nautilus
    INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

# Thunar
if command -v thunar >/dev/null 2>&1 || [ -d "$HOME/.config/Thunar" ]; then
    install_thunar
    INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

# Dolphin
if command -v dolphin >/dev/null 2>&1 || [ -d "$HOME/.local/share/kservices5" ] || [ -d "$HOME/.local/share/kservices6" ]; then
    install_dolphin
    INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

# Nemo
if command -v nemo >/dev/null 2>&1 || [ -d "$HOME/.local/share/nemo" ]; then
    install_nemo
    INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

# Caja
if command -v caja >/dev/null 2>&1 || [ -d "$HOME/.local/share/caja" ]; then
    install_caja
    INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

################################################################################
# Summary
################################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

if $UNINSTALL; then
    log_success "Context menu integration uninstalled from $INSTALLED_COUNT file manager(s)"
else
    if [ "$INSTALLED_COUNT" -eq 0 ]; then
        log_warning "No supported file managers detected; nothing was installed."
        echo "You can still run DSMIL AI manually:"
        echo "  - Start server:  python3 dsmil_terminal_api.py --daemon"
        echo "  - Open project:  ./open_dsmil_ai.sh /path/to/project"
    else
        log_success "Context menu integration installed for $INSTALLED_COUNT file manager(s)"
        echo ""
        echo -e "${BOLD}Usage:${NC}"
        echo "  1. Open your file manager (Nautilus, Thunar, Dolphin, etc.)"
        echo "  2. Right-click on any folder"
        echo "  3. Select ${GREEN}'Open DSMIL AI'${NC} from the context menu"
        echo "  4. The DSMIL AI coding assistant will open in a terminal"
        echo ""
        echo -e "${BOLD}Note:${NC} You may need to restart your file manager for changes to take effect."
    fi
fi

echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

exit 0
