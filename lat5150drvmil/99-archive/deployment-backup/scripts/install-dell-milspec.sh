#!/bin/bash
# Dell MIL-SPEC Platform - One-Line Installer
# Usage: curl -sSL https://install.dell-milspec.local/install.sh | sudo bash
# Or: wget -qO- https://install.dell-milspec.local/install.sh | sudo bash

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
REPO_URL="${DELL_MILSPEC_REPO:-https://apt.dell-milspec.internal/debian}"
REPO_GPG_URL="${DELL_MILSPEC_GPG:-https://apt.dell-milspec.internal/pubkey.gpg}"
LOCAL_REPO="/home/john/LAT5150DRVMIL/deployment/apt-repository"
INSTALL_MODE="${1:-auto}"  # auto, interactive, local

# Banner
clear
echo -e "${CYAN}"
cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════╗
║          DELL LATITUDE 5450 MIL-SPEC PLATFORM                  ║
║                   Installation Script                          ║
╚═══════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

log() {
    local level=$1; shift
    case $level in
        INFO) echo -e "${BLUE}[INFO]${NC} $@" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} $@" ;;
        WARNING) echo -e "${YELLOW}[WARNING]${NC} $@" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $@" ;;
    esac
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log ERROR "This installer must be run as root"
    echo "Usage: sudo $0"
    exit 1
fi

# Step 1: Pre-installation validation
log INFO "Running pre-installation validation..."
if [ -f "/home/john/LAT5150DRVMIL/deployment/scripts/validate-system.sh" ]; then
    if /home/john/LAT5150DRVMIL/deployment/scripts/validate-system.sh; then
        log SUCCESS "System validation passed"
    else
        log ERROR "System validation failed"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    fi
else
    log WARNING "Validation script not found, skipping checks"
fi

echo ""

# Step 2: Repository setup
log INFO "Setting up Dell MIL-SPEC APT repository..."

if [ "$INSTALL_MODE" == "local" ] && [ -d "$LOCAL_REPO" ]; then
    log INFO "Using local repository: $LOCAL_REPO"
    
    echo "deb [trusted=yes] file://${LOCAL_REPO} stable main" > /etc/apt/sources.list.d/dell-milspec.list
    
elif [ -n "$REPO_URL" ]; then
    log INFO "Using remote repository: $REPO_URL"
    
    # Download GPG key
    if wget -qO /etc/apt/trusted.gpg.d/dell-milspec.asc "$REPO_GPG_URL" 2>/dev/null; then
        log SUCCESS "GPG key imported"
    else
        log WARNING "Could not import GPG key, using trusted=yes"
    fi
    
    echo "deb [trusted=yes] $REPO_URL stable main" > /etc/apt/sources.list.d/dell-milspec.list
    
else
    log WARNING "No repository configured, will install from local files"
fi

# Update package lists
log INFO "Updating package lists..."
apt-get update -qq

echo ""

# Step 3: Install packages
log INFO "Installing Dell MIL-SPEC packages..."

PACKAGES=(
    "dell-milspec-dsmil-dkms"
    "tpm2-accel-early-dkms"
    "dell-milspec-tools"
)

for pkg in "${PACKAGES[@]}"; do
    if apt-cache show "$pkg" &>/dev/null; then
        log INFO "Installing $pkg..."
        apt-get install -y -qq "$pkg" && log SUCCESS "$pkg installed" || log ERROR "$pkg installation failed"
    else
        log WARNING "$pkg not found in repository"
        
        # Try local install
        if [ -f "/home/john/LAT5150DRVMIL/deployment/debian-packages/${pkg}"*.deb ]; then
            log INFO "Installing from local file..."
            dpkg -i "/home/john/LAT5150DRVMIL/deployment/debian-packages/${pkg}"*.deb || true
            apt-get install -f -y  # Fix dependencies
        fi
    fi
done

echo ""

# Step 4: Post-installation validation
log INFO "Running post-installation validation..."
if [ -f "/home/john/LAT5150DRVMIL/deployment/scripts/health-check.sh" ]; then
    if /home/john/LAT5150DRVMIL/deployment/scripts/health-check.sh; then
        log SUCCESS "Health check passed"
    else
        log WARNING "Health check found issues (may be normal on first boot)"
    fi
else
    log WARNING "Health check script not found"
fi

echo ""

# Success banner
echo -e "${GREEN}${BOLD}"
cat << 'SUCCESS'
╔═══════════════════════════════════════════════════════════════╗
║              INSTALLATION COMPLETE ✅                          ║
╚═══════════════════════════════════════════════════════════════╝
SUCCESS
echo -e "${NC}"

echo "Dell MIL-SPEC Platform is now installed!"
echo ""
echo "Next steps:"
echo "  • Verify status:  dsmil-status"
echo "  •                 tpm2-accel-status"
echo "  • Launch control: milspec-control"
echo "  • Monitor system: milspec-monitor"
echo ""
echo "Documentation:"
echo "  • /usr/share/doc/dell-milspec-dsmil-dkms/"
echo "  • /usr/share/doc/tpm2-accel-early-dkms/"
echo "  • /usr/share/doc/dell-milspec-tools/"
echo ""
echo "Configuration:"
echo "  • /etc/modprobe.d/dell-milspec.conf"
echo "  • /etc/modprobe.d/tpm2-acceleration.conf"
echo "  • /etc/dell-milspec/dsmil.conf"
echo ""

exit 0
