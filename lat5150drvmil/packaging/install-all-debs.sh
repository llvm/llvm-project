#!/bin/bash
# Install all DSMIL .deb packages in correct order
# Usage: sudo ./install-all-debs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ $1${NC}"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "This script must be run as root"
    print_info "Usage: sudo ./install-all-debs.sh"
    exit 1
fi

# Check if packages exist
PACKAGES=(
    "dsmil-platform_8.3.1-1.deb"
    "dell-milspec-tools_1.0.0-1_amd64.deb"
    "tpm2-accel-examples_1.0.0-1.deb"
    "dsmil-complete_8.3.2-1.deb"
)

MISSING=0
for pkg in "${PACKAGES[@]}"; do
    if [ ! -f "$pkg" ]; then
        print_error "Missing: $pkg"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    print_error "Some packages are missing. Build them first:"
    print_info "./build-all-debs.sh"
    exit 1
fi

print_header "Installing DSMIL Packages"

# Install in correct dependency order
print_info "Installing packages in dependency order..."
echo ""

# 1. Platform (no dependencies)
print_info "[1/4] Installing dsmil-platform..."
if dpkg -i dsmil-platform_8.3.1-1.deb; then
    print_success "dsmil-platform installed"
else
    print_error "Failed to install dsmil-platform"
    exit 1
fi
echo ""

# 2. Tools
print_info "[2/4] Installing dell-milspec-tools..."
if dpkg -i dell-milspec-tools_1.0.0-1_amd64.deb; then
    print_success "dell-milspec-tools installed"
else
    print_error "Failed to install dell-milspec-tools"
    exit 1
fi
echo ""

# 3. Examples
print_info "[3/4] Installing tpm2-accel-examples..."
if dpkg -i tpm2-accel-examples_1.0.0-1.deb; then
    print_success "tpm2-accel-examples installed"
else
    print_error "Failed to install tpm2-accel-examples"
    exit 1
fi
echo ""

# 4. Meta-package
print_info "[4/4] Installing dsmil-complete (meta-package)..."
if dpkg -i dsmil-complete_8.3.2-1.deb; then
    print_success "dsmil-complete installed"
else
    print_error "Failed to install dsmil-complete"
    exit 1
fi
echo ""

# Fix any missing dependencies
print_info "Fixing any missing dependencies..."
apt-get install -f -y

print_header "Installation Complete"

# Show installed packages
print_success "All DSMIL packages installed successfully!"
echo ""
print_info "Installed packages:"
dpkg -l | grep -E "dsmil|dell-milspec|tpm2-accel" | awk '{print "  " $2 " " $3}'
echo ""

print_info "Available commands:"
echo "  dsmil-status          - Check DSMIL device status"
echo "  dsmil-test            - Test DSMIL functionality"
echo "  milspec-control       - Control MIL-SPEC features"
echo "  milspec-monitor       - Monitor system health"
echo "  tpm2-accel-status     - Check TPM2 acceleration"
echo "  milspec-emergency-stop - Emergency shutdown"
echo ""

print_info "Documentation:"
echo "  /usr/share/doc/tpm2-accel-examples/"
echo "  /usr/share/dell-milspec/examples/"
echo ""

print_success "Ready to use!"
echo ""

print_info "Verify installation:"
echo "  ./verify-installation.sh"
echo ""
