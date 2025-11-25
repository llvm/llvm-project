#!/bin/bash
# Build all DSMIL .deb packages
# Usage: ./build-all-debs.sh [package-name]
#   package-name: dsmil-platform, dell-milspec-tools, tpm2-accel-examples, or 'all' (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if dpkg-deb is installed
if ! command -v dpkg-deb &> /dev/null; then
    print_error "dpkg-deb not found. Install with: sudo apt-get install dpkg-dev"
    exit 1
fi

# Function to build a .deb package
build_deb() {
    local pkg_dir="$1"
    local pkg_name="$2"
    local output_name="$3"

    if [ ! -d "$pkg_dir" ]; then
        print_error "Package directory not found: $pkg_dir"
        return 1
    fi

    print_header "Building $pkg_name"

    # Check DEBIAN/control exists
    if [ ! -f "$pkg_dir/DEBIAN/control" ]; then
        print_error "Missing DEBIAN/control in $pkg_dir"
        return 1
    fi

    # Set proper permissions
    print_info "Setting permissions..."
    chmod -R 755 "$pkg_dir/DEBIAN"
    chmod 644 "$pkg_dir/DEBIAN/control"
    [ -f "$pkg_dir/DEBIAN/postinst" ] && chmod 755 "$pkg_dir/DEBIAN/postinst"
    [ -f "$pkg_dir/DEBIAN/prerm" ] && chmod 755 "$pkg_dir/DEBIAN/prerm"
    [ -f "$pkg_dir/DEBIAN/postrm" ] && chmod 755 "$pkg_dir/DEBIAN/postrm"

    # Set ownership to root (if running as root or with sudo)
    if [ "$EUID" -eq 0 ]; then
        print_info "Setting ownership to root:root..."
        chown -R root:root "$pkg_dir"
    else
        print_info "Not running as root - skipping ownership change"
        print_info "Package will be built with current user ownership"
    fi

    # Build the package
    print_info "Building package: $output_name"
    dpkg-deb --build "$pkg_dir" "$output_name"

    if [ $? -eq 0 ]; then
        print_success "Built: $output_name"
        ls -lh "$output_name"
        return 0
    else
        print_error "Failed to build: $output_name"
        return 1
    fi
}

# Parse arguments
PACKAGE="${1:-all}"

case "$PACKAGE" in
    dsmil-platform|1)
        build_deb "dsmil-platform_8.3.1-1" "DSMIL Platform" "dsmil-platform_8.3.1-1.deb"
        ;;

    dell-milspec-tools|2)
        build_deb "dell-milspec-tools" "Dell MIL-SPEC Tools" "dell-milspec-tools_1.0.0-1_amd64.deb"
        ;;

    tpm2-accel-examples|3)
        build_deb "tpm2-accel-examples_1.0.0-1" "TPM2 Acceleration Examples" "tpm2-accel-examples_1.0.0-1.deb"
        ;;

    all|*)
        print_header "Building ALL packages"

        FAILED=0

        build_deb "dsmil-platform_8.3.1-1" "DSMIL Platform" "dsmil-platform_8.3.1-1.deb" || FAILED=1
        build_deb "dell-milspec-tools" "Dell MIL-SPEC Tools" "dell-milspec-tools_1.0.0-1_amd64.deb" || FAILED=1
        build_deb "tpm2-accel-examples_1.0.0-1" "TPM2 Acceleration Examples" "tpm2-accel-examples_1.0.0-1.deb" || FAILED=1

        if [ $FAILED -eq 0 ]; then
            print_header "Build Complete - ALL packages built successfully"
            print_success "All .deb packages are ready!"
            echo ""
            print_info "To install all packages:"
            echo "  sudo dpkg -i dsmil-platform_8.3.1-1.deb"
            echo "  sudo dpkg -i dell-milspec-tools_1.0.0-1_amd64.deb"
            echo "  sudo dpkg -i tpm2-accel-examples_1.0.0-1.deb"
            echo "  sudo dpkg -i dsmil-complete_8.3.2-1.deb"
            echo ""
            print_info "Or install all dependencies and complete package:"
            echo "  sudo dpkg -i *.deb"
            echo "  sudo apt-get install -f  # Fix any missing dependencies"
        else
            print_error "Some packages failed to build"
            exit 1
        fi
        ;;
esac

print_success "Done!"
