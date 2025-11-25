#!/bin/bash
# DSMIL Rust Build Script
# Automates building Rust safety layer for kernel integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUST_TARGET="x86_64-unknown-linux-gnu"
RUST_LIB="libdsmil_rust.a"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup Rust environment
setup_rust() {
    log_info "Setting up Rust environment..."
    
    if ! command_exists rustc; then
        log_error "Rust not found. Installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    # Add required components
    rustup component add rust-src 2>/dev/null || true
    rustup target add "$RUST_TARGET" 2>/dev/null || true
    
    log_success "Rust environment ready"
}

# Verify kernel Rust support
check_kernel_rust() {
    local kdir="${KDIR:-/lib/modules/$(uname -r)/build}"
    
    log_info "Checking kernel Rust support..."
    
    if [ ! -d "$kdir" ]; then
        log_warn "Kernel headers not found at $kdir"
        return 1
    fi
    
    if [ -f "$kdir/.config" ]; then
        if grep -q "CONFIG_RUST=y" "$kdir/.config" 2>/dev/null; then
            log_success "Kernel Rust support detected"
            return 0
        else
            log_warn "Kernel compiled without Rust support (CONFIG_RUST=y missing)"
            return 1
        fi
    else
        log_warn "Kernel config not found, unable to verify Rust support"
        return 1
    fi
}

# Build Rust library
build_rust() {
    log_info "Building DSMIL Rust safety layer..."
    
    cd "$SCRIPT_DIR"
    
    # Clean previous builds
    cargo clean
    
    # Build release version
    if cargo build --release --target="$RUST_TARGET"; then
        # Copy library to current directory
        cp "target/$RUST_TARGET/release/$RUST_LIB" .
        log_success "Rust library built: $RUST_LIB"
        
        # Show library info
        local lib_size=$(stat -c%s "$RUST_LIB")
        local lib_symbols=$(nm "$RUST_LIB" | grep -c ' T ' || echo '0')
        log_info "Library size: ${lib_size} bytes"
        log_info "Exported functions: ${lib_symbols}"
        
        return 0
    else
        log_error "Failed to build Rust library"
        return 1
    fi
}

# Run tests
run_tests() {
    log_info "Running Rust unit tests..."
    
    cd "$SCRIPT_DIR"
    
    if cargo test --features testing; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        return 1
    fi
}

# Check code quality
check_quality() {
    log_info "Checking code quality..."
    
    cd "$SCRIPT_DIR"
    
    # Format check
    if ! cargo fmt -- --check; then
        log_warn "Code formatting issues detected, running formatter..."
        cargo fmt
    fi
    
    # Lint with Clippy
    if command_exists clippy; then
        if cargo clippy --target="$RUST_TARGET" -- -D warnings; then
            log_success "Code quality checks passed"
        else
            log_warn "Clippy warnings detected"
        fi
    else
        log_warn "Clippy not available, skipping lint check"
    fi
}

# Integration with kernel module
integrate_kernel() {
    log_info "Preparing kernel module integration..."
    
    local makefile="$KERNEL_DIR/Makefile"
    local rust_lib_path="rust/$RUST_LIB"
    
    # Check if main Makefile exists
    if [ ! -f "$makefile" ]; then
        log_error "Kernel module Makefile not found: $makefile"
        return 1
    fi
    
    # Backup original Makefile
    if [ ! -f "$makefile.backup" ]; then
        cp "$makefile" "$makefile.backup"
        log_info "Backed up original Makefile"
    fi
    
    # Check if Rust integration already present
    if grep -q "rust/" "$makefile"; then
        log_info "Rust integration already present in Makefile"
    else
        log_info "Adding Rust integration to Makefile..."
        
        # Add Rust library to module objects
        sed -i '/^dsmil-72dev-objs/s/$/ rust\/libdsmil_rust.a/' "$makefile" || {
            log_warn "Could not automatically modify Makefile"
            log_info "Manual integration required:"
            log_info "Add 'rust/libdsmil_rust.a' to dsmil-72dev-objs in $makefile"
        }
    fi
    
    # Create integration rule
    cat >> "$makefile" << 'EOF'

# Rust integration
rust/libdsmil_rust.a:
	$(MAKE) -C rust -f Makefile.rust

clean-rust:
	$(MAKE) -C rust -f Makefile.rust clean

.PHONY: clean-rust
EOF
    
    log_success "Kernel integration prepared"
}

# Show build information
show_info() {
    log_info "DSMIL Rust Build Information"
    echo "=========================================="
    echo "Rust version: $(rustc --version 2>/dev/null || echo 'Not installed')"
    echo "Cargo version: $(cargo --version 2>/dev/null || echo 'Not installed')"
    echo "Target: $RUST_TARGET"
    echo "Library: $RUST_LIB"
    echo "Script directory: $SCRIPT_DIR"
    echo "Kernel directory: $KERNEL_DIR"
    echo "=========================================="
    
    if [ -f "$RUST_LIB" ]; then
        echo "Library size: $(stat -c%s "$RUST_LIB") bytes"
        echo "Library symbols: $(nm "$RUST_LIB" | grep -c ' T ' || echo '0')"
    else
        echo "Library: Not built"
    fi
}

# Clean build artifacts
clean_all() {
    log_info "Cleaning all build artifacts..."
    
    cd "$SCRIPT_DIR"
    cargo clean
    rm -f "$RUST_LIB" "${RUST_LIB}.debug"
    
    log_success "Cleanup complete"
}

# Main function
main() {
    local command="${1:-build}"
    
    case "$command" in
        "setup")
            setup_rust
            ;;
        "check")
            check_kernel_rust
            ;;
        "build")
            setup_rust
            check_quality
            build_rust
            ;;
        "test")
            setup_rust
            run_tests
            ;;
        "integrate")
            build_rust
            integrate_kernel
            ;;
        "info")
            show_info
            ;;
        "clean")
            clean_all
            ;;
        "full")
            setup_rust
            check_kernel_rust || log_warn "Kernel Rust support not detected, continuing anyway..."
            check_quality
            run_tests
            build_rust
            integrate_kernel
            show_info
            log_success "Full build and integration complete!"
            ;;
        "help"|"-h"|"--help")
            echo "DSMIL Rust Build Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  setup     - Setup Rust environment"
            echo "  check     - Check kernel Rust support"
            echo "  build     - Build Rust library (default)"
            echo "  test      - Run unit tests"
            echo "  integrate - Build and integrate with kernel module"
            echo "  info      - Show build information"
            echo "  clean     - Clean build artifacts"
            echo "  full      - Complete build and integration"
            echo "  help      - Show this help"
            echo ""
            echo "Examples:"
            echo "  $0              # Build Rust library"
            echo "  $0 full         # Complete setup and build"
            echo "  $0 integrate    # Build and integrate with kernel"
            ;;
        *)
            log_error "Unknown command: $command"
            log_info "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"