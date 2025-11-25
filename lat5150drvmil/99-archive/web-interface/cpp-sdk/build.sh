#!/bin/bash

# DSMIL C++ SDK Build Script
# High-performance build system for the DSMIL control system SDK

set -e

# Build configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_EXAMPLES=${BUILD_EXAMPLES:-ON}
BUILD_TESTS=${BUILD_TESTS:-ON}
ENABLE_HARDWARE_SECURITY=${ENABLE_HARDWARE_SECURITY:-ON}
ENABLE_LOGGING=${ENABLE_LOGGING:-ON}
ENABLE_METRICS=${ENABLE_METRICS:-ON}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print build configuration
print_config() {
    log_info "DSMIL C++ SDK Build Configuration"
    echo "=================================="
    echo "Build Type: $BUILD_TYPE"
    echo "Build Examples: $BUILD_EXAMPLES"
    echo "Build Tests: $BUILD_TESTS"
    echo "Enable Hardware Security: $ENABLE_HARDWARE_SECURITY"
    echo "Enable Logging: $ENABLE_LOGGING"
    echo "Enable Metrics: $ENABLE_METRICS"
    echo "Install Prefix: $INSTALL_PREFIX"
    echo "Build Directory: $BUILD_DIR"
    echo ""
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check for required tools
    local missing_deps=()
    
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
        missing_deps+=("make or ninja")
    fi
    
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        missing_deps+=("g++ or clang++")
    fi
    
    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    fi
    
    # Check for required libraries
    if ! pkg-config --exists openssl; then
        missing_deps+=("openssl-dev")
    fi
    
    if ! pkg-config --exists libcurl; then
        missing_deps+=("libcurl-dev")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies:"
        log_info "Ubuntu/Debian: sudo apt install cmake build-essential pkg-config libssl-dev libcurl4-openssl-dev"
        log_info "CentOS/RHEL: sudo yum install cmake gcc-c++ pkgconfig openssl-devel libcurl-devel"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Check optional dependencies
check_optional_dependencies() {
    log_info "Checking optional dependencies..."
    
    # Check for Boost
    if pkg-config --exists boost; then
        log_success "Boost found - enhanced async operations enabled"
    else
        log_warning "Boost not found - using basic async operations"
    fi
    
    # Check for io_uring (Linux only)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /usr/include/liburing.h ] || [ -f /usr/local/include/liburing.h ]; then
            log_success "liburing found - high-performance I/O enabled"
        else
            log_warning "liburing not found - using standard I/O (install liburing-dev for better performance)"
        fi
    fi
    
    # Check for GoogleTest (for testing)
    if [ "$BUILD_TESTS" = "ON" ]; then
        if pkg-config --exists gtest; then
            log_success "GoogleTest found - using system installation"
        else
            log_info "GoogleTest not found - will download and build"
        fi
    fi
}

# Setup build environment
setup_build_environment() {
    log_info "Setting up build environment..."
    
    # Create build directory
    if [ -d "$BUILD_DIR" ]; then
        log_warning "Build directory exists - cleaning..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    log_success "Build environment ready"
}

# Configure build with CMake
configure_build() {
    log_info "Configuring build with CMake..."
    
    # Determine generator
    local generator=""
    if command -v ninja &> /dev/null; then
        generator="-G Ninja"
        log_info "Using Ninja generator"
    else
        log_info "Using Unix Makefiles generator"
    fi
    
    # Configure CMake
    cmake $generator \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DDSMIL_BUILD_EXAMPLES="$BUILD_EXAMPLES" \
        -DDSMIL_BUILD_TESTS="$BUILD_TESTS" \
        -DDSMIL_ENABLE_LOGGING="$ENABLE_LOGGING" \
        -DDSMIL_ENABLE_METRICS="$ENABLE_METRICS" \
        -DDSMIL_ENABLE_HARDWARE_SECURITY="$ENABLE_HARDWARE_SECURITY" \
        "$SCRIPT_DIR"
    
    log_success "CMake configuration complete"
}

# Build the project
build_project() {
    log_info "Building DSMIL C++ SDK..."
    
    # Determine number of cores for parallel build
    local cores
    if command -v nproc &> /dev/null; then
        cores=$(nproc)
    elif [ -f /proc/cpuinfo ]; then
        cores=$(grep -c ^processor /proc/cpuinfo)
    else
        cores=4  # Default fallback
    fi
    
    log_info "Using $cores parallel jobs"
    
    # Build
    if command -v ninja &> /dev/null && [ -f build.ninja ]; then
        ninja
    else
        make -j"$cores"
    fi
    
    log_success "Build complete"
}

# Run tests
run_tests() {
    if [ "$BUILD_TESTS" != "ON" ]; then
        log_info "Tests disabled - skipping"
        return 0
    fi
    
    log_info "Running tests..."
    
    if ! ctest --output-on-failure -j4; then
        log_error "Some tests failed!"
        return 1
    fi
    
    log_success "All tests passed"
}

# Install the SDK
install_sdk() {
    log_info "Installing DSMIL C++ SDK to $INSTALL_PREFIX..."
    
    if [ "$EUID" -eq 0 ]; then
        # Running as root
        make install
    else
        # Check if we need sudo
        if [ ! -w "$INSTALL_PREFIX" ]; then
            log_info "Installing with sudo (need write access to $INSTALL_PREFIX)..."
            sudo make install
        else
            make install
        fi
    fi
    
    # Update library cache if installing system-wide
    if [[ "$INSTALL_PREFIX" == "/usr"* ]]; then
        if command -v ldconfig &> /dev/null; then
            log_info "Updating library cache..."
            if [ "$EUID" -eq 0 ]; then
                ldconfig
            else
                sudo ldconfig
            fi
        fi
    fi
    
    log_success "Installation complete"
}

# Generate package
generate_package() {
    if ! command -v cpack &> /dev/null; then
        log_warning "CPack not available - skipping package generation"
        return 0
    fi
    
    log_info "Generating packages..."
    
    cpack
    
    if [ $? -eq 0 ]; then
        log_success "Packages generated successfully"
        ls -la *.deb *.rpm *.tar.gz 2>/dev/null || true
    else
        log_warning "Package generation failed (non-critical)"
    fi
}

# Print build summary
print_summary() {
    log_success "Build Summary"
    echo "============="
    echo "Build Type: $BUILD_TYPE"
    echo "Install Prefix: $INSTALL_PREFIX"
    
    if [ -f "$BUILD_DIR/libdsmil_client.so" ] || [ -f "$BUILD_DIR/src/libdsmil_client.so" ]; then
        echo "Shared Library: ✓"
    fi
    
    if [ -f "$BUILD_DIR/libdsmil_client.a" ] || [ -f "$BUILD_DIR/src/libdsmil_client.a" ]; then
        echo "Static Library: ✓"
    fi
    
    if [ "$BUILD_EXAMPLES" = "ON" ]; then
        echo "Examples: ✓"
        echo "Run examples:"
        echo "  ./examples/simple_monitoring"
        echo "  ./examples/performance_benchmark"
        echo "  ./examples/kernel_module_demo"
    fi
    
    if [ "$BUILD_TESTS" = "ON" ]; then
        echo "Tests: ✓"
        echo "Run tests: ctest"
    fi
    
    echo ""
    echo "Usage:"
    echo "  Add to CMakeLists.txt:"
    echo "    find_package(DSMILClient 2.0 REQUIRED)"
    echo "    target_link_libraries(your_app DSMILClient::DSMILClient)"
    echo ""
    echo "  Compile manually:"
    echo "    g++ -std=c++17 main.cpp -ldsmil_client -lcurl -lssl -lcrypto -pthread"
    echo ""
}

# Main execution
main() {
    print_config
    check_dependencies
    check_optional_dependencies
    setup_build_environment
    configure_build
    build_project
    
    if ! run_tests; then
        log_error "Build completed but tests failed"
        exit 1
    fi
    
    # Ask user if they want to install
    if [ "${INSTALL_SDK:-}" = "1" ] || [ "${CI:-}" = "true" ]; then
        install_sdk
    else
        echo ""
        read -p "Install SDK to $INSTALL_PREFIX? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_sdk
        else
            log_info "Skipping installation"
        fi
    fi
    
    # Generate packages if requested
    if [ "${GENERATE_PACKAGES:-}" = "1" ]; then
        generate_package
    fi
    
    print_summary
    log_success "DSMIL C++ SDK build complete!"
}

# Handle command line arguments
case "${1:-}" in
    "clean")
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        log_success "Clean complete"
        exit 0
        ;;
    "install")
        INSTALL_SDK=1
        ;;
    "package")
        GENERATE_PACKAGES=1
        ;;
    "help"|"-h"|"--help")
        echo "DSMIL C++ SDK Build Script"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  (none)    Build the SDK"
        echo "  clean     Clean build directory"
        echo "  install   Build and install"
        echo "  package   Build and generate packages"
        echo "  help      Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  BUILD_TYPE=Release|Debug|RelWithDebInfo|MinSizeRel"
        echo "  BUILD_EXAMPLES=ON|OFF"
        echo "  BUILD_TESTS=ON|OFF"
        echo "  ENABLE_HARDWARE_SECURITY=ON|OFF"
        echo "  ENABLE_LOGGING=ON|OFF"
        echo "  ENABLE_METRICS=ON|OFF"
        echo "  INSTALL_PREFIX=/path/to/install"
        echo ""
        echo "Examples:"
        echo "  $0                              # Basic build"
        echo "  $0 install                      # Build and install"
        echo "  BUILD_TYPE=Debug $0             # Debug build"
        echo "  INSTALL_PREFIX=~/dsmil $0       # Custom install location"
        exit 0
        ;;
esac

# Run main function
main "$@"