#!/bin/bash
################################################################################
# Build Script for Customizable Functions Feature Development
#
# This script configures, builds, and tests the customizable functions feature
# in Clang with optimized settings for faster iteration.
#
# Usage:
#   ./build-customizable-functions.sh [command] [options]
#
# Commands:
#   configure [debug|release|minimal]  - Configure CMake build
#   build [target]                     - Build clang (or specific target)
#   test [pattern]                     - Run tests (optionally filter by pattern)
#   clean                              - Clean build directory
#   rebuild                            - Clean and rebuild
#   help                               - Show this help
#
# Examples:
#   ./build-customizable-functions.sh configure debug
#   ./build-customizable-functions.sh build
#   ./build-customizable-functions.sh test customizable
#   ./build-customizable-functions.sh rebuild
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_DIR="${SCRIPT_DIR}"
BUILD_DIR="${SCRIPT_DIR}/build-custom-functions"
INSTALL_DIR="${BUILD_DIR}/install"

# Build settings
DEFAULT_BUILD_TYPE="Debug"
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_help() {
    cat << EOF
${BLUE}Customizable Functions Build Script${NC}

${GREEN}Usage:${NC}
  ./build-customizable-functions.sh [command] [options]

${GREEN}Commands:${NC}
  ${YELLOW}configure [mode]${NC}      Configure CMake build
                          Modes: debug (default), release, minimal
  ${YELLOW}build [target]${NC}        Build clang or specific target
                          Targets: clang, check-clang, check-clang-sema
  ${YELLOW}test [pattern]${NC}        Run tests, optionally filter by pattern
                          Examples: customizable, parser, sema
  ${YELLOW}clean${NC}                 Clean build directory
  ${YELLOW}rebuild${NC}               Clean and rebuild from scratch
  ${YELLOW}info${NC}                  Show build information
  ${YELLOW}help${NC}                  Show this help

${GREEN}Examples:${NC}
  # Initial setup - configure and build
  ./build-customizable-functions.sh configure debug
  ./build-customizable-functions.sh build

  # Run all customizable functions tests
  ./build-customizable-functions.sh test customizable

  # Run only parser tests
  ./build-customizable-functions.sh test parser

  # Rebuild everything from scratch
  ./build-customizable-functions.sh rebuild

  # Build and run specific test
  ./build-customizable-functions.sh build check-clang

${GREEN}Build Modes:${NC}
  ${YELLOW}debug${NC}     - Debug build with assertions (default)
              - Best for development and debugging
              - Slower but easier to debug

  ${YELLOW}release${NC}   - Optimized release build
              - Faster but harder to debug
              - Use for performance testing

  ${YELLOW}minimal${NC}   - Minimal debug build
              - Only builds clang, not all of LLVM
              - Fastest iteration time

${GREEN}Environment Variables:${NC}
  ${YELLOW}BUILD_JOBS${NC}      Number of parallel build jobs (default: ${NUM_JOBS})
  ${YELLOW}BUILD_TYPE${NC}      Build type override (Debug/Release/RelWithDebInfo)
  ${YELLOW}CC${NC}              C compiler (default: auto-detect)
  ${YELLOW}CXX${NC}             C++ compiler (default: auto-detect)

EOF
}

################################################################################
# CMake Configuration
################################################################################

configure_build() {
    local build_mode="${1:-debug}"

    print_header "Configuring CMake Build (${build_mode} mode)"

    # Create build directory
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # Determine build type
    local cmake_build_type="${DEFAULT_BUILD_TYPE}"
    local extra_flags=""

    case "${build_mode}" in
        debug)
            cmake_build_type="Debug"
            print_info "Debug build: Assertions enabled, optimizations disabled"
            ;;
        release)
            cmake_build_type="Release"
            print_info "Release build: Optimizations enabled, assertions disabled"
            ;;
        minimal)
            cmake_build_type="Debug"
            extra_flags="-DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON"
            print_info "Minimal build: Only X86 target, debug mode"
            ;;
        *)
            print_error "Unknown build mode: ${build_mode}"
            print_info "Valid modes: debug, release, minimal"
            exit 1
            ;;
    esac

    # Allow override via environment
    if [ -n "${BUILD_TYPE}" ]; then
        cmake_build_type="${BUILD_TYPE}"
        print_warning "Build type overridden to: ${cmake_build_type}"
    fi

    print_info "Build directory: ${BUILD_DIR}"
    print_info "Install directory: ${INSTALL_DIR}"
    print_info "Build type: ${cmake_build_type}"
    print_info "Parallel jobs: ${NUM_JOBS}"

    # Configure CMake with optimized settings for development
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE="${cmake_build_type}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DLLVM_ENABLE_PROJECTS="clang" \
        -DLLVM_ENABLE_RUNTIMES="" \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DLLVM_INCLUDE_TESTS=ON \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_WERROR=OFF \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DCLANG_ENABLE_STATIC_ANALYZER=ON \
        -DCLANG_ENABLE_ARCMT=OFF \
        -DCLANG_BUILD_EXAMPLES=OFF \
        -DLLVM_BUILD_LLVM_DYLIB=OFF \
        -DLLVM_LINK_LLVM_DYLIB=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ${extra_flags} \
        "${LLVM_DIR}/llvm"

    print_success "CMake configuration complete"
    print_info "Build files generated in: ${BUILD_DIR}"

    # Create symlink to compile_commands.json for IDE support
    if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
        ln -sf "${BUILD_DIR}/compile_commands.json" "${LLVM_DIR}/compile_commands.json"
        print_success "Created compile_commands.json symlink for IDE support"
    fi
}

################################################################################
# Build Functions
################################################################################

build_target() {
    local target="${1:-clang}"

    if [ ! -d "${BUILD_DIR}" ]; then
        print_error "Build directory not found. Run 'configure' first."
        exit 1
    fi

    print_header "Building ${target}"

    cd "${BUILD_DIR}"

    # Use BUILD_JOBS env var if set
    local jobs="${BUILD_JOBS:-${NUM_JOBS}}"

    print_info "Building with ${jobs} parallel jobs..."

    local start_time=$(date +%s)

    if ninja -j "${jobs}" "${target}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Build complete in ${duration} seconds"

        if [ "${target}" = "clang" ]; then
            print_info "Clang binary: ${BUILD_DIR}/bin/clang"
        fi
    else
        print_error "Build failed"
        exit 1
    fi
}

################################################################################
# Test Functions
################################################################################

run_tests() {
    local pattern="${1:-customizable-functions}"

    if [ ! -d "${BUILD_DIR}" ]; then
        print_error "Build directory not found. Run 'configure' first."
        exit 1
    fi

    cd "${BUILD_DIR}"

    print_header "Running Tests"
    print_info "Filter pattern: ${pattern}"

    # Check if pattern is a specific test category
    case "${pattern}" in
        customizable*|custom*)
            print_info "Running customizable functions tests..."
            ninja check-clang 2>&1 | grep -i "customizable" || true
            ;;
        parser)
            print_info "Running parser tests..."
            ./bin/llvm-lit -v "${LLVM_DIR}/clang/test/Parser" -a 2>&1 | grep -i "customizable" || true
            ;;
        sema*)
            print_info "Running semantic analysis tests..."
            ./bin/llvm-lit -v "${LLVM_DIR}/clang/test/SemaCXX" -a 2>&1 | grep -i "customizable" || true
            ;;
        codegen)
            print_info "Running code generation tests..."
            ./bin/llvm-lit -v "${LLVM_DIR}/clang/test/CodeGenCXX" -a 2>&1 | grep -i "customizable" || true
            ;;
        ast)
            print_info "Running AST tests..."
            ./bin/llvm-lit -v "${LLVM_DIR}/clang/test/AST" -a 2>&1 | grep -i "customizable" || true
            ;;
        all)
            print_info "Running all Clang tests..."
            ninja check-clang
            ;;
        *)
            print_info "Running tests matching: ${pattern}"
            ./bin/llvm-lit -v "${LLVM_DIR}/clang/test" --filter="${pattern}"
            ;;
    esac

    print_success "Test run complete"
}

################################################################################
# Utility Functions
################################################################################

clean_build() {
    print_header "Cleaning Build Directory"

    if [ -d "${BUILD_DIR}" ]; then
        print_warning "Removing ${BUILD_DIR}..."
        rm -rf "${BUILD_DIR}"
        print_success "Build directory cleaned"
    else
        print_info "Build directory does not exist, nothing to clean"
    fi
}

rebuild_all() {
    print_header "Rebuilding from Scratch"

    clean_build
    configure_build "${1:-debug}"
    build_target "clang"

    print_success "Rebuild complete"
}

show_info() {
    print_header "Build Information"

    if [ -d "${BUILD_DIR}" ]; then
        echo -e "${GREEN}Build Directory:${NC} ${BUILD_DIR}"

        if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
            local build_type=$(grep "CMAKE_BUILD_TYPE:" "${BUILD_DIR}/CMakeCache.txt" | cut -d'=' -f2)
            echo -e "${GREEN}Build Type:${NC} ${build_type}"

            local targets=$(grep "LLVM_TARGETS_TO_BUILD:" "${BUILD_DIR}/CMakeCache.txt" | cut -d'=' -f2)
            echo -e "${GREEN}Targets:${NC} ${targets}"
        fi

        if [ -f "${BUILD_DIR}/bin/clang" ]; then
            echo -e "${GREEN}Clang Binary:${NC} ${BUILD_DIR}/bin/clang"
            echo -e "${GREEN}Clang Version:${NC}"
            "${BUILD_DIR}/bin/clang" --version | head -1
        else
            echo -e "${YELLOW}Clang not built yet${NC}"
        fi

        echo ""
        echo -e "${GREEN}Available Test Commands:${NC}"
        echo -e "  ./build-customizable-functions.sh test customizable  ${BLUE}# Run customizable functions tests${NC}"
        echo -e "  ./build-customizable-functions.sh test parser        ${BLUE}# Run parser tests${NC}"
        echo -e "  ./build-customizable-functions.sh test sema          ${BLUE}# Run semantic tests${NC}"
        echo -e "  ./build-customizable-functions.sh test codegen       ${BLUE}# Run codegen tests${NC}"
        echo -e "  ./build-customizable-functions.sh test all           ${BLUE}# Run all tests${NC}"
    else
        print_warning "Build not configured yet"
        echo -e "Run: ${YELLOW}./build-customizable-functions.sh configure${NC}"
    fi
}

################################################################################
# Quick Development Workflow Commands
################################################################################

quick_test() {
    print_header "Quick Test Workflow"
    print_info "Building and testing customizable functions..."

    build_target "clang"
    run_tests "customizable"

    print_success "Quick test complete"
}

################################################################################
# Main Script
################################################################################

main() {
    local command="${1:-help}"
    shift || true

    case "${command}" in
        configure|config)
            configure_build "$@"
            ;;
        build)
            build_target "$@"
            ;;
        test)
            run_tests "$@"
            ;;
        clean)
            clean_build
            ;;
        rebuild)
            rebuild_all "$@"
            ;;
        info)
            show_info
            ;;
        quick)
            quick_test
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: ${command}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
