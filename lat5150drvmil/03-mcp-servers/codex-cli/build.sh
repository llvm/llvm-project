#!/usr/bin/env bash
#
# Build script for Codex CLI
# Optimized for Intel Meteor Lake (Core Ultra 7)
#

set -e

echo "======================================"
echo " Codex CLI Build Script"
echo "======================================"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ Cargo not found${NC}"
    echo "  Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
echo -e "${GREEN}✓ Cargo found${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 found${NC}"

# Check Python dependencies
if ! python3 -c "import mcp" 2>/dev/null; then
    echo -e "${YELLOW}⚠ MCP SDK not installed${NC}"
    echo "  Installing: pip install mcp"
    pip install mcp
fi
echo -e "${GREEN}✓ MCP SDK available${NC}"

echo

# Build configuration
BUILD_TYPE="${1:-release}"

if [ "$BUILD_TYPE" = "debug" ]; then
    echo "Building in DEBUG mode..."
    cargo build
    BINARY_PATH="target/debug/codex-cli"
elif [ "$BUILD_TYPE" = "release" ]; then
    echo "Building in RELEASE mode with Meteor Lake optimizations..."

    # Detect if AVX-512 is available
    if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}✓ AVX-512 detected${NC}"
        echo "  Building with AVX-512 support..."

        RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512dq,+avx512bw,+avx512vl" \
        cargo build --release
    else
        echo "  Building with AVX2 support..."
        RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma,+aes" \
        cargo build --release
    fi

    BINARY_PATH="target/release/codex-cli"
else
    echo -e "${RED}Unknown build type: $BUILD_TYPE${NC}"
    echo "Usage: $0 [debug|release]"
    exit 1
fi

echo

# Verify build
if [ -f "$BINARY_PATH" ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
    echo
    echo "Binary: $BINARY_PATH"
    echo "Size: $(du -h $BINARY_PATH | cut -f1)"

    # Make executable
    chmod +x "$BINARY_PATH"

    # Test basic functionality
    echo
    echo "Testing binary..."
    if "$BINARY_PATH" --version &> /dev/null; then
        echo -e "${GREEN}✓ Binary works${NC}"
    else
        echo -e "${YELLOW}⚠ Binary test failed (may need authentication)${NC}"
    fi
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo
echo "======================================"
echo " Build Complete!"
echo "======================================"
echo
echo "Next steps:"
echo "  1. Initialize config:    $BINARY_PATH config init"
echo "  2. Authenticate:         $BINARY_PATH auth login"
echo "  3. Test:                 $BINARY_PATH exec 'Hello world'"
echo "  4. Start MCP server:     python3 codex_mcp_server.py"
echo
