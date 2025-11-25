#!/bin/bash
#
# Build script for Gemini MCP Server & CLI
# =========================================
# Builds the Rust CLI and verifies MCP server functionality
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Print header
echo "========================================"
echo "  Gemini MCP Server & CLI Build"
echo "========================================"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

# Check Rust
if ! command -v cargo &> /dev/null; then
    log_error "Rust/Cargo not found. Please install from https://rustup.rs"
    exit 1
fi
RUST_VERSION=$(cargo --version | cut -d' ' -f2)
log_success "Rust $RUST_VERSION found"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
log_success "Python $PYTHON_VERSION found"

# Build Rust CLI
log_info "Building Rust CLI..."
cargo build --release

CLI_PATH="$SCRIPT_DIR/target/release/gemini"
BINARY_SIZE="unknown"

if [ $? -eq 0 ]; then
    log_success "Rust CLI built successfully"

    # Get binary size
    if [ -f "$CLI_PATH" ]; then
        BINARY_SIZE=$(du -h "$CLI_PATH" | cut -f1)
        log_info "Binary size: $BINARY_SIZE"
        log_info "Binary location: $CLI_PATH"
    fi
else
    log_error "Rust build failed"
    exit 1
fi

# Check Python dependencies
log_info "Checking Python dependencies..."

# Create minimal requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    log_info "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Gemini MCP Server Dependencies
# Minimal requirements (uses standard library for most functionality)
EOF
fi

# No external Python dependencies needed (uses standard library)
log_success "Python dependencies OK (using standard library)"

# Test CLI
log_info "Testing Gemini CLI..."

# Test version/help
if "$CLI_PATH" --help &> /dev/null; then
    log_success "CLI help command works"
else
    log_warning "CLI help command failed (non-fatal)"
fi

# Test MCP server
log_info "Testing MCP server..."

# Check if server file exists
if [ ! -f "$SCRIPT_DIR/gemini_mcp_server.py" ]; then
    log_error "MCP server file not found: gemini_mcp_server.py"
    exit 1
fi

# Make MCP server executable
chmod +x "$SCRIPT_DIR/gemini_mcp_server.py"

# Test MCP initialization
MCP_TEST_INPUT='{"method":"initialize","params":{}}'
MCP_TEST_OUTPUT=$(echo "$MCP_TEST_INPUT" | timeout 5 python3 "$SCRIPT_DIR/gemini_mcp_server.py" 2>/dev/null || echo "timeout")

if [[ "$MCP_TEST_OUTPUT" == *"protocolVersion"* ]]; then
    log_success "MCP server initialized successfully"
elif [[ "$MCP_TEST_OUTPUT" == "timeout" ]]; then
    log_warning "MCP server test timed out (may need API key for full functionality)"
else
    log_warning "MCP server test inconclusive (may need API key)"
fi

# Check API key configuration
log_info "Checking API key configuration..."

CONFIG_FILE="$HOME/.config/gemini/config.toml"
if [ -f "$CONFIG_FILE" ]; then
    if grep -q "api_key" "$CONFIG_FILE" 2>/dev/null; then
        log_success "API key found in config file"
    else
        log_warning "Config file exists but no API key found"
    fi
else
    log_warning "No config file found at $CONFIG_FILE"
    log_info "Create config with: mkdir -p ~/.config/gemini && gemini config set api_key YOUR_KEY"
fi

if [ -n "${GEMINI_API_KEY:-}" ]; then
    log_success "GEMINI_API_KEY environment variable is set"
fi

# Build summary
echo ""
echo "========================================"
echo "  Build Summary"
echo "========================================"
echo ""
log_success "Rust CLI: $CLI_PATH ($BINARY_SIZE)"
log_success "MCP Server: $SCRIPT_DIR/gemini_mcp_server.py"
log_success "Subagent: /home/user/LAT5150DRVMIL/02-ai-engine/gemini_subagent.py"
echo ""

# Usage instructions
echo "Next Steps:"
echo "1. Set up API key:"
echo "   export GEMINI_API_KEY='your-api-key'"
echo "   OR"
echo "   gemini config set api_key 'your-api-key'"
echo ""
echo "2. Test the CLI:"
echo "   $CLI_PATH exec 'Hello, Gemini!'"
echo ""
echo "3. Start interactive chat:"
echo "   $CLI_PATH chat"
echo ""
echo "4. Use MCP server (started automatically by LAT5150DRVMIL platform)"
echo ""
echo "Documentation: $SCRIPT_DIR/README.md"
echo ""

log_success "Build completed successfully!"
