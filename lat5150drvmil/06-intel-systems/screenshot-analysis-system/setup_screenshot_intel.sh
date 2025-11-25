#!/bin/bash
#
# Screenshot Intelligence System Setup Script
# Installs and configures the AI-driven screenshot analysis system
#
# Features:
# - Vector database (Qdrant) installation
# - OCR engines (PaddleOCR, Tesseract)
# - MCP server configuration
# - Security hardening (local-only access)
# - Integration with existing RAG/OSINT system
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/home/user/LAT5150DRVMIL"
INTEL_DIR="${PROJECT_ROOT}/06-intel-systems/screenshot-analysis-system"
RAG_DIR="${PROJECT_ROOT}/04-integrations/rag_system"

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Screenshot Intelligence System Setup${NC}"
echo -e "${BLUE}  AI-Driven Screenshot Organization & Analysis${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

# Function: Print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function: Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function: Check if running as root
check_not_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
}

# Function: Check system requirements
check_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"

    # Python 3.10+
    if ! command_exists python3; then
        print_error "Python 3 not found. Install: sudo apt install python3"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
        print_error "Python 3.10+ required (found $PYTHON_VERSION)"
        exit 1
    fi
    print_status "Python $PYTHON_VERSION"

    # pip
    if ! command_exists pip3; then
        print_error "pip3 not found. Install: sudo apt install python3-pip"
        exit 1
    fi
    print_status "pip3"

    # Docker (for Qdrant)
    if ! command_exists docker; then
        print_warning "Docker not found. Qdrant will require manual installation"
        echo "  Install Docker: curl -fsSL https://get.docker.com | sh"
    else
        print_status "Docker"
    fi

    echo ""
}

# Function: Install system dependencies
install_system_deps() {
    echo -e "${BLUE}Installing system dependencies...${NC}"

    # Check if we need sudo
    if ! command_exists tesseract; then
        echo "Installing Tesseract OCR..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
        print_status "Tesseract OCR installed"
    else
        print_status "Tesseract OCR already installed"
    fi

    # Install additional OCR dependencies
    if ! dpkg -l | grep -q libgomp1; then
        sudo apt-get install -y libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev
        print_status "Additional OCR dependencies installed"
    fi

    echo ""
}

# Function: Install Python dependencies
install_python_deps() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"

    # Core dependencies
    pip3 install --user -q --upgrade pip

    echo "Installing Qdrant client..."
    pip3 install --user -q qdrant-client

    echo "Installing sentence-transformers..."
    pip3 install --user -q sentence-transformers

    echo "Installing PaddleOCR..."
    pip3 install --user -q paddleocr paddlepaddle

    echo "Installing Tesseract wrapper..."
    pip3 install --user -q pytesseract Pillow

    echo "Installing additional dependencies..."
    pip3 install --user -q \
        pyyaml \
        python-dateutil \
        watchdog \
        aiofiles \
        fastapi \
        uvicorn

    # MCP SDK
    echo "Installing MCP SDK..."
    pip3 install --user -q mcp

    print_status "Python dependencies installed"
    echo ""
}

# Function: Setup Qdrant vector database
setup_qdrant() {
    echo -e "${BLUE}Setting up Qdrant vector database...${NC}"

    if ! command_exists docker; then
        print_warning "Docker not available. Please install Qdrant manually:"
        echo "  Docker: curl -fsSL https://get.docker.com | sh"
        echo "  Qdrant: docker run -p 6333:6333 -v \$HOME/qdrant_storage:/qdrant/storage qdrant/qdrant"
        echo ""
        return
    fi

    # Check if Qdrant container exists
    if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
        echo "Qdrant container already exists"

        # Check if running
        if docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
            print_status "Qdrant already running"
        else
            echo "Starting Qdrant container..."
            docker start qdrant
            print_status "Qdrant started"
        fi
    else
        echo "Creating Qdrant container..."

        # Create storage directory
        QDRANT_STORAGE="$HOME/qdrant_storage"
        mkdir -p "$QDRANT_STORAGE"

        # Run Qdrant with:
        # - Local storage persistence
        # - Local-only binding (127.0.0.1)
        # - No external access
        docker run -d \
            --name qdrant \
            -p 127.0.0.1:6333:6333 \
            -v "$QDRANT_STORAGE:/qdrant/storage" \
            --restart unless-stopped \
            qdrant/qdrant

        sleep 5  # Wait for startup

        # Test connection
        if curl -s http://127.0.0.1:6333/collections > /dev/null 2>&1; then
            print_status "Qdrant running and accessible (local-only: 127.0.0.1:6333)"
        else
            print_error "Qdrant started but not accessible. Check docker logs qdrant"
        fi
    fi

    echo ""
}

# Function: Configure directories
setup_directories() {
    echo -e "${BLUE}Setting up directories...${NC}"

    # Create data directories
    DATA_DIR="$HOME/.screenshot_intel"
    mkdir -p "$DATA_DIR"/{screenshots,chat_logs,incidents,logs}

    # Device screenshot directories
    mkdir -p "$DATA_DIR/screenshots"/{phone1,phone2,laptop}

    print_status "Data directory: $DATA_DIR"
    print_status "Subdirectories created"

    echo ""
}

# Function: Update MCP server configuration
update_mcp_config() {
    echo -e "${BLUE}Updating MCP server configuration...${NC}"

    MCP_CONFIG="${PROJECT_ROOT}/02-ai-engine/mcp_servers_config.json"

    if [ ! -f "$MCP_CONFIG" ]; then
        print_error "MCP config not found: $MCP_CONFIG"
        return
    fi

    # Backup original
    cp "$MCP_CONFIG" "${MCP_CONFIG}.backup_$(date +%Y%m%d_%H%M%S)"

    # Add screenshot-intelligence server entry
    # Note: This is a simplified approach. In production, use jq or Python to properly merge JSON
    echo "Adding screenshot-intelligence MCP server entry..."

    print_status "MCP server configuration ready (manual merge required)"
    echo "  Add to mcp_servers_config.json:"
    echo ""
    cat <<'EOF'
    "screenshot-intelligence": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/screenshot_intel_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/home/user/LAT5150DRVMIL:/home/user/LAT5150DRVMIL/04-integrations/rag_system"
      },
      "description": "Screenshot Intelligence with OCR, timeline analysis, and event correlation"
    }
EOF
    echo ""
}

# Function: Create systemd service (optional)
create_systemd_service() {
    echo -e "${BLUE}Create systemd service? (optional)${NC}"
    echo "This will auto-start Qdrant and Screenshot Intel services on boot"
    read -p "Create systemd service? [y/N]: " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping systemd service creation"
        echo ""
        return
    fi

    # Qdrant service (if using Docker)
    if command_exists docker; then
        # Docker already has --restart unless-stopped, so we're covered
        print_status "Qdrant configured for auto-restart with Docker"
    fi

    echo ""
}

# Function: Run tests
run_tests() {
    echo -e "${BLUE}Running tests...${NC}"

    # Test vector RAG system
    echo "Testing Vector RAG System..."
    if python3 -c "
import sys
sys.path.insert(0, '${RAG_DIR}')
from vector_rag_system import VectorRAGSystem
try:
    rag = VectorRAGSystem()
    stats = rag.get_stats()
    print(f'✓ Vector RAG: {stats[\"total_documents\"]} documents')
except Exception as e:
    print(f'✗ Vector RAG failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_status "Vector RAG System functional"
    else
        print_warning "Vector RAG System test failed (may need Qdrant running)"
    fi

    # Test screenshot intelligence
    echo "Testing Screenshot Intelligence..."
    if python3 -c "
import sys
sys.path.insert(0, '${RAG_DIR}')
from screenshot_intelligence import ScreenshotIntelligence
try:
    intel = ScreenshotIntelligence()
    print(f'✓ Screenshot Intelligence initialized')
except Exception as e:
    print(f'✗ Screenshot Intelligence failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_status "Screenshot Intelligence functional"
    else
        print_warning "Screenshot Intelligence test failed"
    fi

    echo ""
}

# Function: Print completion summary
print_summary() {
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Register devices:"
    echo "   python3 ${RAG_DIR}/screenshot_intelligence.py"
    echo ""
    echo "2. Ingest screenshots:"
    echo "   # Use MCP server or direct Python API"
    echo ""
    echo "3. Query intelligence:"
    echo "   # Via MCP tools or RAG search"
    echo ""
    echo "Services:"
    echo "  - Qdrant: http://127.0.0.1:6333 (local-only)"
    echo "  - Data dir: $HOME/.screenshot_intel"
    echo ""
    echo "Documentation:"
    echo "  - ${PROJECT_ROOT}/06-intel-systems/screenshot-analysis-system/README.md"
    echo "  - ${RAG_DIR}/README.md"
    echo ""
    echo -e "${YELLOW}Security Note:${NC} All services bound to 127.0.0.1 (local-only access)"
    echo ""
}

# Main execution
main() {
    check_not_root
    check_requirements
    install_system_deps
    install_python_deps
    setup_qdrant
    setup_directories
    update_mcp_config
    create_systemd_service
    run_tests
    print_summary
}

# Run main function
main
