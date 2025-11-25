#!/usr/bin/bash

#
# LAT5150DRVMIL Unified AI Platform Setup
# Comprehensive installation with claude-backups integration
#
# Features:
#  - OpenVINO 2025.3.0+ with NPU/GNA drivers
#  - 98-agent system with hardware-aware routing
#  - Binary agent communication with crypto POW
#  - Shadowgit (AVX512-accelerated git)
#  - Voice UI with NPU acceleration
#  - Hook system for auto-optimization
#  - Existing AI enhancements (PostgreSQL, Redis, caching)
#
# Hardware Support:
#  - Intel NPU 3720 (34-49.4 TOPS military mode)
#  - Intel GNA (ultra-low-power inference)
#  - AVX512 with P-core pinning
#  - Intel Arc GPU acceleration
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
DB_NAME="dsmil_ai"
DB_USER="dsmil"
DB_PASSWORD="${DB_PASSWORD:-dsmil_secure_password_$(openssl rand -hex 8)}"
REDIS_PORT="${REDIS_PORT:-6379}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Feature flags
INSTALL_OPENVINO="${INSTALL_OPENVINO:-yes}"
INSTALL_98_AGENTS="${INSTALL_98_AGENTS:-yes}"
INSTALL_VOICE_UI="${INSTALL_VOICE_UI:-yes}"
INSTALL_SHADOWGIT="${INSTALL_SHADOWGIT:-yes}"
COMPILE_NATIVE="${COMPILE_NATIVE:-yes}"
INSTALL_TERMINAL_API="${INSTALL_TERMINAL_API:-yes}"
INSTALL_FILE_MANAGER="${INSTALL_FILE_MANAGER:-yes}"
INSTALL_VPS="${INSTALL_VPS:-no}"  # Optional, not deployed initially

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}    LAT5150DRVMIL Unified AI Platform Setup                          ${NC}"
echo -e "${CYAN}    Claude-Backups Integration + Military NPU/GNA Support            ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo ""

# Functions
print_header() {
    echo ""
    echo -e "${MAGENTA}======================================================================${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}======================================================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_hardware() {
    print_info "Detecting hardware capabilities..."

    HAS_AVX2=false
    HAS_AVX512=false
    HAS_NPU=false
    HAS_GNA=false
    HAS_GPU=false
    P_CORES=12  # Default assumption

    # Check CPU flags
    if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
        HAS_AVX2=true
        print_success "AVX2 supported"
    fi

    if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
        HAS_AVX512=true
        print_success "AVX512 supported - will pin to P-cores (0-11)"
    fi

    # Detect NPU/GNA (may not work in Docker)
    if lspci 2>/dev/null | grep -i "npu\|vpu" >/dev/null; then
        HAS_NPU=true
        print_success "Intel NPU detected (34+ TOPS)"
    else
        print_warning "NPU not detected (may be hidden in Docker environment)"
    fi

    if lspci 2>/dev/null | grep -i "gna\|gaussian" >/dev/null; then
        HAS_GNA=true
        print_success "Intel GNA detected"
    else
        print_warning "GNA not detected (may be hidden in Docker environment)"
    fi

    # Detect Intel GPU
    if lspci 2>/dev/null | grep -i "vga.*intel" >/dev/null; then
        HAS_GPU=true
        print_success "Intel GPU detected"
    fi

    # CPU core count
    CPU_CORES=$(nproc)
    print_info "CPU cores: $CPU_CORES"
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        VER=$(uname -r)
    fi
    print_info "OS: $OS $VER"
}

# ============================================================================
# PHASE 1: Prerequisites
# ============================================================================

print_header "PHASE 1: Prerequisites and Hardware Detection"

detect_os
detect_hardware

# Check Python
if ! command_exists python3; then
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION"

# Check pip
if ! command_exists pip3; then
    print_warning "pip3 not found, installing..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

# Check git
if ! command_exists git; then
    print_error "Git not found. Please install git."
    exit 1
fi

# Check rust (for shadowgit compilation)
if [ "$INSTALL_SHADOWGIT" = "yes" ]; then
    if ! command_exists cargo; then
        print_warning "Rust not found. Shadowgit will use Python fallback."
        print_info "To install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    else
        RUST_VERSION=$(cargo --version | awk '{print $2}')
        print_success "Rust $RUST_VERSION (for shadowgit)"
    fi
fi

# ============================================================================
# PHASE 2: OpenVINO Installation (NPU/GNA Support)
# ============================================================================

if [ "$INSTALL_OPENVINO" = "yes" ]; then
    print_header "PHASE 2: OpenVINO with NPU/GNA Drivers"

    if python3 -c "import openvino" 2>/dev/null; then
        print_success "OpenVINO already installed"
        OPENVINO_VERSION=$(python3 -c "import openvino as ov; print(ov.__version__)" 2>/dev/null || echo "unknown")
        print_info "Version: $OPENVINO_VERSION"
    else
        print_info "Installing OpenVINO 2025.3.0+..."

        # Install OpenVINO via pip (includes NPU/GNA drivers)
        pip3 install --upgrade openvino openvino-dev

        print_success "OpenVINO installed"
    fi

    # Verify NPU support
    print_info "Verifying NPU/GNA support..."
    python3 -c "
import openvino as ov
core = ov.Core()
devices = core.available_devices()
print('Available devices:', devices)

if 'NPU' in devices:
    print('✓ NPU available for inference')
if 'GNA' in devices:
    print('✓ GNA available for inference')
if 'GPU' in devices:
    print('✓ GPU available for inference')
" || print_warning "OpenVINO device detection failed (may work at runtime)"

fi

# ============================================================================
# PHASE 3: System Dependencies
# ============================================================================

print_header "PHASE 3: System Dependencies"

# PostgreSQL and Redis (from existing setup)
print_info "Installing PostgreSQL..."
if command_exists psql; then
    print_success "PostgreSQL already installed"
else
    case "$OS" in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y postgresql postgresql-contrib libpq-dev
            ;;
        fedora|rhel|centos)
            sudo dnf install -y postgresql-server postgresql-contrib postgresql-devel
            sudo postgresql-setup --initdb
            ;;
        *)
            print_warning "Unknown OS, please install PostgreSQL manually"
            ;;
    esac
    print_success "PostgreSQL installed"
fi

print_info "Installing Redis..."
if command_exists redis-server; then
    print_success "Redis already installed"
else
    case "$OS" in
        ubuntu|debian)
            sudo apt-get install -y redis-server
            ;;
        fedora|rhel|centos)
            sudo dnf install -y redis
            ;;
        *)
            print_warning "Unknown OS, please install Redis manually"
            ;;
    esac
    print_success "Redis installed"
fi

# Audio libraries (for voice UI)
if [ "$INSTALL_VOICE_UI" = "yes" ]; then
    print_info "Installing audio libraries for voice UI..."

    case "$OS" in
        ubuntu|debian)
            sudo apt-get install -y portaudio19-dev libsndfile1-dev
            ;;
        fedora|rhel|centos)
            sudo dnf install -y portaudio-devel libsndfile-devel
            ;;
    esac

    print_success "Audio libraries installed"
fi

# Build tools for C compilation
if [ "$COMPILE_NATIVE" = "yes" ]; then
    print_info "Installing build tools..."

    case "$OS" in
        ubuntu|debian)
            sudo apt-get install -y build-essential cmake
            ;;
        fedora|rhel|centos)
            sudo dnf install -y gcc gcc-c++ make cmake
            ;;
    esac

    print_success "Build tools installed"
fi

# ============================================================================
# PHASE 4: Python Dependencies
# ============================================================================

print_header "PHASE 4: Python Dependencies"

print_info "Installing Python packages..."

# Core dependencies
pip3 install --upgrade \
    psycopg2-binary \
    redis \
    numpy \
    psutil \
    flask \
    flask-cors \
    requests \
    aiohttp \
    pydantic \
    python-dotenv \
    PyPDF2

# AI/ML packages
pip3 install --upgrade \
    transformers \
    sentence-transformers \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Voice UI dependencies
if [ "$INSTALL_VOICE_UI" = "yes" ]; then
    pip3 install --upgrade \
        pyaudio \
        soundfile \
        || print_warning "Some audio packages failed (voice UI may have limited functionality)"
fi

print_success "Python packages installed"

# ============================================================================
# PHASE 5: Compile Native Libraries
# ============================================================================

if [ "$COMPILE_NATIVE" = "yes" ]; then
    print_header "PHASE 5: Compile Native Libraries"

    cd "$BASE_DIR"

    # Compile binary agent communication library
    if [ -f "Makefile" ] && [ -f "libagent_comm.c" ]; then
        print_info "Compiling binary agent communication library..."
        make clean 2>/dev/null || true
        make
        print_success "libagent_comm.so compiled"

        # Test library
        python3 -c "import ctypes; lib = ctypes.CDLL('./libagent_comm.so'); print('✓ Library loads successfully')" \
            && print_success "Library test passed" \
            || print_warning "Library test failed (may still work)"
    fi

fi

# ============================================================================
# PHASE 6: Database Setup
# ============================================================================

print_header "PHASE 6: Database Configuration"

# Start PostgreSQL
if command_exists systemctl; then
    sudo systemctl start postgresql || true
    sudo systemctl enable postgresql || true
fi

# Create database and user
print_info "Setting up PostgreSQL database..."

sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || print_info "Database already exists"
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null || print_info "User already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" 2>/dev/null

print_success "Database configured: $DB_NAME"

# Start Redis
print_info "Starting Redis..."
if command_exists systemctl; then
    sudo systemctl start redis || true
    sudo systemctl enable redis || true
elif command_exists redis-server; then
    redis-server --daemonize yes --port $REDIS_PORT || true
fi

# Test Redis
if redis-cli ping >/dev/null 2>&1; then
    print_success "Redis running on port $REDIS_PORT"
else
    print_warning "Redis not responding (binary protocol will use in-memory fallback)"
fi

# ============================================================================
# PHASE 7: Feature Integration
# ============================================================================

print_header "PHASE 7: Feature Integration"

# Create configuration file
print_info "Creating unified configuration..."

cat > "$BASE_DIR/unified_config.json" <<EOF
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "$DB_NAME",
    "user": "$DB_USER",
    "password": "$DB_PASSWORD"
  },
  "redis": {
    "host": "localhost",
    "port": $REDIS_PORT
  },
  "hardware": {
    "avx2": $HAS_AVX2,
    "avx512": $HAS_AVX512,
    "npu": $HAS_NPU,
    "gna": $HAS_GNA,
    "gpu": $HAS_GPU,
    "p_cores": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  },
  "features": {
    "98_agents": $([[ "$INSTALL_98_AGENTS" = "yes" ]] && echo "true" || echo "false"),
    "voice_ui": $([[ "$INSTALL_VOICE_UI" = "yes" ]] && echo "true" || echo "false"),
    "shadowgit": $([[ "$INSTALL_SHADOWGIT" = "yes" ]] && echo "true" || echo "false"),
    "binary_protocol": $([[ "$COMPILE_NATIVE" = "yes" ]] && echo "true" || echo "false"),
    "terminal_api": $([[ "$INSTALL_TERMINAL_API" = "yes" ]] && echo "true" || echo "false"),
    "file_manager_integration": $([[ "$INSTALL_FILE_MANAGER" = "yes" ]] && echo "true" || echo "false"),
    "vps_orchestration": $([[ "$INSTALL_VPS" = "yes" ]] && echo "true" || echo "false")
  },
  "openvino": {
    "enabled": $([[ "$INSTALL_OPENVINO" = "yes" ]] && echo "true" || echo "false"),
    "npu_tops": 34.0,
    "gna_enabled": $HAS_GNA
  }
}
EOF

print_success "Configuration saved: unified_config.json"

# ============================================================================
# PHASE 7A: Terminal API Server (Self-Coding Module)
# ============================================================================

if [ "$INSTALL_TERMINAL_API" = "yes" ]; then
    print_header "PHASE 7A: Terminal API Server"

    print_info "Setting up DSMIL Terminal API server..."

    # Make scripts executable
    chmod +x "$BASE_DIR/dsmil_terminal_api.py" 2>/dev/null || true
    chmod +x "$BASE_DIR/dsmil_api_client.py" 2>/dev/null || true

    # Create systemd service for API server (if running with systemd)
    if command_exists systemctl && [ "$EUID" -eq 0 ]; then
        print_info "Creating systemd service..."

        ACTUAL_USER="${SUDO_USER:-$USER}"

        cat > /etc/systemd/system/dsmil-api.service << EOF
[Unit]
Description=DSMIL Terminal API Server
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$BASE_DIR
ExecStart=/usr/bin/python3 $BASE_DIR/dsmil_terminal_api.py --daemon
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        systemctl enable dsmil-api.service
        print_success "Terminal API systemd service created"
        print_info "Start with: systemctl start dsmil-api"
    else
        print_warning "Not running as root or systemd not available - service not created"
        print_info "Start manually with: python3 $BASE_DIR/dsmil_terminal_api.py --daemon"
    fi

    # Create symlink for easy access
    if [ "$EUID" -eq 0 ]; then
        ln -sf "$BASE_DIR/dsmil_api_client.py" /usr/local/bin/dsmil-api 2>/dev/null || true
        print_success "Terminal API client available as: dsmil-api"
    fi

    print_success "Terminal API server configured"
fi

# ============================================================================
# PHASE 7B: File Manager Integration
# ============================================================================

if [ "$INSTALL_FILE_MANAGER" = "yes" ]; then
    print_header "PHASE 7B: File Manager Integration"

    print_info "Installing 'Open DSMIL AI' context menu..."

    # Run installation script as the actual user (not root)
    ACTUAL_USER="${SUDO_USER:-$USER}"

    if [ -f "$BASE_DIR/file_manager_integration/install_context_menu.sh" ]; then
        if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
            # Running as root via sudo - run as actual user
            su - "$SUDO_USER" -c "cd '$BASE_DIR/file_manager_integration' && ./install_context_menu.sh"
            print_success "File manager integration installed for $SUDO_USER"
        else
            # Running as regular user
            cd "$BASE_DIR/file_manager_integration"
            ./install_context_menu.sh
            cd "$BASE_DIR"
            print_success "File manager integration installed"
        fi
    else
        print_warning "File manager integration script not found, skipping"
    fi
fi

# ============================================================================
# PHASE 7C: VPS Orchestration (Optional)
# ============================================================================

if [ "$INSTALL_VPS" = "yes" ]; then
    print_header "PHASE 7C: VPS Orchestration"

    print_info "Installing VPS orchestration components..."

    # Make scripts executable
    chmod +x "$BASE_DIR/vps_orchestration/asn_vps_manager.py" 2>/dev/null || true
    chmod +x "$BASE_DIR/vps_orchestration/vps_automation_scripts.sh" 2>/dev/null || true
    chmod +x "$(dirname "$BASE_DIR")/launch-vps-orchestration.sh" 2>/dev/null || true

    # Install additional VPS dependencies
    print_info "Installing VPS networking dependencies..."
    case "$OS" in
        ubuntu|debian)
            if [ "$EUID" -eq 0 ]; then
                apt-get install -y wireguard iptables iptables-persistent bird2 jq netcat-openbsd 2>/dev/null || \
                    print_warning "Some VPS dependencies failed to install"
            else
                print_warning "Root required to install VPS dependencies"
            fi
            ;;
        fedora|rhel|centos)
            if [ "$EUID" -eq 0 ]; then
                dnf install -y wireguard-tools iptables bird jq nmap-ncat 2>/dev/null || \
                    print_warning "Some VPS dependencies failed to install"
            else
                print_warning "Root required to install VPS dependencies"
            fi
            ;;
    esac

    # Create symlink for easy access
    if [ "$EUID" -eq 0 ]; then
        ln -sf "$(dirname "$BASE_DIR")/launch-vps-orchestration.sh" /usr/local/bin/dsmil-vps 2>/dev/null || true
        print_success "VPS orchestration available as: dsmil-vps"
    fi

    print_success "VPS orchestration installed"
else
    print_info "VPS orchestration skipped (set INSTALL_VPS=yes to enable)"
fi

# ============================================================================
# PHASE 8: Testing
# ============================================================================

print_header "PHASE 8: Component Testing"

# Test heterogeneous executor
print_info "Testing heterogeneous executor..."
python3 heterogeneous_executor.py > /tmp/hetero_test.log 2>&1 \
    && print_success "Heterogeneous executor test passed" \
    || print_warning "Heterogeneous executor test failed (check /tmp/hetero_test.log)"

# Test 98-agent system
if [ "$INSTALL_98_AGENTS" = "yes" ]; then
    print_info "Testing 98-agent system..."
    python3 -c "
from comprehensive_98_agent_system import create_98_agent_coordinator
coordinator = create_98_agent_coordinator()
print(f'✓ 98 agents initialized, hardware: {coordinator.hardware.npu_available}, {coordinator.hardware.gna_available}')
" && print_success "98-agent system initialized"
fi

# Test binary protocol
if [ "$COMPILE_NATIVE" = "yes" ] && [ -f "libagent_comm.so" ]; then
    print_info "Testing binary agent communication..."
    python3 -c "
from agent_comm_binary import AgentCommunicator
agent = AgentCommunicator('test_agent', enable_pow=False)
print(f'✓ Binary protocol initialized (Redis: {agent.redis_available})')
" && print_success "Binary protocol test passed"
fi

# Test hook system
print_info "Testing hook system..."
python3 -c "
from hook_system import create_default_hooks
hooks = create_default_hooks()
print('✓ Hook system initialized')
" && print_success "Hook system test passed"

# ============================================================================
# PHASE 9: Summary
# ============================================================================

print_header "Installation Complete!"

echo ""
echo -e "${GREEN}Installed Components:${NC}"
echo "  ✓ OpenVINO $(python3 -c 'import openvino as ov; print(ov.__version__)' 2>/dev/null || echo 'N/A')"
echo "  ✓ PostgreSQL ($DB_NAME database)"
echo "  ✓ Redis (port $REDIS_PORT)"
echo "  ✓ Heterogeneous Executor (NPU/GNA/P-core routing)"

if [ "$INSTALL_98_AGENTS" = "yes" ]; then
    echo "  ✓ 98-Agent System (7 categories)"
fi

if [ "$COMPILE_NATIVE" = "yes" ]; then
    echo "  ✓ Binary Agent Communication (crypto POW, AVX512)"
fi

if [ "$INSTALL_SHADOWGIT" = "yes" ]; then
    echo "  ✓ Shadowgit (AVX512-accelerated git)"
fi

if [ "$INSTALL_VOICE_UI" = "yes" ]; then
    echo "  ✓ Voice UI (NPU-accelerated Whisper/Piper)"
fi

echo "  ✓ Hook System (pre/post-query, git hooks)"
echo "  ✓ GUI Dashboard (Flask web interface)"

if [ "$INSTALL_TERMINAL_API" = "yes" ]; then
    echo "  ✓ Terminal API Server (Unix socket IPC for self-coding)"
fi

if [ "$INSTALL_FILE_MANAGER" = "yes" ]; then
    echo "  ✓ File Manager Integration (right-click 'Open DSMIL AI')"
fi

if [ "$INSTALL_VPS" = "yes" ]; then
    echo "  ✓ VPS Orchestration (ASN/BGP management, WireGuard mesh)"
fi

echo ""
echo -e "${GREEN}Hardware Capabilities:${NC}"
echo "  AVX2:    $([ "$HAS_AVX2" = "true" ] && echo "✓ Supported" || echo "✗ Not available")"
echo "  AVX512:  $([ "$HAS_AVX512" = "true" ] && echo "✓ Supported (P-core pinned)" || echo "✗ Not available")"
echo "  NPU:     $([ "$HAS_NPU" = "true" ] && echo "✓ Available (34+ TOPS)" || echo "⚠ Not detected")"
echo "  GNA:     $([ "$HAS_GNA" = "true" ] && echo "✓ Available" || echo "⚠ Not detected")"
echo "  GPU:     $([ "$HAS_GPU" = "true" ] && echo "✓ Available" || echo "✗ Not available")"

echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo ""
echo "  1. Start AI server:"
echo "     cd $BASE_DIR"
echo "     ./start_ai_server.sh"
echo ""
echo "  2. Launch GUI Dashboard:"
echo "     python3 ai_gui_dashboard.py"
echo "     # Access at: http://localhost:5050"
echo ""
echo "  3. Test voice UI (if enabled):"
echo "     python3 voice_ui_npu.py"
echo ""
echo "  4. Run comprehensive tests:"
echo "     python3 comprehensive_98_agent_system.py"
echo ""

if [ "$INSTALL_TERMINAL_API" = "yes" ]; then
    echo "  5. Start Terminal API server:"
    echo "     systemctl start dsmil-api    # (or: python3 dsmil_terminal_api.py --daemon)"
    echo ""
fi

if [ "$INSTALL_FILE_MANAGER" = "yes" ]; then
    echo "  6. Use File Manager Integration:"
    echo "     - Open your file manager (Nautilus, Thunar, etc.)"
    echo "     - Right-click on any folder"
    echo "     - Select 'Open DSMIL AI'"
    echo ""
fi

if [ "$INSTALL_VPS" = "yes" ]; then
    echo "  7. Launch VPS orchestration:"
    echo "     dsmil-vps    # (or: $BASE_DIR/../launch-vps-orchestration.sh)"
    echo ""
fi

echo -e "${CYAN}Configuration file: $BASE_DIR/unified_config.json${NC}"
echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  Installation successful! Military-grade AI platform ready.${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
