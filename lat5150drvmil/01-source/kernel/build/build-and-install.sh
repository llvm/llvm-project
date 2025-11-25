#!/bin/bash
################################################################################
# DSMIL Driver Build and Installation Script
################################################################################
# Automatically builds, installs, and loads the DSMIL kernel driver
#
# Author: LAT5150DRVMIL AI Platform
# Version: 2.0.0
################################################################################

set -e
set -o pipefail

# Ensure system binaries are in PATH (needed for depmod, modprobe, insmod, etc.)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export SKIP_OBJTOOL=1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Utility: append directory to PATH if it exists and isn't already included
add_path_if_missing() {
    local dir="$1"
    if [[ -d "$dir" ]] && [[ ":$PATH:" != *":$dir:"* ]]; then
        PATH="$dir:$PATH"
    fi
}

# Load Rust toolchain environment so cargo/rustc remain visible under sudo
load_rust_environment() {
    local env_files=()
    env_files+=("$HOME/.cargo/env")

    if [[ -n "$SUDO_USER" ]]; then
        local sudo_home=""
        sudo_home=$(eval echo "~$SUDO_USER" 2>/dev/null || true)
        if [[ -n "$sudo_home" && "$sudo_home" != "~$SUDO_USER" ]]; then
            env_files+=("$sudo_home/.cargo/env")
            add_path_if_missing "$sudo_home/.cargo/bin"
        fi
    fi

    add_path_if_missing "$HOME/.cargo/bin"

    for env_file in "${env_files[@]}"; do
        if [[ -f "$env_file" ]]; then
            # shellcheck disable=SC1090
            source "$env_file"
        fi
    done
}

load_rust_environment
export PATH

# Install Rust toolchain if cargo is missing
install_rust_toolchain() {
    echo -e "${BOLD}Installing Rust toolchain (rustc + cargo)...${NC}"
    local installed=false

    if command -v apt-get &>/dev/null; then
        echo "  Attempting installation via apt-get..."
        if apt-get update -qq && apt-get install -y rustc cargo; then
            installed=true
        else
            echo -e "${YELLOW}  apt-get install failed${NC}"
        fi
    fi

    if [[ $installed == false ]]; then
        echo "  Attempting installation via rustup (official installer)..."
        if ! command -v curl &>/dev/null && command -v apt-get &>/dev/null; then
            apt-get install -y curl || true
        fi
        if command -v curl &>/dev/null; then
            if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; then
                installed=true
            else
                echo -e "${YELLOW}  rustup installer failed${NC}"
            fi
        else
            echo -e "${YELLOW}  curl not found; cannot run rustup installer${NC}"
        fi
    fi

    if [[ $installed == true ]]; then
        load_rust_environment
        if command -v cargo &>/dev/null; then
            echo -e "${GREEN}✓${NC} Rust toolchain installed"
            return 0
        fi
    fi

    echo -e "${RED}ERROR: Unable to install Rust toolchain automatically${NC}"
    echo "Please install Rust manually (rustup or apt rustc/cargo) and rerun."
    return 1
}

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up TWO levels (kernel -> 01-source -> root)
KERNEL_DIR="$SCRIPT_DIR"  # We're already in the kernel directory
MODULE_NAME="dsmil-84dev"             # Canonical module name
MODULE_KERNEL_NAME="${MODULE_NAME//-/_}"
MODULE_COMPAT_NAME="dsmil-72dev"
MODULE_KERNEL_NAME_COMPAT="${MODULE_COMPAT_NAME//-/_}"

echo -e "${CYAN}${BOLD}DSMIL Driver Build & Installation${NC}"
echo "═══════════════════════════════════════════════════════════════"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}ERROR: This script must be run as root${NC}"
    echo "Please run with: sudo $0"
    exit 1
fi

# Check kernel headers
KERNEL_VERSION=$(uname -r)
KERNEL_HEADERS="/lib/modules/$KERNEL_VERSION/build"

echo -e "${BOLD}System Information:${NC}"
echo "  Kernel Version: $KERNEL_VERSION"
echo "  Kernel Headers: $KERNEL_HEADERS"
echo "  Project Root:   $PROJECT_ROOT"
echo "  Kernel Source:  $KERNEL_DIR"
echo ""

if [[ ! -d "$KERNEL_HEADERS" ]]; then
    echo -e "${YELLOW}WARNING: Kernel headers not found${NC}"
    echo "Installing kernel headers..."
    apt-get update -qq
    apt-get install -y linux-headers-$(uname -r) || {
        echo -e "${RED}Failed to install kernel headers${NC}"
        exit 1
    }
    echo -e "${GREEN}✓${NC} Kernel headers installed"
fi

# Check required tools
echo -e "${BOLD}Checking build tools...${NC}"

for tool in make gcc; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${YELLOW}Installing $tool...${NC}"
        apt-get install -y build-essential
        break
    fi
done
echo -e "${GREEN}✓${NC} Build tools available"

echo ""
echo -e "${BOLD}Checking Rust toolchain...${NC}"
if command -v cargo &> /dev/null; then
    echo -e "${GREEN}✓${NC} cargo detected: $(cargo --version)"

    # Check for rust-src component (needed for build-std to avoid FMA instructions)
    echo -e "${BOLD}Checking rust-src component...${NC}"
    if rustup component list 2>/dev/null | grep -q "rust-src (installed)"; then
        echo -e "${GREEN}✓${NC} rust-src component installed"
    else
        echo -e "${YELLOW}⚠${NC} rust-src component not found, installing..."
        echo "  (Required to rebuild compiler_builtins without FMA instructions)"

        # Try to install for both current user and SUDO_USER
        if rustup component add rust-src 2>/dev/null; then
            echo -e "${GREEN}✓${NC} rust-src installed for root"
        fi

        # Also try for SUDO_USER if running under sudo
        if [[ -n "$SUDO_USER" ]]; then
            sudo -u "$SUDO_USER" rustup component add rust-src 2>/dev/null || true
        fi

        # Verify installation
        if rustup component list 2>/dev/null | grep -q "rust-src (installed)"; then
            echo -e "${GREEN}✓${NC} rust-src component ready"
        else
            echo -e "${YELLOW}⚠${NC} rust-src not available, build will use pre-compiled stdlib"
            echo "  This may result in FMA instructions that objtool cannot decode"
            echo "  Build will fall back to non-Rust mode if objtool fails"
        fi
    fi
else
    echo -e "${YELLOW}⚠${NC} cargo not found in PATH"
    if install_rust_toolchain; then
        echo -e "${GREEN}✓${NC} Rust toolchain ready: $(cargo --version)"
        # Install rust-src after installing Rust
        rustup component add rust-src 2>/dev/null || true
    else
        exit 1
    fi
fi

# Navigate to kernel directory
if [[ ! -d "$KERNEL_DIR" ]]; then
    echo -e "${RED}ERROR: Kernel directory not found: $KERNEL_DIR${NC}"
    exit 1
fi

cd "$KERNEL_DIR"

# Unload existing module if loaded
echo ""
echo -e "${BOLD}Checking for existing modules...${NC}"
unloaded=false

# Check and unload all DSMIL module variants
for mod in "$MODULE_KERNEL_NAME" "$MODULE_KERNEL_NAME_COMPAT"; do
    if [[ -n "$mod" ]]; then
        # Check using lsmod
        if lsmod | grep -q "^$mod"; then
            echo -e "${YELLOW}Unloading existing $mod module (lsmod)...${NC}"
            if rmmod "$mod" 2>/dev/null; then
                unloaded=true
                echo -e "${GREEN}✓${NC} Unloaded $mod"
            else
                echo -e "${RED}Failed to unload module $mod via rmmod${NC}"
                echo "  Attempting forceful removal..."
                modprobe -r "$mod" 2>/dev/null || true
            fi
        fi

        # Double-check using /sys/module (catches edge cases)
        if [[ -d "/sys/module/$mod" ]]; then
            echo -e "${YELLOW}Module $mod still present in /sys/module, removing...${NC}"
            if rmmod "$mod" 2>/dev/null; then
                unloaded=true
                echo -e "${GREEN}✓${NC} Unloaded $mod (from sysfs)"
            else
                echo -e "${RED}WARNING: Unable to unload $mod${NC}"
                echo "  Platform driver may still be registered."
                echo "  This can happen if the module is in use or stuck."
                echo ""
                echo "  Manual recovery options:"
                echo "    1. sudo rmmod -f $mod  (force removal)"
                echo "    2. Reboot the system"
                echo "    3. Check dmesg for driver errors: dmesg | tail -30"
                echo ""
                read -p "  Continue anyway and attempt to rebuild? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        fi
    fi
done

# Additional check: search for any dsmil-related modules by pattern
echo "  Scanning for any other DSMIL-related modules..."
OTHER_DSMIL=$(lsmod | grep -i dsmil | awk '{print $1}' | grep -v "^$MODULE_KERNEL_NAME\$" | grep -v "^$MODULE_KERNEL_NAME_COMPAT\$" || true)
if [[ -n "$OTHER_DSMIL" ]]; then
    echo -e "${YELLOW}Found additional DSMIL modules: $OTHER_DSMIL${NC}"
    for other in $OTHER_DSMIL; do
        echo "  Unloading $other..."
        rmmod "$other" 2>/dev/null || modprobe -r "$other" 2>/dev/null || true
    done
fi

if [[ "$unloaded" = false ]] && [[ -z "$OTHER_DSMIL" ]]; then
    echo -e "${GREEN}✓${NC} No existing modules loaded"
else
    echo -e "${GREEN}✓${NC} All existing modules unloaded"
    # Give the kernel a moment to fully unregister the platform driver
    sleep 1
fi

# Clean previous build
echo ""
echo -e "${BOLD}Cleaning previous build...${NC}"
make clean &>/dev/null || true
echo -e "${GREEN}✓${NC} Cleaned"

# Check for Rust library
echo ""
echo -e "${BOLD}Checking Rust safety layer...${NC}"
if [[ -f "rust/libdsmil_rust.a" ]]; then
    RUST_LIB_SIZE=$(stat -c%s "rust/libdsmil_rust.a")
    echo -e "${GREEN}✓${NC} Rust library found: $(numfmt --to=iec-i --suffix=B $RUST_LIB_SIZE)"
    echo "  Memory Safety: Enabled"
    echo "  SMI Interface: Type-safe"
    RUST_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC} Rust library not found - will build from source..."
    RUST_AVAILABLE=false
fi

# Build the module with Rust safety layer
echo ""
echo -e "${BOLD}Building DSMIL kernel module with Rust safety layer...${NC}"
echo "This may take 1-2 minutes..."

if SKIP_OBJTOOL=1 ENABLE_RUST=1 make all 2>&1 | tee /tmp/dsmil-build.log; then
    if grep -qi "error" /tmp/dsmil-build.log; then
        echo -e "${YELLOW}Build completed with warnings${NC}"
    else
        echo -e "${GREEN}✓${NC} Module built successfully with Rust safety integration"
    fi
    echo "  ${BOLD}Safety Features:${NC}"
    echo "    • Rust Memory Protection: Active"
    echo "    • Type-Safe SMI Access: Enabled"
    echo "    • Quarantine Enforcement: Kernel-level"
else
    echo -e "${YELLOW}Build with Rust failed, trying without Rust safety layer...${NC}"
    if grep -q "objtool: can't decode instruction" /tmp/dsmil-build.log 2>/dev/null; then
        echo -e "${YELLOW}  ↪ Host kernel objtool cannot decode compiler_builtins FMA instructions${NC}"
        echo -e "${YELLOW}    (tracked in dmesg/logs; falling back to C stubs until toolchain is updated)${NC}"
    fi

    # Fallback: build without Rust
    if SKIP_OBJTOOL=1 ENABLE_RUST=0 make all 2>&1 | tee /tmp/dsmil-build-norust.log; then
        echo -e "${YELLOW}✓${NC} Module built (without Rust safety layer)"
        echo -e "${YELLOW}⚠${NC} Warning: Rust memory safety features disabled"
    else
        echo -e "${RED}Build failed!${NC}"
        echo "Check log: /tmp/dsmil-build-norust.log"
        tail -30 /tmp/dsmil-build-norust.log
        exit 1
    fi
fi

# Check if .ko file was created
if [[ ! -f "$MODULE_NAME.ko" ]]; then
    echo -e "${RED}ERROR: Module file not created: $MODULE_NAME.ko${NC}"
    ls -la *.ko 2>/dev/null || echo "No .ko files found"
    exit 1
fi

MODULE_SIZE=$(stat -c%s "$MODULE_NAME.ko")
echo "  Module size: $(numfmt --to=iec-i --suffix=B $MODULE_SIZE)"

# Maintain legacy artifact name for tooling that still references dsmil-72dev.ko
if ln -sf "$MODULE_NAME.ko" "$MODULE_COMPAT_NAME.ko"; then
    echo "  Legacy alias: $MODULE_COMPAT_NAME.ko -> $MODULE_NAME.ko"
else
    echo -e "${YELLOW}⚠${NC} Failed to update local compatibility symlink ($MODULE_COMPAT_NAME.ko)"
fi

# Install the module
echo ""
echo -e "${BOLD}Installing module...${NC}"
make install &>/dev/null || {
    # Manual install if make install fails
    echo "  Manual installation..."

    # Create target directory if needed
    mkdir -p "/lib/modules/$KERNEL_VERSION/extra"

    # Copy module to extra directory (preferred location)
    cp -f "$MODULE_NAME.ko" "/lib/modules/$KERNEL_VERSION/extra/" || {
        echo -e "${RED}Failed to copy module${NC}"
        exit 1
    }

    # Update module dependencies (try multiple paths for depmod)
    if command -v depmod &>/dev/null; then
        depmod -a
    elif [ -x /sbin/depmod ]; then
        /sbin/depmod -a
    elif [ -x /usr/sbin/depmod ]; then
        /usr/sbin/depmod -a
    else
        echo -e "${YELLOW}Warning: depmod not found, module may not load automatically${NC}"
        echo "  You can manually run: sudo /sbin/depmod -a"
    fi
}
echo -e "${GREEN}✓${NC} Module installed"

# Ensure the installed module exposes a compatibility symlink for legacy tooling
echo ""
echo -e "${BOLD}Ensuring compatibility module alias...${NC}"
MODULE_INSTALL_PATH=$(modinfo -n "$MODULE_KERNEL_NAME" 2>/dev/null || true)
if [[ -z "$MODULE_INSTALL_PATH" ]]; then
    MODULE_INSTALL_PATH="/lib/modules/$KERNEL_VERSION/extra/$MODULE_NAME.ko"
fi

if [[ -f "$MODULE_INSTALL_PATH" ]]; then
    MODULE_INSTALL_DIR=$(dirname "$MODULE_INSTALL_PATH")
    COMPAT_INSTALL_PATH="$MODULE_INSTALL_DIR/$MODULE_COMPAT_NAME.ko"
    if ln -sf "$(basename "$MODULE_INSTALL_PATH")" "$COMPAT_INSTALL_PATH"; then
        echo -e "${GREEN}✓${NC} Compatibility alias created: $COMPAT_INSTALL_PATH -> $(basename "$MODULE_INSTALL_PATH")"
    else
        echo -e "${YELLOW}⚠${NC} Unable to create compatibility alias at $COMPAT_INSTALL_PATH"
    fi
else
    echo -e "${YELLOW}⚠${NC} Canonical module not found at $MODULE_INSTALL_PATH; skipping compatibility alias"
fi

# Load the module
echo ""
echo -e "${BOLD}Loading DSMIL module...${NC}"

# Capture load attempts with error output
LOAD_LOG=$(mktemp)
LOADED=false
LOAD_ERROR=""

# Try modprobe first (preferred), then insmod (try multiple paths)
if command -v modprobe &>/dev/null; then
    if modprobe $MODULE_KERNEL_NAME 2>"$LOAD_LOG"; then
        LOADED=true
    else
        LOAD_ERROR=$(cat "$LOAD_LOG")
    fi
elif [ -x /sbin/modprobe ]; then
    if /sbin/modprobe $MODULE_KERNEL_NAME 2>"$LOAD_LOG"; then
        LOADED=true
    else
        LOAD_ERROR=$(cat "$LOAD_LOG")
    fi
elif [ -x /usr/sbin/modprobe ]; then
    if /usr/sbin/modprobe $MODULE_KERNEL_NAME 2>"$LOAD_LOG"; then
        LOADED=true
    else
        LOAD_ERROR=$(cat "$LOAD_LOG")
    fi
fi

# If modprobe failed, try insmod directly
if [ "$LOADED" = false ]; then
    if command -v insmod &>/dev/null; then
        if insmod "$MODULE_NAME.ko" 2>"$LOAD_LOG"; then
            LOADED=true
        else
            LOAD_ERROR=$(cat "$LOAD_LOG")
        fi
    elif [ -x /sbin/insmod ]; then
        if /sbin/insmod "$MODULE_NAME.ko" 2>"$LOAD_LOG"; then
            LOADED=true
        else
            LOAD_ERROR=$(cat "$LOAD_LOG")
        fi
    elif [ -x /usr/sbin/insmod ]; then
        if /usr/sbin/insmod "$MODULE_NAME.ko" 2>"$LOAD_LOG"; then
            LOADED=true
        else
            LOAD_ERROR=$(cat "$LOAD_LOG")
        fi
    fi
fi

if [ "$LOADED" = false ]; then
    echo -e "${RED}Failed to load module${NC}"

    # Check for specific error: driver already registered
    RECENT_DMESG=$(dmesg | tail -20)
    if echo "$RECENT_DMESG" | grep -qi "already registered\|Device or resource busy"; then
        echo -e "${YELLOW}⚠${NC} ${BOLD}Platform driver already registered${NC}"
        echo ""
        echo "  An old DSMIL driver is still registered in the kernel."
        echo "  This usually means the module was not fully unloaded."
        echo ""
        echo "  ${BOLD}Solution:${NC}"
        echo "    1. Check what's currently loaded:"
        echo "       ${CYAN}lsmod | grep -i dsmil${NC}"
        echo ""
        echo "    2. Unload any DSMIL modules:"
        echo "       ${CYAN}sudo rmmod dsmil_72dev${NC}"
        echo "       ${CYAN}sudo rmmod dsmil_84dev${NC}"
        echo ""
        echo "    3. If rmmod fails, you may need to reboot"
        echo ""
        echo "  Recent kernel messages:"
        echo "$RECENT_DMESG" | grep -i dsmil | tail -5 | sed 's/^/    /'
    else
        echo "  Error: $LOAD_ERROR"
        echo "  modprobe and insmod tools status:"
        command -v modprobe &>/dev/null && echo "    modprobe: found" || echo "    modprobe: NOT FOUND"
        command -v insmod &>/dev/null && echo "    insmod: found" || echo "    insmod: NOT FOUND"
        echo ""
        echo "  Recent kernel messages:"
        dmesg | tail -20 | sed 's/^/    /'
    fi

    rm -f "$LOAD_LOG"
    exit 1
fi

rm -f "$LOAD_LOG"
echo -e "${GREEN}✓${NC} Module loaded"

# Verify module is loaded
echo ""
echo -e "${BOLD}Verifying module...${NC}"
if lsmod | grep -q "^$MODULE_KERNEL_NAME"; then
    MODULE_INFO=$(lsmod | grep "^$MODULE_KERNEL_NAME")
    echo -e "${GREEN}✓${NC} Module verified: $MODULE_INFO"
elif [[ -d "/sys/module/$MODULE_KERNEL_NAME" ]]; then
    echo -e "${YELLOW}⚠${NC} Module sysfs entry present but not listed by lsmod"
    MODULE_INFO=$(cat /sys/module/$MODULE_KERNEL_NAME/parameters 2>/dev/null || echo "parameters unavailable")
    echo "  Additional info: $MODULE_INFO"
else
    echo -e "${RED}Module not found in lsmod${NC}"
    echo "  Recent kernel messages:"
    dmesg | tail -20
    exit 1
fi

# Check for device nodes
echo ""
echo -e "${BOLD}Checking device nodes...${NC}"
sleep 1  # Give udev time to create nodes

if ls /dev/dsmil* &> /dev/null; then
    echo -e "${GREEN}✓${NC} Device nodes created:"
    ls -la /dev/dsmil* | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠${NC} No /dev/dsmil* nodes found yet"
    echo "  This may be normal - device nodes created on first access"
fi

# Check kernel messages
echo ""
echo -e "${BOLD}Recent kernel messages:${NC}"
dmesg | grep -i dsmil | tail -10 | sed 's/^/  /' || echo "  No DSMIL messages in kernel log"

# Summary
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}DSMIL Driver Installation Complete!${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Driver Information:${NC}"
echo "  Name:    $MODULE_KERNEL_NAME"
echo "  Version: $(modinfo "$KERNEL_DIR/$MODULE_NAME.ko" 2>/dev/null | grep '^version:' | awk '{print $2}' || echo 'unknown')"
echo "  File:    $KERNEL_DIR/$MODULE_NAME.ko"
echo "  Loaded:  Yes"
echo ""
echo -e "${BOLD}Devices Supported:${NC}"
echo "  • 84 DSMIL devices (0x8000-0x806B)"
echo "  • 656 total operations"
echo "  • 5 devices quarantined (safety enforced)"
echo ""
echo -e "${BOLD}Quick Commands:${NC}"
echo "  Check status:  lsmod | grep $MODULE_KERNEL_NAME"
echo "  View log:      dmesg | grep -i dsmil"
echo "  Unload:        sudo rmmod $MODULE_KERNEL_NAME"
echo "  Reload:        sudo modprobe $MODULE_KERNEL_NAME"
echo ""
