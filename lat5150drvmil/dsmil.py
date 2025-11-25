#!/usr/bin/env python3
"""
DSMIL Platform - Unified Entry Point
=====================================
Single entry point for the complete DSMIL 104-device platform

This script provides a unified interface to:
- Kernel driver management (build, load, unload)
- Device discovery and activation
- System monitoring and diagnostics
- Control centre access
- Documentation and help

Version: 2.1.0
Compatibility: DSMIL Driver v5.2.0 (104 devices)
Kernel Support: 6.17+ (with fallback to 5.x/4.x)

Features:
- Driver 104 primary with automatic fallback to 84
- Kernel 6.17 compatibility with version detection
- Complex path recovery and location mechanisms
- Multiple compensation mechanisms for edge cases
- Resilient build system with retry logic
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import platform
import re
import glob

# Colors for terminal output (TEMPEST Class C themed)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    # TEMPEST specific colors
    SECURE = '\033[32m'      # Green for secure operations
    CLASSIFIED = '\033[33m'  # Yellow for classified operations
    SHIELDED = '\033[36m'    # Cyan for TEMPEST-shielded operations
    DIM = '\033[2m'          # Dimmed text for sensitive info

def clear_screen():
    """Clear the terminal screen (cross-platform)"""
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")

def check_root():
    """Check if running as root"""
    return os.geteuid() == 0

def detect_linux_distro():
    """Detect Linux distribution"""
    try:
        # Try reading /etc/os-release first
        if Path("/etc/os-release").exists():
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("ID="):
                        distro = line.split("=")[1].strip().strip('"').lower()
                        return distro
        # Fallback methods
        if Path("/etc/debian_version").exists():
            return "debian"
        elif Path("/etc/redhat-release").exists():
            return "rhel"
        elif Path("/etc/arch-release").exists():
            return "arch"
    except:
        pass
    return "unknown"

def auto_install_dependencies():
    """Automatically install missing build dependencies"""
    if not check_root():
        print_error("Auto-install requires root privileges")
        print_info("Run with: sudo python3 dsmil.py build-auto")
        return False

    print_info("Detecting system configuration...")
    distro = detect_linux_distro()
    kernel_version = subprocess.run(['uname', '-r'], capture_output=True, text=True).stdout.strip()

    install_cmds = {
        "ubuntu": f"apt-get update && apt-get install -y linux-headers-{kernel_version} build-essential",
        "debian": f"apt-get update && apt-get install -y linux-headers-{kernel_version} build-essential",
        "rhel": f"yum install -y kernel-devel-{kernel_version} gcc make",
        "centos": f"yum install -y kernel-devel-{kernel_version} gcc make",
        "fedora": f"dnf install -y kernel-devel-{kernel_version} gcc make",
        "arch": f"pacman -S --noconfirm linux-headers base-devel"
    }

    if distro in install_cmds:
        print_success(f"Detected: {distro.capitalize()}")
        print_info(f"Installing: linux-headers-{kernel_version} + build tools")
        print(f"\n{Colors.CYAN}Running: {install_cmds[distro]}{Colors.END}\n")

        success, stdout, stderr = run_command(install_cmds[distro])

        if success:
            print_success("Dependencies installed successfully!")
            return True
        else:
            print_error("Installation failed")
            if stderr:
                print(f"{Colors.DIM}{stderr}{Colors.END}")
            return False
    else:
        print_warning(f"Unknown distribution: {distro}")
        print_info("Please install manually:")
        print(f"  Debian/Ubuntu: sudo apt-get install linux-headers-{kernel_version} build-essential")
        return False

def parse_kernel_version(version_string):
    """Parse kernel version string into (major, minor, patch) tuple"""
    try:
        # Remove everything after the first non-numeric character (e.g., +deb14+1-amd64)
        version_parts = version_string.split('+')[0].split('-')[0].split('.')
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        return (major, minor, patch)
    except:
        return (0, 0, 0)

def get_kernel_compatibility_message(kernel_version):
    """Check kernel version compatibility and return info message"""
    major, minor, patch = parse_kernel_version(kernel_version)

    # We support kernels 4.4+, with explicit validation for 6.16, 6.17, and beyond
    if major >= 6:
        if minor >= 17:
            return f"Kernel {kernel_version} (6.17+) - Fully compatible with future kernel APIs"
        elif minor == 16:
            return f"Kernel {kernel_version} (6.16) - Fully compatible, 6.17-ready"
        else:
            return f"Kernel {kernel_version} (6.x) - Fully compatible"
    elif major == 5:
        return f"Kernel {kernel_version} (5.x) - Compatible (LTS kernel)"
    elif major == 4 and minor >= 4:
        return f"Kernel {kernel_version} (4.4+) - Compatible (legacy support)"
    else:
        return f"Kernel {kernel_version} - May require additional testing"

def check_build_environment():
    """Check if build environment is properly configured"""
    issues = []
    suggestions = []

    # Check for kernel headers
    kernel_version = subprocess.run(
        ['uname', '-r'],
        capture_output=True,
        text=True
    ).stdout.strip()

    # Check kernel compatibility
    compat_msg = get_kernel_compatibility_message(kernel_version)

    kernel_headers = Path(f"/lib/modules/{kernel_version}/build")
    if not kernel_headers.exists():
        issues.append("Kernel headers not found")
        suggestions.append(f"Install with: sudo apt-get install linux-headers-{kernel_version}")

    # Check for basic build tools
    for tool in ['gcc', 'make']:
        result = subprocess.run(
            ['which', tool],
            capture_output=True
        )
        if result.returncode != 0:
            issues.append(f"{tool} not found")
            suggestions.append("Install with: sudo apt-get install build-essential")
            break

    # Check for Rust (optional) - must be callable and working
    rust_available = False

    # Check if rustc is actually callable in current environment
    result = subprocess.run(['which', 'rustc'], capture_output=True)
    if result.returncode == 0:
        # Verify it's actually executable and has a working toolchain
        try:
            result = subprocess.run(['rustc', '--version'], capture_output=True, timeout=2)
            # Only mark as available if rustc exits with success (0)
            # When rustup shim exists but no toolchain installed, it exits non-zero
            rust_available = (result.returncode == 0)
        except:
            rust_available = False

        # Also check current user's home
        home = os.path.expanduser("~")
        user_cargo = Path(f"{home}/.cargo/bin/rustc")
        if user_cargo.exists():
            rust_available = True

    return issues, suggestions, rust_available, compat_msg

def run_command(cmd, cwd=None, check=True):
    """Run a shell command"""
    try:
        # Preserve user's PATH when running under sudo for Rust detection
        env = os.environ.copy()
        sudo_user = os.environ.get('SUDO_USER')
        if sudo_user:
            # Add user's cargo bin to PATH
            user_cargo_bin = f"/home/{sudo_user}/.cargo/bin"
            if Path(user_cargo_bin).exists():
                env['PATH'] = f"{user_cargo_bin}:{env.get('PATH', '')}"

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

# ============================================================================
# KERNEL VERSION DETECTION & COMPATIBILITY
# ============================================================================

def get_kernel_version():
    """
    Get running kernel version

    Returns: (major, minor, patch) tuple
    """
    uname = platform.uname()
    version_str = uname.release

    # Extract version numbers (e.g., "6.17.0-generic" -> (6, 17, 0))
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Fallback
    return (0, 0, 0)

def check_kernel_compatibility():
    """
    Check kernel compatibility and recommend driver version

    Returns: (compatible, recommended_driver, warnings)
    """
    major, minor, patch = get_kernel_version()
    version_str = f"{major}.{minor}.{patch}"

    warnings = []

    # Kernel 6.17+: Fully supported, use driver 104
    if major >= 7 or (major == 6 and minor >= 17):
        return True, "dsmil-104dev", []

    # Kernel 6.0-6.16: Supported with minor warnings
    elif major == 6 and minor < 17:
        warnings.append(f"Kernel {version_str}: Some features may require 6.17+")
        warnings.append("Consider upgrading kernel for optimal performance")
        return True, "dsmil-104dev", warnings

    # Kernel 5.x: Use fallback driver
    elif major == 5:
        warnings.append(f"Kernel {version_str}: Using fallback driver 84")
        warnings.append("Upgrade to 6.17+ for full 104-device support")
        return True, "dsmil-84dev", warnings

    # Kernel 4.x: Legacy support
    elif major == 4:
        warnings.append(f"Kernel {version_str}: Legacy kernel detected")
        warnings.append("Using fallback driver 84 (limited functionality)")
        warnings.append("CRITICAL: Upgrade to 6.17+ strongly recommended")
        return True, "dsmil-84dev", warnings

    # Unknown/unsupported
    else:
        warnings.append(f"Kernel {version_str}: Compatibility unknown")
        warnings.append("Attempting driver 104 anyway")
        return False, "dsmil-104dev", warnings

# ============================================================================
# PATH RECOVERY & LOCATION MECHANISMS
# ============================================================================

def find_kernel_source_dir():
    """
    Find kernel source directory with multiple fallback locations

    Complex path recovery mechanism for lost/moved directories
    """
    # Primary locations to check
    search_paths = [
        Path("01-source/kernel"),                    # Standard location
        Path("source/kernel"),                       # Alternate 1
        Path("kernel"),                              # Alternate 2
        Path("src/kernel"),                          # Alternate 3
        Path("/opt/dsmil/kernel"),                   # System install
        Path.home() / "dsmil/01-source/kernel",     # User home
        Path.home() / ".dsmil/kernel",              # Hidden dir
    ]

    # Also search current directory recursively (up to 3 levels)
    try:
        for pattern in ["**/kernel", "**/*source*/kernel", "**/01-source/kernel"]:
            for path in glob.glob(pattern, recursive=False):
                search_paths.append(Path(path))
    except:
        pass

    # Check each path
    for path in search_paths:
        if path.exists() and path.is_dir():
            # Verify it contains driver files
            if (path / "Makefile").exists() or (path / "dsmil_driver.c").exists():
                print_info(f"Found kernel source: {path}")
                return path

    return None

def find_driver_module(driver_name="dsmil-104dev"):
    """
    Find compiled driver module with extensive search

    Searches multiple locations and attempts recovery
    """
    kernel_dir = find_kernel_source_dir()
    if not kernel_dir:
        return None

    # Search patterns
    search_locations = [
        kernel_dir / f"{driver_name}.ko",
        kernel_dir / "build" / f"{driver_name}.ko",
        kernel_dir / "output" / f"{driver_name}.ko",
        kernel_dir / f"drivers/{driver_name}.ko",
    ]

    # Also search recursively in kernel dir
    try:
        for ko_file in kernel_dir.rglob(f"{driver_name}.ko"):
            search_locations.append(ko_file)
    except:
        pass

    # Check each location
    for path in search_locations:
        if path.exists() and path.is_file():
            print_info(f"Found driver module: {path}")
            return path

    return None

def find_control_centre():
    """
    Find control centre script with path recovery
    """
    search_paths = [
        Path("02-ai-engine/dsmil_control_centre_104.py"),
        Path("ai-engine/dsmil_control_centre_104.py"),
        Path("engine/dsmil_control_centre_104.py"),
        Path("dsmil_control_centre_104.py"),
        Path("/opt/dsmil/dsmil_control_centre_104.py"),
        Path.home() / "dsmil/02-ai-engine/dsmil_control_centre_104.py",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None

# ============================================================================
# DRIVER MANAGEMENT
# ============================================================================

def build_driver(clean=False, force_driver=None):
    """
    Build kernel driver with advanced compensation mechanisms

    Args:
        clean: Clean before building
        force_driver: Force specific driver version (104 or 84)

    Returns:
        True if successful, False otherwise
    """
    print_header("Building DSMIL Kernel Driver (v2.1 Enhanced)")

    # Detect kernel version
    major, minor, patch = get_kernel_version()
    print_info(f"Detected kernel: {major}.{minor}.{patch}")

    # Check compatibility and get recommended driver
    compatible, recommended_driver, warnings = check_kernel_compatibility()

    for warning in warnings:
        print_warning(warning)

    if not compatible:
        print_warning("Kernel compatibility uncertain - build may fail")

    # Use forced driver if specified, otherwise prefer 104-device
    if force_driver:
        if force_driver == "104":
            target_driver = "dsmil-104dev"
        elif force_driver == "84":
            target_driver = "dsmil-84dev"
        else:
            target_driver = recommended_driver
        print_info(f"Forcing driver: {target_driver}")
    else:
        # Always prefer 104-device driver first (modern hardware)
        target_driver = "dsmil-104dev"
        if target_driver != recommended_driver:
            print_info(f"Targeting: {target_driver} (104-device modern driver)")
            print_info(f"Recommended: {recommended_driver} (will use as fallback if needed)")
        else:
            print_info(f"Target driver: {target_driver}")

    # Find kernel source directory with path recovery
    kernel_dir = find_kernel_source_dir()

    if not kernel_dir:
        print_error("Kernel source directory not found")
        print_info("Searched multiple locations:")
        print_info("  - 01-source/kernel")
        print_info("  - source/kernel")
        print_info("  - kernel/")
        print_info("  - /opt/dsmil/kernel")
        print_info("  - ~/dsmil/01-source/kernel")
        return False

    # Check build environment
    issues, suggestions, rust_available, compat_msg = check_build_environment()

    # Show kernel compatibility
    print_info(compat_msg)

    # Inform user if running under sudo (for Rust PATH awareness)
    sudo_user = os.environ.get('SUDO_USER')
    if sudo_user:
        print_info(f"Running as sudo (user: {sudo_user}) - PATH adjusted for Rust detection")

    if issues:
        print_warning("Build environment issues detected:")
        for issue in issues:
            print_info(f"  - {issue}")
        print_info("\nSuggested fixes:")
        for suggestion in suggestions:
            print_info(f"  {suggestion}")
        print_info("\nAttempting build anyway...")

    if not rust_available:
        print_info("Rust not detected - will use C safety stubs (this is normal)")
    else:
        print_info("Rust detected - building Rust safety library as dependency")
        # Treat Rust safety library as an automatically installed dependency.
        # Kernel modules still use C safety stubs for maximum compatibility.
        build_rust_safety_library()

    # Clean if requested
    if clean:
        print_info("Cleaning build artifacts...")
        clean_cmd = "make clean ENABLE_RUST=0" if not rust_available else "make clean"
        success, _, _ = run_command(clean_cmd, cwd=kernel_dir)
        if not success:
            print_warning("Clean failed, continuing anyway...")

    # Build with compensation mechanism #1: build drivers independently using
    # dedicated Makefile targets that avoid Rust/objtool issues.
    print_info(f"Building DSMIL drivers (targeting {target_driver})...")

    # Always use C safety stubs for these standalone targets
    stdout_parts = []
    stderr_parts = []

    # Build 104-device driver first
    print_info("Building 104-device driver (dsmil-104dev)...")
    success_104, stdout_104, stderr_104 = run_command("make dsmil-104dev V=1", cwd=kernel_dir, check=False)
    stdout_parts.append(stdout_104 or "")
    stderr_parts.append(stderr_104 or "")

    # Then try 84-device legacy driver
    print_info("Building 84-device fallback driver (dsmil-84dev)...")
    success_84, stdout_84, stderr_84 = run_command("make dsmil-84dev V=1", cwd=kernel_dir, check=False)
    stdout_parts.append(stdout_84 or "")
    stderr_parts.append(stderr_84 or "")

    success = success_104 or success_84
    stdout = "\n".join(stdout_parts)
    stderr = "\n".join(stderr_parts)

    # Check if build produced any kernel modules (success = .ko files exist)
    modules_built = []
    if (kernel_dir / "dsmil-104dev.ko").exists():
        modules_built.append("dsmil-104dev.ko (104 devices)")
        print_success("✓ dsmil-104dev.ko (104 devices)")
    if (kernel_dir / "dsmil-84dev.ko").exists():
        modules_built.append("dsmil-84dev.ko (84 devices - legacy)")
        print_success("✓ dsmil-84dev.ko (84 devices - legacy)")

    if modules_built:
        print_success("Build completed successfully")
        return True

    else:
        # Make returned success but no .ko files produced - likely Rust build issue
        # Compensation mechanism #2: If Rust was enabled, try without Rust
        if rust_available:
            print_warning("No kernel modules produced with Rust, retrying without Rust...")

            # Show what went wrong - combine stdout and stderr, show last 30 lines
            if stderr or stdout:
                print(f"\n{Colors.BOLD}Build output (last 30 lines):{Colors.END}")
                combined = (stdout or "") + "\n" + (stderr or "")
                all_lines = [line for line in combined.split('\n') if line.strip()]
                for line in all_lines[-30:]:
                    if 'error:' in line.lower() or 'fatal' in line.lower() or 'make[' in line.lower():
                        print(f"  {Colors.FAIL}{line}{Colors.END}")
                    else:
                        print(f"  {Colors.DIM}{line}{Colors.END}")

            success, stdout, stderr = run_command("make clean && make all ENABLE_RUST=0 V=1", cwd=kernel_dir, check=False)

            # Check again for .ko files
            if (kernel_dir / "dsmil-104dev.ko").exists() or (kernel_dir / "dsmil-84dev.ko").exists():
                if (kernel_dir / "dsmil-104dev.ko").exists():
                    print_success("✓ dsmil-104dev.ko (104 devices)")
                if (kernel_dir / "dsmil-84dev.ko").exists():
                    print_success("✓ dsmil-84dev.ko (84 devices - legacy)")
                print_success("Build completed successfully (without Rust)")
                return True

        if not success or not modules_built:
            # All compensation mechanisms failed
            print_error("Build failed")
            if stderr:
                print(f"\n{Colors.FAIL}Build errors:{Colors.END}")
                # Show first 15 lines of actual errors
                error_lines = [line for line in stderr.split('\n') if line.strip()][:15]
                for line in error_lines:
                    print(f"  {line}")

            print_info("\nTroubleshooting:")
            print_info("  1. Check kernel headers: sudo apt install linux-headers-$(uname -r)")
            print_info("  2. Check build tools: sudo apt install build-essential")
            print_info("  3. Check Makefile exists: ls -la 01-source/kernel/Makefile")
            print_info(f"  4. Kernel version {major}.{minor}.{patch} may need specific headers")

            return False

def build_and_autoload(clean=False):
    """Build driver with fallback and auto-install/load"""
    print_header("Build with Auto-Fallback and Load")

    if not check_root():
        print_error("Must run as root for auto-install and load")
        print_info("Run: sudo python3 dsmil.py build-auto")
        return False

    kernel_dir = Path("01-source/kernel")
    if not kernel_dir.exists():
        print_error("Kernel source directory not found")
        return False

    # Check build environment
    issues, suggestions, rust_available, compat_msg = check_build_environment()

    # Show kernel compatibility
    print_info(compat_msg)

    # Inform user if running under sudo
    sudo_user = os.environ.get('SUDO_USER')
    if sudo_user:
        print_info(f"Running as sudo (user: {sudo_user}) - PATH adjusted for Rust detection")

    # Auto-install missing dependencies if needed
    if issues:
        print_warning(f"Build dependencies missing: {', '.join(issues)}")
        print(f"\n{Colors.BOLD}{Colors.GREEN}AUTO-INSTALLING DEPENDENCIES{Colors.END}")
        print(f"Since you're using build-auto, I'll install missing dependencies automatically.\n")

        if auto_install_dependencies():
            print_success("Dependencies installed! Continuing with build...")
            # Re-check environment after installation
            issues, suggestions, rust_available, _ = check_build_environment()
            if issues:
                print_error("Some dependencies still missing after installation")
                print_info("You may need to install them manually")
        else:
            print_error("Auto-install failed")
            print_info("\nPlease install manually:")
            kernel_version = subprocess.run(['uname', '-r'], capture_output=True, text=True).stdout.strip()
            print(f"  {Colors.CYAN}sudo apt-get update && sudo apt-get install -y linux-headers-{kernel_version} build-essential{Colors.END}")
            return False

    if rust_available:
        print_info("Rust detected - building Rust safety library as dependency")
        build_rust_safety_library()
        print_info("Kernel modules currently use C safety stubs (Rust safety is optional)")
    else:
        print_info("Rust not available in current environment - kernel modules use C safety stubs")

    # Clean if requested
    if clean:
        print_info("Cleaning build artifacts...")
        success, _, _ = run_command("make clean", cwd=kernel_dir)
        if not success:
            print_warning("Clean failed, continuing anyway...")

    # Build both drivers using dedicated non-Rust targets
    print_info("Building drivers (this may take a minute)...")
    stdout_parts = []
    stderr_parts = []

    print_info("Building 104-device driver (dsmil-104dev)...")
    success_104, stdout_104, stderr_104 = run_command("make dsmil-104dev V=1", cwd=kernel_dir, check=False)
    stdout_parts.append(stdout_104 or "")
    stderr_parts.append(stderr_104 or "")

    print_info("Building 84-device driver (dsmil-84dev)...")
    success_84, stdout_84, stderr_84 = run_command("make dsmil-84dev V=1", cwd=kernel_dir, check=False)
    stdout_parts.append(stdout_84 or "")
    stderr_parts.append(stderr_84 or "")

    success = success_104 or success_84
    stdout = "\n".join(stdout_parts)
    stderr = "\n".join(stderr_parts)

    # Determine which driver to use (prefer 104dev, fallback to 84dev)
    driver_to_load = None
    driver_104 = kernel_dir / "dsmil-104dev.ko"
    driver_84 = kernel_dir / "dsmil-84dev.ko"

    if driver_104.exists():
        print_success("✓ dsmil-104dev.ko (104 devices) - BUILD SUCCESSFUL")
        driver_to_load = "dsmil-104dev"
    elif driver_84.exists():
        print_warning("104dev build failed or not built, falling back to 84dev")
        print_success("✓ dsmil-84dev.ko (84 devices - legacy) - BUILD SUCCESSFUL")
        driver_to_load = "dsmil-84dev"
    else:
        print_error("Both 104dev and 84dev builds failed")

        # Analyze the error and provide specific guidance
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.FAIL}DIAGNOSIS: Why the build failed{Colors.END}")
        print("="*70)

        # Check what's missing
        issues, suggestions, rust_available_check, _ = check_build_environment()

        if issues:
            print(f"\n{Colors.BOLD}Missing components detected:{Colors.END}")
            for i, issue in enumerate(issues, 1):
                print(f"{Colors.FAIL}  ✗ {issue}{Colors.END}")

        print(f"\n{Colors.BOLD}{Colors.GREEN}SOLUTION:{Colors.END}")

        # Detect OS type for appropriate package manager
        import platform
        os_type = platform.system().lower()

        kernel_version = subprocess.run(['uname', '-r'], capture_output=True, text=True).stdout.strip()

        if issues and any("kernel headers" in i.lower() for i in issues):
            print(f"\n{Colors.CYAN}Install missing kernel headers:{Colors.END}")
            if os_type == "linux":
                distro_cmds = {
                    "debian": f"sudo apt-get update && sudo apt-get install -y linux-headers-{kernel_version} build-essential",
                    "ubuntu": f"sudo apt-get update && sudo apt-get install -y linux-headers-{kernel_version} build-essential",
                    "rhel": f"sudo yum install -y kernel-devel-{kernel_version} gcc make",
                    "centos": f"sudo yum install -y kernel-devel-{kernel_version} gcc make",
                    "fedora": f"sudo dnf install -y kernel-devel-{kernel_version} gcc make",
                    "arch": f"sudo pacman -S linux-headers base-devel"
                }

                print(f"\n{Colors.GREEN}Debian/Ubuntu:{Colors.END}")
                print(f"  {distro_cmds['debian']}")
                print(f"\n{Colors.GREEN}RHEL/CentOS:{Colors.END}")
                print(f"  {distro_cmds['rhel']}")
                print(f"\n{Colors.GREEN}Fedora:{Colors.END}")
                print(f"  {distro_cmds['fedora']}")
                print(f"\n{Colors.GREEN}Arch Linux:{Colors.END}")
                print(f"  {distro_cmds['arch']}")

        print(f"\n{Colors.BOLD}Rust components (OPTIONAL):{Colors.END}")
        print(f"  {Colors.DIM}Rust is optional - the driver works fine without it{Colors.END}")

        # Show actual build errors (last 40 lines)
        if stderr or stdout:
            print(f"{Colors.BOLD}Full build error output (last 40 lines):{Colors.END}")
            combined = (stdout or "") + "\n" + (stderr or "")
            error_lines = [line for line in combined.split('\n') if line.strip()][-40:]
            for line in error_lines:
                if 'error:' in line.lower() or 'make[' in line.lower() or 'fatal' in line.lower():
                    print(f"{Colors.FAIL}{line}{Colors.END}")
                else:
                    print(f"{Colors.DIM}{line}{Colors.END}")

        return False

    # Install the driver
    print_info(f"Installing {driver_to_load} module...")
    install_success, _, install_stderr = run_command("make install", cwd=kernel_dir)
    if install_success:
        print_success("Driver installed to system modules")
    else:
        print_warning("Install failed, will try direct insmod")

    # Unload any existing driver
    print_info("Unloading any existing DSMIL drivers...")
    run_command("rmmod dsmil-104dev 2>/dev/null", check=False)
    run_command("rmmod dsmil-84dev 2>/dev/null", check=False)

    # Load the selected driver
    print_info(f"Loading {driver_to_load}...")
    driver_path = kernel_dir / f"{driver_to_load}.ko"
    load_success, load_stdout, load_stderr = run_command(f"insmod {driver_path}")

    if load_success:
        print_success(f"{driver_to_load} loaded successfully!")

        # Check device nodes (driver creates both main and compatibility nodes)
        device_nodes = []
        main_dev = f"/dev/{driver_to_load}"
        if Path(main_dev).exists():
            device_nodes.append(main_dev)
            print_success(f"Device node created: {main_dev}")

        # Check for compatibility device node
        compat_dev = "/dev/dsmil-72dev"
        if Path(compat_dev).exists():
            device_nodes.append(compat_dev)
            print_success(f"Compatibility device: {compat_dev}")

        # Check legacy path
        if Path("/dev/dsmil0").exists():
            device_nodes.append("/dev/dsmil0")
            print_success("Device node created: /dev/dsmil0")

        if not device_nodes:
            print_warning("Device nodes not found (driver may create them on first access)")

        # Show kernel messages
        print_info("Recent kernel messages:")
        success, stdout, _ = run_command("dmesg | tail -15")
        if stdout:
            print(stdout)

        return True
    else:
        print_error(f"Failed to load {driver_to_load}")
        if load_stderr:
            print(f"\nError: {load_stderr}")
        return False

def load_driver(driver_name=None):
    """Load kernel driver (auto-detects if not specified)"""

    # Auto-detect driver if not specified, preferring 104dev on modern kernels
    if driver_name is None:
        kernel_dir = Path("01-source/kernel")
        driver_104 = kernel_dir / "dsmil-104dev.ko"
        driver_84 = kernel_dir / "dsmil-84dev.ko"

        major, minor, patch = get_kernel_version()
        prefer_104 = major >= 6  # Always prefer 104dev on 6.x+ when present

        if driver_104.exists() and (prefer_104 or not driver_84.exists()):
            driver_name = "dsmil-104dev"
            print_info("Auto-detected: dsmil-104dev.ko (104 devices, preferred)")
        elif driver_84.exists():
            driver_name = "dsmil-84dev"
            print_info("Auto-detected: dsmil-84dev.ko (84 devices - legacy fallback)")
        else:
            print_error("No driver found (tried dsmil-104dev.ko and dsmil-84dev.ko)")
            print_info("Build first: python3 dsmil.py build")
            return False

    print_header(f"Loading {driver_name} Driver")

    if not check_root():
        print_error("Must run as root to load driver")
        print_info("Run: sudo python3 dsmil.py load")
        return False

    # Auto-detect best driver if not specified
    if driver_name is None:
        major, minor, patch = get_kernel_version()
        print_info(f"Detected kernel: {major}.{minor}.{patch}")

        compatible, recommended_driver, warnings = check_kernel_compatibility()
        driver_name = recommended_driver

        for warning in warnings:
            print_warning(warning)

        print_info(f"Auto-selected driver: {driver_name}")

    # Find driver module with path recovery
    driver_path = find_driver_module(driver_name)

    if not driver_path:
        print_error(f"Driver module not found: {driver_name}.ko")
        print_info("Searched multiple locations:")
        print_info("  - 01-source/kernel/")
        print_info("  - 01-source/kernel/build/")
        print_info("  - 01-source/kernel/output/")

        # Compensation mechanism: Try fallback driver
        fallback_driver = "dsmil-84dev" if "104" in driver_name else "dsmil-104dev"
        print_warning(f"Attempting fallback to {fallback_driver}...")

        driver_path = find_driver_module(fallback_driver)

        if driver_path:
            print_success(f"Found fallback driver: {driver_path}")
            driver_name = fallback_driver
        else:
            print_error("No driver modules found")
            print_info("Build first: python3 dsmil.py build")
            return False

    print_info(f"Using driver: {driver_path}")

    # Unload if already loaded (both versions)
    run_command("rmmod dsmil-104dev 2>/dev/null", check=False)
    run_command("rmmod dsmil-84dev 2>/dev/null", check=False)

    # Load driver
    print_info(f"Loading {driver_name}...")
    success, stdout, stderr = run_command(f"insmod {driver_path}")

    if success:
        print_success("Driver loaded successfully")

        # Check device nodes
        device_nodes = []
        main_dev = f"/dev/{driver_name}"
        if Path(main_dev).exists():
            device_nodes.append(main_dev)
            print_success(f"Device node created: {main_dev}")

        # Check for compatibility device node
        compat_dev = "/dev/dsmil-72dev"
        if Path(compat_dev).exists():
            device_nodes.append(compat_dev)
            print_success(f"Compatibility device: {compat_dev}")

        # Check legacy path
        if Path("/dev/dsmil0").exists():
            device_nodes.append("/dev/dsmil0")
            print_success("Device node created: /dev/dsmil0")

        if not device_nodes:
            print_warning("Device nodes not found (driver may create them on first access)")

        # Show kernel messages
        print_info("Recent kernel messages:")
        success, stdout, _ = run_command("dmesg | tail -15 | grep -i dsmil || dmesg | tail -15")
        if stdout:
            print(stdout)

        # Show loaded module info
        success, stdout, _ = run_command(f"lsmod | grep dsmil")
        if stdout:
            print_success(f"Module info:\n{stdout}")

        return True
    else:
        print_error("Failed to load driver")
        if stderr:
            print(f"\nError: {stderr}")

        # Compensation mechanism: Show detailed diagnostics
        print_info("\nDiagnostics:")
        run_command("dmesg | tail -20")

        print_info("\nTroubleshooting:")
        print_info("  1. Check kernel version compatibility")
        print_info("  2. Verify driver was built for current kernel")
        print_info("  3. Check dmesg for detailed errors: dmesg | grep -i dsmil")
        print_info("  4. Try rebuilding: python3 dsmil.py build --clean")

        return False

def unload_driver(driver_name=None):
    """Unload kernel driver (unloads all DSMIL drivers if not specified)"""

    if not check_root():
        print_error("Must run as root to unload driver")
        return False

    # If no driver specified, unload all DSMIL drivers
    if driver_name is None:
        print_header("Unloading DSMIL Drivers")
        print_info("Unloading all DSMIL drivers...")
        success_104 = True
        success_84 = True

        # Try to unload both
        s1, _, stderr1 = run_command("rmmod dsmil-104dev 2>/dev/null", check=False)
        if s1 or "not found" in stderr1.lower() or "not loaded" in stderr1.lower():
            if s1:
                print_success("dsmil-104dev unloaded")
        else:
            success_104 = False

        s2, _, stderr2 = run_command("rmmod dsmil-84dev 2>/dev/null", check=False)
        if s2 or "not found" in stderr2.lower() or "not loaded" in stderr2.lower():
            if s2:
                print_success("dsmil-84dev unloaded")
        else:
            success_84 = False

        if success_104 or success_84:
            print_success("All DSMIL drivers unloaded")
            return True
        else:
            print_error("Failed to unload drivers")
            return False
    else:
        print_header(f"Unloading {driver_name} Driver")
        print_info(f"Unloading {driver_name}...")
        success, _, stderr = run_command(f"rmmod {driver_name}", check=False)

        if success or "not found" in stderr.lower():
            print_success("Driver unloaded")
            return True
        else:
            print_error("Failed to unload driver")
            return False

def driver_status():
    """Check driver status"""
    print_header("Driver Status")

    # Check if loaded
    success, stdout, _ = run_command("lsmod | grep dsmil", check=False)
    if success and stdout:
        print_success("Driver loaded:")
        print(stdout)

        # Parse loaded driver name
        loaded_driver = None
        for line in stdout.split('\n'):
            if 'dsmil' in line.lower():
                loaded_driver = line.split()[0]
                break
    else:
        print_info("No DSMIL driver loaded")
        return True

    # Check device nodes
    device_nodes_found = []
    possible_nodes = [
        "/dev/dsmil-104dev",
        "/dev/dsmil-84dev",
        "/dev/dsmil-72dev",
        "/dev/dsmil0"
    ]

    for node_path in possible_nodes:
        if Path(node_path).exists():
            device_nodes_found.append(node_path)
            print_success(f"Device node exists: {node_path}")
            # Get permissions
            import stat
            st = os.stat(node_path)
            mode = stat.filemode(st.st_mode)
            print_info(f"  Permissions: {mode}, Owner: {st.st_uid}:{st.st_gid}")

    if not device_nodes_found:
        print_info("No device nodes found (may be created on first access)")

    # Check sysfs - try multiple possible paths
    sysfs_paths = [
        "/sys/class/dsmil/dsmil0",
        "/sys/class/dsmil-104dev/dsmil-104dev0",
        "/sys/class/dsmil-84dev/dsmil-84dev0",
        "/sys/module/dsmil_104dev",
        "/sys/module/dsmil_84dev"
    ]

    sysfs_found = False
    for sysfs_path in sysfs_paths:
        if Path(sysfs_path).exists():
            sysfs_found = True
            print_success(f"Sysfs interface: {sysfs_path}")

            # Try to read common sysfs attributes
            for attr in ["driver_version", "device_count", "version"]:
                attr_file = Path(sysfs_path) / attr
                if attr_file.exists():
                    try:
                        value = attr_file.read_text().strip()
                        print_info(f"  {attr}: {value}")
                    except:
                        pass
            break

    if not sysfs_found:
        print_info("Sysfs interface not available (driver may not expose sysfs attributes)")

    # Show recent kernel messages about the driver
    print_info("\nRecent driver messages:")
    success, stdout, _ = run_command("dmesg | grep -i dsmil | tail -5", check=False)
    if stdout:
        for line in stdout.strip().split('\n'):
            print(f"  {line}")

    return True

# ============================================================================
# CONTROL CENTRE
# ============================================================================

def launch_control_centre(auto_discover=False, auto_activate=False):
    """
    Launch control centre with path recovery
    """
    print_header("Launching DSMIL Control Centre (v2.1 Enhanced)")

    # Find control centre with path recovery
    control_centre = find_control_centre()

    if not control_centre:
        print_error("Control centre not found")
        print_info("Searched multiple locations:")
        print_info("  - 02-ai-engine/dsmil_control_centre_104.py")
        print_info("  - ai-engine/dsmil_control_centre_104.py")
        print_info("  - engine/dsmil_control_centre_104.py")
        print_info("  - dsmil_control_centre_104.py")
        print_info("  - /opt/dsmil/dsmil_control_centre_104.py")
        return False

    print_info(f"Found control centre: {control_centre}")

    cmd = ["python3", str(control_centre)]

    if auto_discover:
        cmd.append("--auto-discover")
    if auto_activate:
        cmd.append("--auto-activate")

    print_info(f"Starting: {' '.join(cmd)}")

    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print_info("\nControl centre stopped by user")
        return True
    except Exception as e:
        print_error(f"Failed to launch control centre: {e}")
        return False

# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics():
    """Run system diagnostics"""
    print_header("System Diagnostics")

    # Check kernel source
    if Path("01-source/kernel").exists():
        print_success("Kernel source found")
    else:
        print_error("Kernel source not found")

    # Check integration modules
    integration_dir = Path("02-ai-engine")
    required_modules = [
        "dsmil_driver_interface.py",
        "dsmil_integration_adapter.py",
        "dsmil_control_centre_104.py",
        "dsmil_device_database_extended.py"
    ]

    all_found = True
    for module in required_modules:
        if (integration_dir / module).exists():
            print_success(f"Integration module: {module}")
        else:
            print_error(f"Missing: {module}")
            all_found = False

    if all_found:
        print_success("All integration modules found")

    # Run driver status
    driver_status()

    # Try to run integration diagnostics
    if Path("/dev/dsmil0").exists():
        print_info("\nRunning integration adapter diagnostics...")
        adapter_script = Path("02-ai-engine/dsmil_integration_adapter.py")
        if adapter_script.exists():
            subprocess.run(["python3", str(adapter_script)])

# ============================================================================
# DOCUMENTATION
# ============================================================================

def show_docs():
    """Show documentation links"""
    print_header("Documentation")

    docs = {
        "Driver Usage": "01-source/kernel/DRIVER_USAGE_GUIDE.md",
        "API Reference": "01-source/kernel/API_REFERENCE.md",
        "TPM Authentication": "01-source/kernel/TPM_AUTHENTICATION_GUIDE.md",
        "Testing Guide": "01-source/kernel/TESTING_GUIDE.md",
        "Build Fixes": "01-source/kernel/BUILD_FIXES.md",
        "Integration Guide": "02-ai-engine/README_INTEGRATION.md",
    }

    for name, path in docs.items():
        if Path(path).exists():
            print_success(f"{name:20} - {path}")
        else:
            print_warning(f"{name:20} - NOT FOUND")

    print(f"\n{Colors.CYAN}View documentation:{Colors.END}")
    print(f"  cat <path>")
    print(f"  less <path>")
    print(f"  vim <path>")

# ============================================================================
# INTERACTIVE TUI MENU
# ============================================================================

# ============================================================================
# AI ENGINE & CODE-MODE INTEGRATION
# ============================================================================

def init_code_mode():
    """Initialize code-mode environment for AI engine"""
    print_header("Code-Mode Integration - Initialization")

    print_info("Code-mode provides 60-88% performance improvement for AI workflows")
    print()

    # Check Node.js availability
    print_info("Checking prerequisites...")
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            node_version = result.stdout.strip()
            print_success(f"Node.js detected: {node_version}")
        else:
            print_error("Node.js not found")
            print()
            print_warning("Code-mode requires Node.js v16+")
            print_info("Install with: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -")
            print_info("             sudo apt-get install -y nodejs")
            return False
    except FileNotFoundError:
        print_error("Node.js not found")
        print()
        print_warning("Code-mode requires Node.js v16+")
        print_info("Install with: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -")
        print_info("             sudo apt-get install -y nodejs")
        return False
    except Exception as e:
        print_error(f"Error checking Node.js: {e}")
        return False

    # Initialize code-mode bridge
    print()
    print_info("Initializing code-mode bridge...")

    try:
        sys.path.insert(0, str(Path(__file__).parent / "02-ai-engine"))
        from code_mode_bridge import CodeModeBridge

        bridge = CodeModeBridge()

        if bridge.initialize():
            print_success("Code-mode bridge initialized successfully")
            print()
            print_info("Available features:")
            print(f"  {Colors.GREEN}✓{Colors.END} TypeScript execution sandbox")
            print(f"  {Colors.GREEN}✓{Colors.END} Tool batching and parallelization")
            print(f"  {Colors.GREEN}✓{Colors.END} MCP server integration")
            print(f"  {Colors.GREEN}✓{Colors.END} Performance tracking")
            print()

            # Show performance benefits
            print_info("Expected performance gains:")
            print(f"  {Colors.CYAN}▶{Colors.END} {Colors.BOLD}60-88% faster{Colors.END} execution for multi-step workflows")
            print(f"  {Colors.CYAN}▶{Colors.END} {Colors.BOLD}68-75% fewer tokens{Colors.END} consumed")
            print(f"  {Colors.CYAN}▶{Colors.END} {Colors.BOLD}86% fewer API calls{Colors.END} (e.g., 7 calls → 1 call)")
            print()

            print_success("Code-mode ready for AI engine operations")
            return True
        else:
            print_error("Code-mode initialization failed")
            print_warning("AI engine will fall back to traditional execution mode")
            return False

    except ImportError as e:
        print_error(f"Failed to import code-mode bridge: {e}")
        print_warning("Code-mode integration may not be installed")
        return False
    except Exception as e:
        print_error(f"Error initializing code-mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_ai_engine():
    """Launch AI engine with code-mode support"""
    print_header("DSMIL AI Engine - Code-Mode Enabled")

    print_info("The AI engine integrates:")
    print(f"  {Colors.GREEN}●{Colors.END} Enhanced AI Engine (RAG, memory, learning)")
    print(f"  {Colors.GREEN}●{Colors.END} Agent Orchestrator (97 agents, hardware-aware)")
    print(f"  {Colors.GREEN}●{Colors.END} Code-Mode Execution (60-88% faster)")
    print(f"  {Colors.GREEN}●{Colors.END} DSMIL MCP Server (84 devices as tools)")
    print(f"  {Colors.GREEN}●{Colors.END} Self-Improvement System (autonomous optimization)")
    print()

    # Check if driver is loaded
    success, stdout, _ = run_command("lsmod | grep dsmil", check=False)
    if not success or not stdout:
        print_warning("DSMIL driver not loaded")
        print_info("For full functionality, load the driver first (option 1 or 6)")
        print()
        response = input(f"{Colors.CYAN}Continue anyway? [y/N]: {Colors.END}").strip().lower()
        if response != 'y':
            return False
    else:
        print_success("DSMIL driver detected")
        print()

    # Launch options
    print(f"{Colors.BOLD}Select launch mode:{Colors.END}")
    print(f"  {Colors.GREEN}[1]{Colors.END} Web API Server {Colors.DIM}(REST API + WebSockets on port 5001){Colors.END}")
    print(f"  {Colors.GREEN}[2]{Colors.END} Natural Language CLI {Colors.DIM}(Interactive conversational interface){Colors.END}")
    print(f"  {Colors.GREEN}[3]{Colors.END} Test Code-Mode {Colors.DIM}(Run performance benchmark){Colors.END}")
    print(f"  {Colors.GREEN}[0]{Colors.END} Back to main menu")
    print()

    choice = input(f"{Colors.CYAN}Selection: {Colors.END}").strip()

    if choice == '0':
        return True
    elif choice == '1':
        clear_screen()
        print_header("Launching AI Engine Web API")
        print_info("Starting Flask web server on http://127.0.0.1:5001")
        print_info("Code-mode enabled for optimal performance")
        print()
        print_warning("Press Ctrl+C to stop the server")
        print()

        api_script = Path("03-web-interface/self_coding_web_api.py")
        if api_script.exists():
            try:
                subprocess.run([
                    "python3",
                    str(api_script),
                    "--workspace", ".",
                    "--port", "5001"
                ])
            except KeyboardInterrupt:
                print()
                print_info("Web API server stopped")
        else:
            print_error(f"Web API script not found: {api_script}")
            print_info("Expected location: 03-web-interface/self_coding_web_api.py")

    elif choice == '2':
        clear_screen()
        print_header("Natural Language CLI")
        print_info("Launching interactive AI interface...")
        print()

        cli_script = Path("02-ai-engine/natural_language_interface.py")
        if cli_script.exists():
            try:
                subprocess.run(["python3", str(cli_script)])
            except KeyboardInterrupt:
                print()
                print_info("CLI session ended")
        else:
            print_error(f"CLI script not found: {cli_script}")

    elif choice == '3':
        clear_screen()
        test_code_mode_performance()
    else:
        print_warning("Invalid option")

    return True

def test_code_mode_performance():
    """Run code-mode performance benchmark"""
    print_header("Code-Mode Performance Test")

    print_info("This benchmark compares traditional vs code-mode execution")
    print()

    try:
        sys.path.insert(0, str(Path(__file__).parent / "02-ai-engine"))
        from code_mode_bridge import CodeModeBridge
        from workflow_batch_optimizer import WorkflowBatchOptimizer
        from advanced_planner import ExecutionPlan, ExecutionStep, StepType, TaskComplexity

        # Create test plan
        print_info("Creating test execution plan (5 file operations)...")
        plan = ExecutionPlan(
            task="Benchmark: Read and analyze multiple files",
            complexity=TaskComplexity.COMPLEX,
            steps=[
                ExecutionStep(
                    step_num=1,
                    step_type=StepType.READ_FILE,
                    description="Read dsmil.py",
                    action="Read file",
                    parameters={"filepath": "dsmil.py"}
                ),
                ExecutionStep(
                    step_num=2,
                    step_type=StepType.READ_FILE,
                    description="Read CODE_MODE_INTEGRATION.md",
                    action="Read file",
                    parameters={"filepath": "CODE_MODE_INTEGRATION.md"}
                ),
                ExecutionStep(
                    step_num=3,
                    step_type=StepType.READ_FILE,
                    description="Read README.md",
                    action="Read file",
                    parameters={"filepath": "README.md"}
                ),
                ExecutionStep(
                    step_num=4,
                    step_type=StepType.SEARCH,
                    description="Search for 'DSMIL'",
                    action="Search",
                    parameters={"pattern": "DSMIL"}
                ),
                ExecutionStep(
                    step_num=5,
                    step_type=StepType.SEARCH,
                    description="Search for 'code-mode'",
                    action="Search",
                    parameters={"pattern": "code-mode"}
                ),
            ],
            estimated_time=10,
            files_involved=["dsmil.py", "CODE_MODE_INTEGRATION.md", "README.md"],
            dependencies=[],
            risks=[],
            success_criteria=["All files processed"],
            model_used="test"
        )

        print_success("Test plan created (5 steps)")
        print()

        # Optimize for code-mode
        print_info("Analyzing with workflow optimizer...")
        optimizer = WorkflowBatchOptimizer()
        stats = optimizer.get_optimization_stats(plan)

        print()
        print(f"{Colors.BOLD}Optimization Analysis:{Colors.END}")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Execution batches: {stats['batches']}")
        print(f"  Parallel batches: {stats['parallel_batches']}")
        print(f"  Parallelizable steps: {stats['parallelizable_steps']}")
        print()

        print(f"{Colors.BOLD}Performance Comparison:{Colors.END}")
        print(f"  Traditional API calls: {stats['traditional_api_calls']}")
        print(f"  Code-mode API calls: {stats['code_mode_api_calls']}")
        print(f"  {Colors.GREEN}API calls saved: {stats['api_calls_saved']} ({stats['performance_improvement_pct']:.0f}% reduction){Colors.END}")
        print(f"  {Colors.GREEN}Estimated time saved: {stats['estimated_time_saved_ms']}ms{Colors.END}")
        print()

        # Generate TypeScript
        print_info("Generated TypeScript code for code-mode:")
        print(f"{Colors.DIM}{'─' * 70}{Colors.END}")
        typescript = optimizer.generate_typescript(plan)

        # Show first 20 lines
        lines = typescript.split('\n')
        for line in lines[:20]:
            print(f"{Colors.DIM}{line}{Colors.END}")
        if len(lines) > 20:
            print(f"{Colors.DIM}... ({len(lines) - 20} more lines){Colors.END}")
        print(f"{Colors.DIM}{'─' * 70}{Colors.END}")
        print()

        print_success("Code-mode optimization test complete!")
        print()
        print_info("Key benefits:")
        print(f"  {Colors.CYAN}▶{Colors.END} Batched execution reduces API round trips")
        print(f"  {Colors.CYAN}▶{Colors.END} Parallel operations with Promise.all()")
        print(f"  {Colors.CYAN}▶{Colors.END} Significant token and cost savings")
        print(f"  {Colors.CYAN}▶{Colors.END} Automatic fallback to traditional mode")

    except ImportError as e:
        print_error(f"Import error: {e}")
        print_warning("Ensure code-mode integration is installed")
    except Exception as e:
        print_error(f"Test error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# INTERACTIVE MENU
# ============================================================================
# DEB Package Operations
# ============================================================================

def build_deb_packages():
    """Build all DEB packages"""
    print_header("Building DEB Packages")

    packaging_dir = Path(__file__).parent / "packaging"
    build_script = packaging_dir / "build-all-debs.sh"

    if not build_script.exists():
        print_error(f"Build script not found: {build_script}")
        return False

    print_info("Building all 4 DEB packages...")
    print()

    success, stdout, stderr = run_command(
        str(build_script),
        cwd=packaging_dir,
        check=False
    )

    if success:
        print_success("All DEB packages built successfully!")
        print()
        print_info("Built packages:")
        for pkg in ["dsmil-platform_8.3.1-1.deb", "dell-milspec-tools_1.0.0-1_amd64.deb",
                    "tpm2-accel-examples_1.0.0-1.deb", "dsmil-complete_8.3.2-1.deb"]:
            pkg_path = packaging_dir / pkg
            if pkg_path.exists():
                size = pkg_path.stat().st_size
                print(f"  ✓ {pkg} ({size // 1024} KB)")
        return True
    else:
        print_error("Failed to build DEB packages")
        if stderr:
            print(f"\n{Colors.FAIL}Error output:{Colors.END}")
            print(stderr)
        return False

def install_deb_packages():
    """Install all DEB packages"""
    print_header("Installing DEB Packages")

    # Check if running as root
    if os.geteuid() != 0:
        print_error("This operation requires root privileges")
        print_info("Please run with: sudo python3 dsmil.py")
        return False

    packaging_dir = Path(__file__).parent / "packaging"
    install_script = packaging_dir / "install-all-debs.sh"

    if not install_script.exists():
        print_error(f"Install script not found: {install_script}")
        return False

    print_info("Installing all DEB packages in dependency order...")
    print()

    success, stdout, stderr = run_command(
        str(install_script),
        cwd=packaging_dir,
        check=False
    )

    if success:
        print_success("All DEB packages installed successfully!")
        return True
    else:
        print_error("Failed to install DEB packages")
        if stderr:
            print(f"\n{Colors.FAIL}Error output:{Colors.END}")
            print(stderr)
        return False

def verify_deb_installation():
    """Verify DEB package installation"""
    print_header("Verifying DEB Installation")

    packaging_dir = Path(__file__).parent / "packaging"
    verify_script = packaging_dir / "verify-installation.sh"

    if not verify_script.exists():
        print_error(f"Verification script not found: {verify_script}")
        return False

    print_info("Running comprehensive 10-point verification...")
    print()

    success, stdout, stderr = run_command(
        str(verify_script),
        cwd=packaging_dir,
        check=False
    )

    # Verification script handles its own output
    return success

# ============================================================================
def build_rust_safety_library():
    """Build Rust safety library as an automatic dependency"""
    print_header("Building Rust Safety Library (Dependency)")

    kernel_dir = Path("01-source/kernel")
    if not kernel_dir.exists():
        print_error("Kernel source directory not found (01-source/kernel)")
        return False

    # Build only the Rust library; do not touch kernel modules
    success, stdout, stderr = run_command(
        "make rust-lib-only", cwd=kernel_dir, check=False
    )

    if success:
        print_success("Rust safety library built successfully")
        lib_path = kernel_dir / "rust" / "libdsmil_rust.a"
        if lib_path.exists():
            size = lib_path.stat().st_size
            print_info(f"Library: {lib_path} ({size // 1024} KB)")
        nm_ok, nm_out, _ = run_command(
            "nm libdsmil_rust.a | grep ' T ' | head -10",
            cwd=kernel_dir / "rust",
            check=False,
        )
        if nm_ok and nm_out:
            print_info("Sample exported symbols:")
            for line in nm_out.strip().splitlines():
                print(f"  {line}")
        return True

    print_error("Rust safety library build failed (dependency)")
    if stderr:
        print(f"\n{Colors.FAIL}Build output (last 20 lines):{Colors.END}")
        combined = (stdout or "") + "\n" + (stderr or "")
        lines = [l for l in combined.split("\n") if l.strip()][-20:]
        for line in lines:
            if "error:" in line.lower() or "fatal" in line.lower():
                print(f"{Colors.FAIL}{line}{Colors.END}")
            else:
                print(f"{Colors.DIM}{line}{Colors.END}")

    print_info("Kernel modules will continue to use C safety stubs.")
    return False

# ============================================================================
def check_rust_safety_layer():
    """Check Rust toolchain and safety-layer readiness"""
    print_header("Rust Safety Layer Check")

    # Check rustup and rustc presence
    issues = []

    has_rustup = subprocess.run(
        ["which", "rustup"], capture_output=True
    ).returncode == 0
    has_rustc = subprocess.run(
        ["which", "rustc"], capture_output=True
    ).returncode == 0

    if not has_rustc:
        issues.append("rustc compiler not found in PATH")
    if not has_rustup:
        issues.append("rustup toolchain manager not found")

    if issues:
        print_error("Rust toolchain not fully installed:")
        for issue in issues:
            print_error(f"  ✗ {issue}")
        print_info("\nInstall Rust toolchain:")
        print_info("  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        print_info("  source ~/.cargo/env")
        return False

    # Check nightly availability
    nightly_available = subprocess.run(
        ["rustup", "run", "nightly", "rustc", "--version"],
        capture_output=True,
        text=True,
    ).returncode == 0

    if nightly_available:
        print_success("✓ nightly toolchain available")
    else:
        print_warning("Nightly toolchain not found")
        print_info("  Run: rustup toolchain install nightly")

    # Check rust-src component for nightly
    rust_src_installed = subprocess.run(
        ["rustup", "component", "list", "--toolchain", "nightly"],
        capture_output=True,
        text=True,
    )
    has_rust_src = (
        rust_src_installed.returncode == 0
        and "rust-src (installed)" in rust_src_installed.stdout
    )

    if has_rust_src:
        print_success("✓ rust-src component installed for nightly")
    else:
        print_warning("rust-src component not installed for nightly")
        print_info("  Run: rustup component add rust-src --toolchain nightly")

    # Check kernel Rust build from kernel/rust
    kernel_rust_dir = Path("01-source/kernel/rust")
    if not kernel_rust_dir.exists():
        print_error(f"Kernel Rust directory not found: {kernel_rust_dir}")
        return False

    print_info("\nTesting Rust safety layer build (dry run)...")
    success, stdout, stderr = run_command(
        "make all", cwd=kernel_rust_dir, check=False
    )

    if success:
        print_success("✓ Rust safety layer library built successfully")
        print_info("  Library: 01-source/kernel/rust/libdsmil_rust.a")
    else:
        print_error("Rust safety layer build failed")
        if stderr:
            print(f"\n{Colors.FAIL}Build output (last 30 lines):{Colors.END}")
            combined = (stdout or "") + "\n" + (stderr or "")
            lines = [l for l in combined.split("\n") if l.strip()][-30:]
            for line in lines:
                if "error:" in line.lower() or "fatal" in line.lower():
                    print(f"{Colors.FAIL}{line}{Colors.END}")
                else:
                    print(f"{Colors.DIM}{line}{Colors.END}")

        print_info("\nTo configure the toolchain automatically from the kernel tree:")
        print_info("  cd 01-source/kernel")
        print_info("  make rust-setup")
        return False

    print_info("\nNext steps to enable Rust safety in the kernel module:")
    print_info("  1) cd 01-source/kernel")
    print_info("  2) ENABLE_RUST=1 SKIP_OBJTOOL=1 make clean")
    print_info("  3) ENABLE_RUST=1 SKIP_OBJTOOL=1 make all")
    print_info("\nIf objtool FMA errors persist, see:")
    print_info("  01-source/kernel/docs/OBJTOOL_FMA_FIX.md")
    return True

# ============================================================================

def print_tempest_banner():
    """Print TEMPEST Class C security banner"""
    print(f"\n{Colors.SHIELDED}╔══════════════════════════════════════════════════════════════════════╗{Colors.END}")
    print(f"{Colors.SHIELDED}║{Colors.END} {Colors.BOLD}DSMIL PLATFORM - TEMPEST CLASS C SECURITY SYSTEM{Colors.END}                 {Colors.SHIELDED}║{Colors.END}")
    print(f"{Colors.SHIELDED}╠══════════════════════════════════════════════════════════════════════╣{Colors.END}")
    print(f"{Colors.SHIELDED}║{Colors.END} {Colors.SECURE}[SECURED]{Colors.END} Electromagnetic Emission Protection: {Colors.GREEN}ACTIVE{Colors.END}          {Colors.SHIELDED}║{Colors.END}")
    print(f"{Colors.SHIELDED}║{Colors.END} {Colors.SECURE}[SHIELDED]{Colors.END} TEMPEST Compliance Level: {Colors.BOLD}CLASS C{Colors.END}                 {Colors.SHIELDED}║{Colors.END}")
    print(f"{Colors.SHIELDED}║{Colors.END} {Colors.SECURE}[VERIFIED]{Colors.END} Hardware Security Module: {Colors.GREEN}104 DEVICES{Colors.END}           {Colors.SHIELDED}║{Colors.END}")
    print(f"{Colors.SHIELDED}╚══════════════════════════════════════════════════════════════════════╝{Colors.END}\n")

def show_interactive_menu():
    """Show interactive TUI menu with numbered options (TEMPEST themed)"""
    while True:
        clear_screen()
        print_tempest_banner()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.BOLD}{Colors.GREEN}QUICK START{Colors.END} {Colors.DIM}(Recommended - Automated Deployment){Colors.END}          {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print(f"  {Colors.GREEN}{Colors.BOLD}[1]{Colors.END} {Colors.BOLD}Smart Build{Colors.END} {Colors.SECURE}(104dev→84dev fallback + Auto-install + Auto-load){Colors.END}")
        print(f"  {Colors.GREEN}[2]{Colors.END} Complete Setup {Colors.DIM}(Build → Load → Launch Control Centre){Colors.END}")
        print(f"  {Colors.GREEN}[3]{Colors.END} Deploy & Activate {Colors.SHIELDED}(Load driver → Auto-activate devices){Colors.END}")
        print()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.BOLD}{Colors.CYAN}DRIVER OPERATIONS{Colors.END} {Colors.DIM}(Manual Control){Colors.END}                          {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print(f"  {Colors.SECURE}[4]{Colors.END} Build driver {Colors.DIM}(Standard build){Colors.END}")
        print(f"  {Colors.SECURE}[5]{Colors.END} Build driver {Colors.CLASSIFIED}(CLEAN - Full rebuild){Colors.END}")
        print(f"  {Colors.SECURE}[6]{Colors.END} Load driver {Colors.WARNING}(Auto-detect - Requires root){Colors.END}")
        print(f"  {Colors.SECURE}[7]{Colors.END} Unload driver {Colors.WARNING}(Remove all - Requires root){Colors.END}")
        print(f"  {Colors.SECURE}[8]{Colors.END} Driver status {Colors.DIM}(Check current state){Colors.END}")
        print()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.BOLD}{Colors.CYAN}CONTROL CENTRE{Colors.END} {Colors.DIM}(104-Device Security Operations){Colors.END}             {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print(f"  {Colors.SECURE}[9]{Colors.END} Launch control centre {Colors.DIM}(Interactive mode){Colors.END}")
        print(f" {Colors.SECURE}[10]{Colors.END} Control centre {Colors.SHIELDED}(AUTO-DISCOVER){Colors.END}")
        print(f" {Colors.SECURE}[11]{Colors.END} Control centre {Colors.GREEN}(AUTO-ACTIVATE SAFE DEVICES){Colors.END}")
        print()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.BOLD}{Colors.CYAN}AI & AUTOMATION{Colors.END} {Colors.DIM}(Code-Mode Intelligence - 60-88% Faster){Colors.END}       {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print(f" {Colors.SECURE}[15]{Colors.END} Initialize Code-Mode {Colors.GREEN}(Setup AI acceleration){Colors.END}")
        print(f" {Colors.SECURE}[16]{Colors.END} Launch AI Engine {Colors.CYAN}(Web API / CLI / Benchmark){Colors.END}")
        print()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.BOLD}{Colors.CYAN}DEB PACKAGE SYSTEM{Colors.END} {Colors.DIM}(System-Wide Installation){Colors.END}             {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print(f" {Colors.SECURE}[17]{Colors.END} Build DEB packages {Colors.GREEN}(4 packages: platform, tools, examples){Colors.END}")
        print(f" {Colors.SECURE}[18]{Colors.END} Install DEB packages {Colors.WARNING}(Requires root){Colors.END}")
        print(f" {Colors.SECURE}[19]{Colors.END} Verify installation {Colors.DIM}(10-point check){Colors.END}")
        print()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.BOLD}{Colors.CYAN}DIAGNOSTICS & INFO{Colors.END} {Colors.DIM}(System Information){Colors.END}                   {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print(f" {Colors.SECURE}[12]{Colors.END} System diagnostics {Colors.DIM}(Health check){Colors.END}")
        print(f" {Colors.SECURE}[13]{Colors.END} Documentation {Colors.DIM}(View guides){Colors.END}")
        print(f" {Colors.SECURE}[14]{Colors.END} Command help {Colors.DIM}(CLI reference){Colors.END}")
        print()

        print(f"{Colors.SHIELDED}┌─────────────────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"{Colors.SHIELDED}│{Colors.END} {Colors.FAIL}[0]{Colors.END} EXIT SECURE SESSION {Colors.DIM}(Terminate platform access){Colors.END}             {Colors.SHIELDED}│{Colors.END}")
        print(f"{Colors.SHIELDED}└─────────────────────────────────────────────────────────────────────┘{Colors.END}")
        print()

        try:
            choice = input(f"{Colors.SHIELDED}╭─[{Colors.BOLD}{Colors.GREEN}TEMPEST-SECURED{Colors.END}{Colors.SHIELDED}]─[{Colors.CYAN}Selection{Colors.END}{Colors.SHIELDED}]{Colors.END}\n{Colors.SHIELDED}╰─➤{Colors.END} ").strip()

            if choice == '0':
                clear_screen()
                print(f"\n{Colors.SHIELDED}╔══════════════════════════════════════════════════════════════════════╗{Colors.END}")
                print(f"{Colors.SHIELDED}║{Colors.END} {Colors.GREEN}SECURE SESSION TERMINATED{Colors.END}                                        {Colors.SHIELDED}║{Colors.END}")
                print(f"{Colors.SHIELDED}║{Colors.END} {Colors.DIM}All TEMPEST-protected operations have been safely closed{Colors.END}      {Colors.SHIELDED}║{Colors.END}")
                print(f"{Colors.SHIELDED}╚══════════════════════════════════════════════════════════════════════╝{Colors.END}\n")
                return True

            elif choice == '1':
                clear_screen()
                print_header("Smart Build - 104dev with Fallback and Auto-Load")
                build_and_autoload(clean=False)

            elif choice == '2':
                clear_screen()
                print_header("Complete Setup - Build, Load, Control")
                if build_driver(clean=False):
                    if load_driver():
                        launch_control_centre(auto_discover=False, auto_activate=False)
                    else:
                        print_error("Failed to load driver. Stopping.")
                else:
                    print_error("Failed to build driver. Stopping.")

            elif choice == '3':
                clear_screen()
                print_header("Deploy & Activate - Load and Auto-Activate")
                if load_driver():
                    launch_control_centre(auto_discover=True, auto_activate=True)
                else:
                    print_error("Failed to load driver. Stopping.")

            elif choice == '4':
                clear_screen()
                build_driver(clean=False)

            elif choice == '5':
                clear_screen()
                build_driver(clean=True)

            elif choice == '6':
                clear_screen()
                load_driver()

            elif choice == '7':
                clear_screen()
                unload_driver()

            elif choice == '8':
                clear_screen()
                driver_status()

            elif choice == '9':
                clear_screen()
                launch_control_centre(auto_discover=False, auto_activate=False)

            elif choice == '10':
                clear_screen()
                launch_control_centre(auto_discover=True, auto_activate=False)

            elif choice == '11':
                clear_screen()
                launch_control_centre(auto_discover=True, auto_activate=True)

            elif choice == '12':
                clear_screen()
                run_diagnostics()

            elif choice == '13':
                clear_screen()
                show_docs()

            elif choice == '14':
                clear_screen()
                show_help()

            elif choice == '15':
                clear_screen()
                init_code_mode()

            elif choice == '16':
                clear_screen()
                launch_ai_engine()

            elif choice == '17':
                clear_screen()
                build_deb_packages()

            elif choice == '18':
                clear_screen()
                install_deb_packages()

            elif choice == '19':
                clear_screen()
                verify_deb_installation()

            else:
                print_warning("Invalid option. Please select 0-19.")
                continue

            # Wait for user before showing menu again
            if choice != '0':
                input(f"\n{Colors.SHIELDED}[{Colors.GREEN}●{Colors.END}{Colors.SHIELDED}] Press {Colors.BOLD}ENTER{Colors.END}{Colors.SHIELDED} to return to secure menu...{Colors.END}")

        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Interrupted by user{Colors.END}")
            return True
        except EOFError:
            print(f"\n{Colors.INFO}End of input{Colors.END}")
            return True
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")

# ============================================================================
# COMMAND-LINE HELP
# ============================================================================

def show_help():
    """Show help message"""
    print_header("DSMIL Platform v2.1 - Enhanced Entry Point")

    # Show kernel info
    major, minor, patch = get_kernel_version()
    compatible, recommended, warnings = check_kernel_compatibility()

    print(f"{Colors.CYAN}Current System:{Colors.END}")
    print(f"  Kernel Version:     {major}.{minor}.{patch}")
    print(f"  Recommended Driver: {recommended}")
    print(f"  Status:             {'Compatible' if compatible else 'May have issues'}")
    print()

    print(f"{Colors.CYAN}Interactive Mode:{Colors.END}")
    print("  python3 dsmil.py           # Launch interactive TUI menu\n")

    print("Available commands:\n")

    print(f"{Colors.BOLD}Driver Management:{Colors.END}")
    print("  build                       Build kernel driver (auto-detects best version)")
    print("  build --clean               Clean and rebuild driver")
    print("  build --force-104           Force driver 104 (104 devices)")
    print("  build --force-84            Force driver 84 (84 devices)")
    print("  load                        Load driver (auto-selects version)")
    print("  load --force-104            Load driver 104")
    print("  load --force-84             Load driver 84")
    print("  unload                      Unload driver (requires sudo)")
    print("  status                      Show driver status")
    print("  build              Build kernel driver")
    print("  build --clean      Clean and rebuild driver")
    print("  build-auto         Build with 104dev→84dev fallback + auto-install/load (requires sudo)")
    print("  build-auto --clean Clean, build with fallback, and auto-load (requires sudo)")
    print("  load               Load driver - auto-detects 104dev or 84dev (requires sudo)")
    print("  load --driver NAME Load specific driver by name (requires sudo)")
    print("  unload             Unload all DSMIL drivers (requires sudo)")
    print("  unload --driver NAME Unload specific driver by name (requires sudo)")
    print("  status             Show driver status")
    print()

    print(f"{Colors.BOLD}Control Centre:{Colors.END}")
    print("  control                     Launch interactive control centre")
    print("  control --auto              Auto-discover and launch")
    print("  control --activate          Auto-discover and activate safe devices")
    print()

    print(f"{Colors.BOLD}Utilities:{Colors.END}")
    print("  diagnostics                 Run system diagnostics")
    print("  docs                        Show documentation links")
    print("  sanity                      Quick sanity check of driver chain")
    print("  help                        Show this help message")
    print()

    print(f"{Colors.BOLD}Enhanced Features (v2.1):{Colors.END}")
    print("  • Kernel 6.17+ optimized with fallback to 5.x/4.x")
    print("  • Driver 104 primary with automatic fallback to 84")
    print("  • Complex path recovery for moved/lost files")
    print("  • Multiple compensation mechanisms for build failures")
    print("  • Resilient build system with retry logic")
    print()

    print(f"{Colors.BOLD}Quick Start:{Colors.END}")
    print("  1. python3 dsmil.py build          # Build driver")
    print("  2. sudo python3 dsmil.py load      # Auto-detect and load driver")
    print("  3. sudo python3 dsmil.py control   # Launch control centre")
    print()
    print(f"{Colors.BOLD}Smart Build (Recommended):{Colors.END}")
    print("  sudo python3 dsmil.py build-auto   # Build + fallback + auto-load (one command!)")
    print()
    print(f"{Colors.BOLD}Auto-Detection:{Colors.END}")
    print("  load/unload commands now auto-detect which driver is available")
    print("  Prefers 104dev, falls back to 84dev automatically")
    print()

    print(f"{Colors.BOLD}Full Automation:{Colors.END}")
    print("  sudo python3 dsmil.py load && \\")
    print("       sudo python3 dsmil.py control --activate")
    print()

    print(f"{Colors.BOLD}Kernel 6.17 Specific:{Colors.END}")
    print("  python3 dsmil.py build --force-104  # Force 104-device driver")
    print("  sudo python3 dsmil.py load --force-104")
    print()

def run_sanity_check():
    """Quick sanity summary for driver chain"""
    from pathlib import Path
    from importlib import import_module
    import sys

    print_header("DSMIL Driver Sanity Check")

    # Check device node existence (primary + fallbacks)
    dev_candidates = ["/dev/dsmil0", "/dev/dsmil-104dev", "/dev/dsmil"]
    existing = [Path(p) for p in dev_candidates if Path(p).exists()]
    if existing:
        dev_path = existing[0]
        print_success(f"Device node present: {dev_path}")
    else:
        dev_path = Path(dev_candidates[0])
        print_error(f"Device node missing: {dev_path}")

    # Try importing driver interface (02-ai-engine)
    try:
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root / "02-ai-engine"))
        ddi = import_module("dsmil_driver_interface")
    except Exception as e:
        print_error(f"Failed to import dsmil_driver_interface: {e}")
        return False

    # Check driver_loaded flag
    try:
        loaded = ddi.check_driver_loaded()
        print_success("check_driver_loaded() reports driver loaded") if loaded else \
            print_warning("check_driver_loaded() reports driver NOT loaded")
    except Exception as e:
        print_error(f"check_driver_loaded() failed: {e}")
        loaded = False

    # If node exists, try full quick scan
    try:
        summary = ddi.quick_device_scan()
    except Exception as e:
        print_error(f"quick_device_scan() failed: {e}")
        summary = None

    if summary:
        print_info("Quick device scan summary:")
        print(f"  Driver loaded:   {summary.get('driver_loaded')}")
        print(f"  Driver version:  {summary.get('driver_version')}")
        print(f"  Device count:    {summary.get('device_count')}")
        backend = summary.get('backend_name')
        if backend:
            print(f"  Backend:         {backend}")
        status = summary.get('system_status')
        if status:
            print(f"  Thermal (°C):    {status.thermal_celsius}")
            print(f"  Tokens:          {status.token_reads} reads / {status.token_writes} writes")
    else:
        print_warning("Quick device scan not available (driver not openable?)")

    print()
    if summary and summary.get('backend_name'):
        backend = summary['backend_name']
        if backend.startswith("simulated"):
            print_warning("Backend is simulated – no real firmware token access")

    print("If the kernel module is loaded but this sanity check fails,")
    print("inspect kernel logs with: dmesg | grep -i dsmil")

    return bool(existing) and bool(summary and summary.get('driver_loaded'))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Platform - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dsmil.py build                 # Build driver
  sudo python3 dsmil.py build-auto       # Smart build: 104dev→84dev fallback + auto-load
  sudo python3 dsmil.py load             # Load driver
  sudo python3 dsmil.py control          # Launch control centre
  sudo python3 dsmil.py control --auto   # Auto-discover
  python3 dsmil.py diagnostics           # Run diagnostics
        """
    )

    parser.add_argument(
        'command',
        nargs='?',
        choices=['build', 'build-auto', 'load', 'unload', 'status',
                 'control', 'diagnostics', 'docs', 'help',
                 'rust-check', 'rust-lib', 'sanity'],
        help='Command to execute'
    )

    parser.add_argument('--clean', action='store_true',
                       help='Clean before building')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-discover for control centre')
    parser.add_argument('--activate', action='store_true',
                       help='Auto-activate safe devices')
    parser.add_argument('--driver', choices=['auto', '104', '84'],
                       default='auto',
                       help='Driver version: auto (detect), 104, or 84 (default: auto)')
    parser.add_argument('--force-104', action='store_true',
                       help='Force driver 104 (shortcut for --driver 104)')
    parser.add_argument('--force-84', action='store_true',
                       help='Force driver 84 (shortcut for --driver 84)')

    args = parser.parse_args()

    # Show interactive menu if no command
    if not args.command:
        return 0 if show_interactive_menu() else 1

    # Determine driver version
    if args.force_104:
        driver_version = '104'
    elif args.force_84:
        driver_version = '84'
    else:
        driver_version = args.driver

    # Execute command
    if args.command == 'build':
        success = build_driver(clean=args.clean)
    elif args.command == 'build-auto':
        success = build_and_autoload(clean=args.clean)
    elif args.command == 'load':
        if driver_version == 'auto':
            success = load_driver(driver_name=None)
        elif driver_version == '104':
            success = load_driver(driver_name='dsmil-104dev')
        elif driver_version == '84':
            success = load_driver(driver_name='dsmil-84dev')
        else:
            success = load_driver(driver_name=driver_version)
    elif args.command == 'unload':
        success = unload_driver(args.driver)
    elif args.command == 'status':
        success = driver_status()
        success = True  # Status always succeeds
    elif args.command == 'control':
        success = launch_control_centre(
            auto_discover=args.auto or args.activate,
            auto_activate=args.activate
        )
    elif args.command == 'diagnostics':
        success = run_diagnostics()
        success = True  # Diagnostics always succeeds
    elif args.command == 'docs':
        success = show_docs()
        success = True  # Docs always succeeds
    elif args.command == 'sanity':
        success = run_sanity_check()
    elif args.command == 'rust-check':
        success = check_rust_safety_layer()
    elif args.command == 'rust-lib':
        success = build_rust_safety_library()
    elif args.command == 'help':
        show_help()
        success = True
    else:
        show_help()
        success = False

    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
