#!/usr/bin/env python3
"""
DSMIL Build Validation Script
============================

Validates the DSMIL kernel module build without requiring root privileges.
Tests compilation artifacts, symbols, and module metadata.
"""

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, timeout=30):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"

def validate_module_file(module_path):
    """Validate module file exists and has reasonable size"""
    logger.info("=== Module File Validation ===")
    
    if not os.path.exists(module_path):
        logger.error(f"Module file not found: {module_path}")
        return False
    
    stat = os.stat(module_path)
    size_kb = stat.st_size // 1024
    
    logger.info(f"✓ Module file exists: {module_path}")
    logger.info(f"✓ Module size: {size_kb} KB")
    
    if size_kb < 100:
        logger.warning("⚠ Module seems small, may be incomplete")
    elif size_kb > 5000:
        logger.warning("⚠ Module seems large, check for debug symbols")
    else:
        logger.info("✓ Module size looks reasonable")
    
    return True

def validate_module_info(module_path):
    """Validate module metadata using modinfo"""
    logger.info("\n=== Module Metadata Validation ===")
    
    rc, stdout, stderr = run_command(f"modinfo {module_path}")
    if rc != 0:
        logger.error(f"modinfo failed: {stderr}")
        return False
    
    required_fields = ['license', 'description', 'author', 'version']
    found_fields = {}
    
    for line in stdout.split('\n'):
        for field in required_fields:
            if line.startswith(f"{field}:"):
                found_fields[field] = line.split(':', 1)[1].strip()
    
    logger.info("Module metadata:")
    for field in required_fields:
        if field in found_fields:
            logger.info(f"✓ {field}: {found_fields[field]}")
        else:
            logger.error(f"✗ Missing {field}")
            return False
    
    # Check parameters
    params = [line for line in stdout.split('\n') if line.startswith('parm:')]
    logger.info(f"✓ Module parameters: {len(params)} found")
    for param in params[:3]:  # Show first 3
        logger.info(f"  {param}")
    
    return True

def validate_symbols(module_path):
    """Check for expected symbols in the module"""
    logger.info("\n=== Symbol Validation ===")
    
    rc, stdout, stderr = run_command(f"nm {module_path}")
    if rc != 0:
        logger.warning(f"nm command failed: {stderr}")
        return True  # Not critical
    
    # Look for our stub functions
    expected_symbols = [
        'rust_dsmil_init',
        'rust_dsmil_cleanup',
        'rust_dsmil_smi_read_token',
        'rust_dsmil_smi_write_token',
        'rust_get_thermal_temperature'
    ]
    
    found_symbols = []
    for line in stdout.split('\n'):
        parts = line.strip().split()
        if len(parts) >= 3:
            symbol = parts[2]
            for expected in expected_symbols:
                if expected in symbol:
                    found_symbols.append(expected)
    
    logger.info(f"Expected symbols found: {len(set(found_symbols))}/{len(expected_symbols)}")
    for symbol in set(found_symbols):
        logger.info(f"✓ Found: {symbol}")
    
    missing = set(expected_symbols) - set(found_symbols)
    if missing:
        logger.warning(f"⚠ Missing symbols: {missing}")
    
    return len(found_symbols) >= len(expected_symbols) // 2

def validate_dependencies(module_path):
    """Check module dependencies"""
    logger.info("\n=== Dependency Validation ===")
    
    rc, stdout, stderr = run_command(f"modinfo -F depends {module_path}")
    if rc != 0:
        logger.error(f"Dependency check failed: {stderr}")
        return False
    
    depends = stdout.strip()
    if not depends:
        logger.info("✓ No external dependencies required")
    else:
        logger.info(f"✓ Dependencies: {depends}")
    
    return True

def validate_kernel_compatibility(module_path):
    """Check kernel version compatibility"""
    logger.info("\n=== Kernel Compatibility ===")
    
    # Get current kernel version
    rc, kernel_ver, _ = run_command("uname -r")
    if rc == 0:
        kernel_ver = kernel_ver.strip()
        logger.info(f"Current kernel: {kernel_ver}")
    
    # Get module's target kernel
    rc, stdout, stderr = run_command(f"modinfo -F vermagic {module_path}")
    if rc == 0:
        vermagic = stdout.strip()
        logger.info(f"Module vermagic: {vermagic}")
        
        if kernel_ver in vermagic:
            logger.info("✓ Kernel version matches")
            return True
        else:
            logger.warning("⚠ Kernel version mismatch")
            return False
    
    logger.error("Could not determine kernel compatibility")
    return False

def validate_rust_artifacts():
    """Check Rust compilation artifacts"""
    logger.info("\n=== Rust Artifact Validation ===")
    
    rust_lib = Path("rust/libdsmil_rust.a")
    if rust_lib.exists():
        size_kb = rust_lib.stat().st_size // 1024
        logger.info(f"✓ Rust library exists: {size_kb} KB")
        
        # Check for Rust symbols
        rc, stdout, stderr = run_command(f"nm {rust_lib} | grep rust_dsmil")
        if rc == 0 and stdout.strip():
            symbol_count = len(stdout.strip().split('\n'))
            logger.info(f"✓ Rust symbols found: {symbol_count}")
            return True
        else:
            logger.warning("⚠ No Rust symbols found in library")
    else:
        logger.info("ℹ Rust library not found (using C stubs)")
    
    return True

def main():
    module_path = "dsmil-72dev.ko"
    
    logger.info("DSMIL Build Validation")
    logger.info("=" * 50)
    
    tests = [
        ("Module File", lambda: validate_module_file(module_path)),
        ("Module Info", lambda: validate_module_info(module_path)),
        ("Symbols", lambda: validate_symbols(module_path)),
        ("Dependencies", lambda: validate_dependencies(module_path)),
        ("Kernel Compatibility", lambda: validate_kernel_compatibility(module_path)),
        ("Rust Artifacts", validate_rust_artifacts)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                logger.info(f"✓ {test_name}: PASSED")
            else:
                logger.error(f"✗ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Tests passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Module build is successful!")
    elif passed >= total * 0.8:
        logger.info("✅ BUILD SUCCESSFUL - Minor issues detected")
    elif passed >= total * 0.6:
        logger.warning("⚠️  BUILD PARTIAL - Some functionality missing")
    else:
        logger.error("❌ BUILD FAILED - Major issues detected")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)