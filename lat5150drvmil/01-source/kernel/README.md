# DSMIL Kernel Module - Reorganized Structure

The DSMIL kernel module has been reorganized into a clean, modular structure for better maintainability and development.

## Directory Structure

```
kernel/
├── core/           - Core driver and Hardware Abstraction Layer (HAL)
├── security/       - Security modules (access control, audit, MFA)
├── safety/         - Safety modules (Rust integration)
├── debug/          - Debug utilities and logging
├── enhanced/       - Enhanced features (threat engine, incident response)
├── rust/           - Rust safety layer
├── build/          - Build artifacts, Makefiles, and test scripts
├── docs/           - Documentation and guides
├── scripts/        - Build and maintenance scripts
├── Makefile        - Top-level Makefile (uses organized structure)
└── README.md       - This file
```

## Core Components

### core/
**Main driver and HAL**
- `dsmil-104dev.c` - Latest driver (104 devices + 3 BIOS interfaces)
- `dsmil_driver_module.c` - Driver entry point and initialization
- `dsmil_hal.c/h` - Hardware Abstraction Layer
- `dsmil_token_map.h` - SMBIOS token mappings
- `military_device_interface.h` - Military-grade device interface

**Note:** Deprecated drivers (dsmil-72dev, dsmil-simple) have been removed. Current production drivers are dsmil-84dev and dsmil-104dev.

### security/
**Security and compliance modules**
- `dsmil_access_control.c` - Role-based access control
- `dsmil_authorization.c` - Authorization framework
- `dsmil_audit_framework.c` - Comprehensive audit logging
- `dsmil_compliance.c` - Compliance enforcement (NIST, FIPS)
- `dsmil_mfa_auth.c` - Multi-factor authentication
- `dsmil_security_types.h` - Security type definitions

### safety/
**Memory safety and Rust integration**
- `dsmil_safety.c` - Memory safety checks
- `dsmil_rust_safety.c/h` - Rust FFI bindings
- `rust_stubs.c` - Fallback stubs when Rust is disabled
- `rust-link.c` - Rust linkage helpers

### debug/
**Debugging and diagnostics**
- `dsmil_debug.c/h` - Debug utilities and logging framework

### enhanced/
**Advanced security features**
- `dsmil_enhanced.c` - Enhanced driver features
- `dsmil_threat_engine.c` - Real-time threat detection
- `dsmil_incident_response.c` - Automated incident response
- `dell-smbios-token-enum.c` - SMBIOS token enumeration

### rust/
**Rust safety layer** (separate Cargo workspace)
- Memory-safe implementations of critical operations
- Zero-cost abstractions for kernel operations
- Formal verification compatible code

### build/
**Build system and artifacts**
- `Makefile*` - Various Makefile configurations
- `*.sh` - Build and installation scripts
- `test_module.py` - Module testing
- `validate_build.py` - Build validation

### docs/
**Documentation**
- `BUILD_DOCUMENTATION.md` - Build instructions
- `MODULE_RELOAD_GUIDE.md` - Module reload procedures
- `OBJTOOL_FMA_FIX.md` - FMA instruction fixes
- `CHUNKED_MEMORY_IMPLEMENTATION.md` - Memory management
- `DSMIL_SMI_INTEGRATION_SUMMARY.md` - SMI integration
- And more...

## Building

### Quick Build
```bash
make                    # Build with Rust (default)
make ENABLE_RUST=0      # Build without Rust
```

### Installation
```bash
sudo make install          # Install module
sudo modprobe dsmil-84dev  # Load 84-device driver (production)
sudo modprobe dsmil-104dev # Load 104-device driver (latest)
```

### Rust Development
```bash
make rust-check         # Check Rust code
make rust-test          # Run Rust tests
make rust-docs          # Generate Rust docs
make debug              # Build with debug info
```

### Information
```bash
make info               # Show build information
make structure          # Show directory structure
make help               # Show all targets
```

## Module Information

**Module Name**: dsmil-84dev
**License**: GPL
**Author**: LAT5150DRVMIL Development Team
**Description**: Dell SMBIOS Military-Grade Security Module

## Features

- **Hardware Abstraction Layer**: Unified interface for SMBIOS operations
- **Security Framework**: Role-based access control, audit logging, MFA
- **Memory Safety**: Rust integration for critical paths
- **Threat Detection**: Real-time monitoring and automated response
- **Compliance**: NIST 800-53, FIPS 140-2, Common Criteria
- **Debug Support**: Comprehensive logging and diagnostics

## System Requirements

- Linux kernel 4.4 or later
- Dell hardware with SMBIOS support
- Rust toolchain (optional, for Rust features)
- AVX-512 CPU (optional, for AVX enabler)

## Integration with AI Engine

The kernel module integrates with the AI Engine at `../../02-ai-engine/`:

- **Hardware Discovery**: `dsmil_ml_discovery.py` uses HAL for device detection
- **Device Activation**: `dsmil_integrated_activation.py` controls module operations
- **System Monitoring**: `dsmil_subsystem_controller.py` monitors kernel state

Access from Python:
```python
from ai_engine import DSMILIntegratedActivation, DSMILHardwareAnalyzer

# Activate hardware
activator = DSMILIntegratedActivation()
activator.run_full_workflow()

# Analyze system
analyzer = DSMILHardwareAnalyzer()
hw_info = analyzer.get_system_info()
```

## Development Guidelines

### Adding New Features

1. Choose the appropriate directory:
   - Core functionality → `core/`
   - Security features → `security/`
   - Safety-critical code → `safety/` (consider Rust)
   - Debug/diagnostics → `debug/`
   - Advanced features → `enhanced/`

2. Follow naming convention: `dsmil_<component>_<feature>.c`

3. Update Makefile object lists if needed

4. Add documentation to `docs/`

5. Write tests in `build/`

### Code Style

- Kernel coding style (see `Documentation/process/coding-style.rst`)
- Use HAL abstractions for hardware access
- Add proper error handling and logging
- Document security implications
- Consider memory safety (use Rust for critical paths)

## Troubleshooting

### Build Issues
```bash
make clean              # Clean build artifacts
make info               # Check configuration
dmesg | grep dsmil      # Check kernel logs
```

### Module Loading Issues
```bash
sudo modprobe -r dsmil-84dev    # Remove module
sudo modprobe dsmil-84dev       # Reload module
lsmod | grep dsmil              # Check if loaded
```

### Rust Issues
```bash
make rust-check         # Validate Rust code
make ENABLE_RUST=0      # Build without Rust
```

## Related Documentation

- **Drivers**: `../drivers/README.md` - Additional kernel drivers
- **AI Engine**: `../../02-ai-engine/README.md` - AI integration
- **Hardware**: `../../02-ai-engine/hardware/` - Hardware accelerators

## Migration from Old Structure

The reorganization maintains backward compatibility:

- All source files are preserved
- Module name unchanged (dsmil-84dev)
- API/ABI unchanged
- Makefile targets unchanged

Old locations are now organized:
- `*.c` files → appropriate subdirectories
- `Makefile*` → `build/`
- `*.md` → `docs/`

## Contributing

When contributing:

1. Follow the directory structure
2. Update documentation
3. Test thoroughly
4. Consider security implications
5. Add Rust implementations for safety-critical code

## License

GPL v2 - See individual source files for details

## Contact

LAT5150DRVMIL Development Team
For issues: See repository issue tracker
