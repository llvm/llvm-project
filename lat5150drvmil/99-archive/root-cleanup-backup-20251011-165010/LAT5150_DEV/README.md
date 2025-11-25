# LAT5150 DSMIL Development Environment

## Directory Structure
```
LAT5150_DEV/
├── src/           # Source code (kernel modules, userspace tools)
├── docs/          # Documentation
├── tools/         # Development tools and utilities
├── tests/         # Test suites
├── scripts/       # Build and deployment scripts
└── config/        # Configuration files
```

## Components
- DSMIL kernel modules
- TPM integration
- Security frameworks
- Hardware abstraction layer
- Monitoring tools

## Build Instructions
See individual component README files in src/ subdirectories.

## TPM Module
TPM integration files are included for secure hardware operations.
See tpm_cmake.txt for build configuration.