# DSMIL Kernel Drivers

This directory contains user-space and kernel-space drivers for the DSMIL platform.

## Directory Structure

```
drivers/
├── dsmil_avx512_enabler/    - AVX-512 CPU feature enabler
└── README.md                 - This file
```

## Driver Components

### dsmil_avx512_enabler
**Purpose**: Enables AVX-512 CPU instructions for advanced vector processing

**Type**: Kernel module

**Location**: `dsmil_avx512_enabler/`

**Build**: See `dsmil_avx512_enabler/README.md` for build instructions

## Future Driver Additions

This directory is structured to accommodate additional drivers:

- TPM drivers (Trusted Platform Module integration)
- Hardware accelerator drivers (NPU, GNA, NCS2)
- Security co-processor drivers
- Custom I/O drivers

## Building Drivers

Each driver subdirectory contains its own Makefile and README with specific build instructions.

General pattern:
```bash
cd dsmil_avx512_enabler/
make
sudo make install
```

## Integration with Kernel Module

The main DSMIL kernel module (in `../kernel/`) integrates with these drivers through the Hardware Abstraction Layer (HAL) in `../kernel/core/dsmil_hal.c`.

## Development Guidelines

When adding new drivers:

1. Create a new subdirectory under `drivers/`
2. Include a README.md with:
   - Driver purpose and description
   - Hardware requirements
   - Build instructions
   - Usage examples
3. Provide a Makefile for building
4. Update this README to list the new driver
5. Ensure proper integration with the main kernel module

## Related Documentation

- Main kernel module: `../kernel/README.md`
- HAL documentation: `../kernel/docs/BUILD_DOCUMENTATION.md`
- Hardware integration: `../../02-ai-engine/hardware/`
