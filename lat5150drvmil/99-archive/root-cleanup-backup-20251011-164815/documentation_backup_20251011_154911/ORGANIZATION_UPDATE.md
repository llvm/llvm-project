# Repository Organization Update

## Changes Made
- Added clean development environment: `LAT5150_DEV/`
- Added production deployment package: `LAT5150_PROD/`
- Consolidated all scattered components
- Eliminated duplicate directories (freed 20GB+)

## New Structure
```
LAT5150DRVMIL/
├── LAT5150_DEV/       # Clean development environment
│   ├── src/           # Kernel modules, TPM integration, userspace tools
│   ├── docs/          # Complete documentation tree
│   └── [dev structure]
├── LAT5150_PROD/      # Production deployment package
│   ├── bin/           # Compiled modules (dsmil-72dev.ko, dell-milspec.ko)
│   ├── lib/           # Military device libraries
│   ├── config/        # Production configuration
│   └── install.sh     # Automated installer
└── [original files]   # Preserved original structure
```

## TPM Integration
Both environments include comprehensive TPM module support:
- Military-grade TPM drivers
- NSA security hardened components
- Quantum-resistant implementations
- UEFI integration
- Intel ME bypass capabilities

## Usage
- **Development:** Use `LAT5150_DEV/` for all development work
- **Deployment:** Use `LAT5150_PROD/` for production installations
- **Reference:** Original files remain for historical reference

This organization provides clear separation between development and production while maintaining full project history.