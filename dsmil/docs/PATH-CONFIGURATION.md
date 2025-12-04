# Dynamic Path Configuration Guide

**DSLLVM v1.6.1+ Dynamic Path Resolution**

## Overview

DSLLVM supports flexible, portable installations through dynamic path resolution. All paths can be configured via environment variables, enabling:

- **Portable installations**: Install DSLLVM in non-standard locations
- **User-specific configurations**: Per-user config directories
- **Container-friendly**: Easy configuration for Docker/Kubernetes
- **Multi-tenant support**: Different paths for different users/tenants

---

## Environment Variables

### Core Path Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DSMIL_PREFIX` | `/opt/dsmil` | Base installation prefix |
| `DSMIL_CONFIG_DIR` | `${DSMIL_PREFIX}/etc` or `/etc/dsmil` | Configuration directory |
| `DSMIL_BIN_DIR` | `${DSMIL_PREFIX}/bin` | Binary executables directory |
| `DSMIL_LIB_DIR` | `${DSMIL_PREFIX}/lib` | Library files directory |
| `DSMIL_DATA_DIR` | `${DSMIL_PREFIX}/share` | Data files directory |
| `DSMIL_RUNTIME_DIR` | `${XDG_RUNTIME_DIR}/dsmil` or `/var/run/dsmil` | Runtime files directory |
| `DSMIL_TRUSTSTORE_DIR` | `${DSMIL_CONFIG_DIR}/truststore` | Trust store directory |
| `DSMIL_LOG_DIR` | `${DSMIL_PREFIX}/var/log` or `/var/log/dsmil` | Log files directory |
| `DSMIL_CACHE_DIR` | `${XDG_CACHE_HOME}/dsmil` or `$HOME/.cache/dsmil` | Cache directory |
| `DSMIL_TMP_DIR` | `${TMPDIR}` or `/tmp` | Temporary files directory |

### Fallback Behavior

Path resolution follows this priority order:

1. **Explicit environment variable** (e.g., `DSMIL_CONFIG_DIR`)
2. **Prefix-based path** (e.g., `${DSMIL_PREFIX}/etc`)
3. **System default** (e.g., `/etc/dsmil`)
4. **XDG Base Directory** (for user-specific paths)
5. **Standard system paths** (e.g., `/tmp`, `/var/log`)

---

## Runtime API

### C/C++ API

Include the path resolution header:

```c
#include <dsmil_paths.h>

int main(void) {
    /* Initialize path system (optional, auto-initializes on first use) */
    dsmil_paths_init();

    /* Get paths */
    const char *config_dir = dsmil_get_config_dir();
    const char *bin_dir = dsmil_get_bin_dir();
    const char *truststore = dsmil_get_truststore_dir();

    printf("Config: %s\n", config_dir);
    printf("Binaries: %s\n", bin_dir);
    printf("Truststore: %s\n", truststore);

    /* Resolve specific files */
    char config_path[PATH_MAX];
    if (dsmil_resolve_config("mission-profiles.json", config_path, sizeof(config_path))) {
        printf("Found config: %s\n", config_path);
    }

    /* Resolve binaries */
    char binary_path[PATH_MAX];
    if (dsmil_resolve_binary("dsmil-clang", binary_path, sizeof(binary_path))) {
        printf("Found binary: %s\n", binary_path);
    }

    /* Ensure directories exist */
    dsmil_ensure_dir(config_dir, 0755);

    dsmil_paths_cleanup();
    return 0;
}
```

### Path Resolution Functions

#### `dsmil_get_prefix()`
Returns the base installation prefix.

#### `dsmil_get_config_dir()`
Returns the configuration directory path.

#### `dsmil_get_bin_dir()`
Returns the binary directory path.

#### `dsmil_get_truststore_dir()`
Returns the truststore directory path.

#### `dsmil_resolve_config(filename, buffer, size)`
Searches for configuration files in:
1. `${DSMIL_CONFIG_DIR}`
2. `$HOME/.config/dsmil` or `${XDG_CONFIG_HOME}/dsmil`
3. `/etc/dsmil`

Returns the first found path, or NULL if not found.

#### `dsmil_resolve_binary(name, buffer, size)`
Searches for binaries in:
1. `${DSMIL_BIN_DIR}`
2. `PATH` environment variable
3. `/opt/dsmil/bin`

Returns the first found path, or NULL if not found.

#### `dsmil_path_exists(path)`
Checks if a path exists and is accessible.

#### `dsmil_ensure_dir(path, mode)`
Creates a directory and parent directories if they don't exist (like `mkdir -p`).

---

## Usage Examples

### Example 1: Portable Installation

Install DSLLVM in a custom location:

```bash
# Set prefix
export DSMIL_PREFIX=/home/user/dsmil

# Install
./configure --prefix=$DSMIL_PREFIX
make install

# Use
export PATH=$DSMIL_PREFIX/bin:$PATH
dsmil-clang --version
```

### Example 2: User-Specific Configuration

Use per-user configuration directory:

```bash
# Set user config directory
export XDG_CONFIG_HOME=$HOME/.config
export DSMIL_CONFIG_DIR=$XDG_CONFIG_HOME/dsmil

# Copy config
mkdir -p $DSMIL_CONFIG_DIR
cp mission-profiles.json $DSMIL_CONFIG_DIR/

# Compiler will find user config automatically
dsmil-clang -fdsmil-mission-profile=border_ops ...
```

### Example 3: Container Deployment

Configure paths for Docker/Kubernetes:

```dockerfile
FROM ubuntu:22.04

# Set DSLLVM paths
ENV DSMIL_PREFIX=/opt/dsmil
ENV DSMIL_CONFIG_DIR=/etc/dsmil
ENV DSMIL_BIN_DIR=/usr/local/bin
ENV DSMIL_LOG_DIR=/var/log/dsmil
ENV DSMIL_RUNTIME_DIR=/var/run/dsmil

# Install DSLLVM
COPY dsmil-install.tar.gz /
RUN tar -xzf /dsmil-install.tar.gz -C /opt/dsmil

# Use
CMD ["dsmil-clang", "--version"]
```

### Example 4: Multi-Tenant System

Different paths for different tenants:

```bash
# Tenant A
export DSMIL_PREFIX=/opt/dsmil/tenant-a
export DSMIL_CONFIG_DIR=/etc/dsmil/tenant-a
export DSMIL_LOG_DIR=/var/log/dsmil/tenant-a

# Tenant B
export DSMIL_PREFIX=/opt/dsmil/tenant-b
export DSMIL_CONFIG_DIR=/etc/dsmil/tenant-b
export DSMIL_LOG_DIR=/var/log/dsmil/tenant-b
```

### Example 5: Shell Scripts

Use environment variable substitution:

```bash
#!/bin/bash

# Use dynamic paths
CONFIG_DIR="${DSMIL_CONFIG_DIR:-/etc/dsmil}"
BIN_DIR="${DSMIL_BIN_DIR:-/opt/dsmil/bin}"
LOG_DIR="${DSMIL_LOG_DIR:-/var/log/dsmil}"

# Verify installation
if [ ! -f "$CONFIG_DIR/mission-profiles.json" ]; then
    echo "Error: Config not found in $CONFIG_DIR"
    exit 1
fi

# Run tool
"$BIN_DIR/dsmil-verify" --config "$CONFIG_DIR/mission-profiles.json" "$@"
```

---

## Integration with Build Systems

### CMake

```cmake
# Find DSLLVM paths
find_program(DSMIL_CLANG
    NAMES dsmil-clang
    PATHS
        $ENV{DSMIL_BIN_DIR}
        $ENV{DSMIL_PREFIX}/bin
        /opt/dsmil/bin
        /usr/local/bin
        /usr/bin
)

# Use in build
set(CMAKE_C_COMPILER ${DSMIL_CLANG})
```

### Makefile

```makefile
# Detect DSLLVM paths
DSMIL_PREFIX ?= $(shell echo $${DSMIL_PREFIX:-/opt/dsmil})
DSMIL_BIN_DIR ?= $(shell echo $${DSMIL_BIN_DIR:-$(DSMIL_PREFIX)/bin})
DSMIL_CONFIG_DIR ?= $(shell echo $${DSMIL_CONFIG_DIR:-/etc/dsmil})

# Use
CC := $(DSMIL_BIN_DIR)/dsmil-clang
CFLAGS += -fdsmil-mission-profile-config=$(DSMIL_CONFIG_DIR)/mission-profiles.json
```

---

## Migration Guide

### From Hardcoded Paths

**Before:**
```bash
dsmil-verify /usr/bin/llm_worker
cat /etc/dsmil/mission-profiles.json
```

**After:**
```bash
# Option 1: Use environment variables
dsmil-verify ${DSMIL_BIN_DIR:-/opt/dsmil/bin}/llm_worker
cat ${DSMIL_CONFIG_DIR:-/etc/dsmil}/mission-profiles.json

# Option 2: Use runtime API (in C code)
#include <dsmil_paths.h>
char binary_path[PATH_MAX];
dsmil_resolve_binary("llm_worker", binary_path, sizeof(binary_path));
dsmil_verify(binary_path);
```

---

## Best Practices

1. **Always use environment variables** in shell scripts and documentation
2. **Provide defaults** using `${VAR:-default}` syntax
3. **Use runtime API** in C/C++ code for maximum portability
4. **Respect XDG Base Directory** for user-specific paths
5. **Document path requirements** in your application's README
6. **Test with custom paths** to ensure portability

---

## Troubleshooting

### Path Not Found

```bash
# Check current paths
dsmil-clang --print-paths

# Or use runtime API
#include <dsmil_paths.h>
printf("Config: %s\n", dsmil_get_config_dir());
printf("Bin: %s\n", dsmil_get_bin_dir());
```

### Permission Denied

```bash
# Ensure directories exist and have correct permissions
sudo mkdir -p ${DSMIL_CONFIG_DIR:-/etc/dsmil}
sudo chmod 755 ${DSMIL_CONFIG_DIR:-/etc/dsmil}
```

### Multiple Config Files

The `dsmil_resolve_config()` function searches multiple locations. To see which one is used:

```c
char path[PATH_MAX];
if (dsmil_resolve_config("mission-profiles.json", path, sizeof(path))) {
    printf("Using config: %s\n", path);
}
```

---

## Related Documentation

- **[ATTRIBUTES.md](ATTRIBUTES.md)**: Source-level attributes reference
- **[MISSION-PROFILES-GUIDE.md](MISSION-PROFILES-GUIDE.md)**: Mission profile configuration
- **[PROVENANCE-CNSA2.md](PROVENANCE-CNSA2.md)**: Provenance and trust store paths
- **[DSLLVM-DESIGN.md](DSLLVM-DESIGN.md)**: Complete design specification

---

**DSLLVM Dynamic Path Configuration**: Enabling portable, flexible deployments for military software systems.
