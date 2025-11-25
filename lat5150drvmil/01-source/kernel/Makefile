# DSMIL Kernel Module Makefile (Reorganized Structure)
# =====================================================

CC ?= gcc
LD_CMD := $(shell $(CC) -print-prog-name=ld)
OBJCOPY ?= $(shell $(CC) -print-prog-name=objcopy)

export SKIP_OBJTOOL := 1
export CONFIG_STACK_VALIDATION := n
OBJTOOL_WRAPPER := $(CURDIR)/scripts/skip-objtool.sh
export OBJTOOL := $(OBJTOOL_WRAPPER)

# Directory structure
CORE_DIR := core
SECURITY_DIR := security
SAFETY_DIR := safety
DEBUG_DIR := debug
ENHANCED_DIR := enhanced
RUST_DIR := rust
BUILD_DIR := build

# Rust library integration
RUST_LIB := $(RUST_DIR)/libdsmil_rust.a
RUST_LIB_OBJ := $(RUST_DIR)/libdsmil_rust.o
RUST_BUILTINS_STUBS := $(RUST_DIR)/compiler_builtins_stubs.o

ENABLE_RUST ?= 1

# Module definition
# Allow building each module independently via ONLY_104DEV / ONLY_84DEV,
# or both together when neither flag is set.
ifdef ONLY_104DEV
obj-m := dsmil-104dev.o
else ifdef ONLY_84DEV
obj-m := dsmil-84dev.o
else
obj-m += dsmil-84dev.o
obj-m += dsmil-104dev.o
endif

# Add include paths for reorganized source directories
ccflags-y := -I$(src)/$(CORE_DIR) \
             -I$(src)/$(SECURITY_DIR) \
             -I$(src)/$(SAFETY_DIR) \
             -I$(src)/$(DEBUG_DIR) \
             -I$(src)/$(ENHANCED_DIR)

# Object files from organized directories
# Note: The driver uses #include to pull in sources, but we list them here for reference
CORE_OBJS := $(CORE_DIR)/dsmil_driver_module.o $(CORE_DIR)/dsmil_hal.o
SECURITY_OBJS := $(SECURITY_DIR)/dsmil_access_control.o $(SECURITY_DIR)/dsmil_authorization.o \
                 $(SECURITY_DIR)/dsmil_audit_framework.o $(SECURITY_DIR)/dsmil_compliance.o \
                 $(SECURITY_DIR)/dsmil_mfa_auth.o
SAFETY_OBJS := $(SAFETY_DIR)/dsmil_safety.o $(SAFETY_DIR)/dsmil_rust_safety.o
DEBUG_OBJS := $(DEBUG_DIR)/dsmil_debug.o
ENHANCED_OBJS := $(ENHANCED_DIR)/dsmil_enhanced.o $(ENHANCED_DIR)/dsmil_threat_engine.o \
                 $(ENHANCED_DIR)/dsmil_incident_response.o

# Build configuration: driver_module includes other .c files directly via #include
# This matches the original build approach where enhanced.c is included into hal.c
# Note: security/dsmil_mfa_auth.o is required as HAL calls dsmil_mfa_authorize_operation()

# Debug: Show ENABLE_RUST value during kbuild evaluation
$(info KBUILD: ENABLE_RUST=$(ENABLE_RUST) KBUILD_SRC=$(KBUILD_SRC) KERNELRELEASE=$(KERNELRELEASE))

# Kernel Rust safety currently disabled for modules; use C safety stubs.
$(info KBUILD: Kernel Rust safety layer disabled (using C stubs for all modules))

dsmil-84dev-objs := $(CORE_DIR)/dsmil_driver_module.o \
                    $(SECURITY_DIR)/dsmil_mfa_auth.o \
                    $(SAFETY_DIR)/rust_stubs.o
OBJECT_FILES_NON_STANDARD_dsmil-84dev.o := y
OBJECT_FILES_NON_STANDARD_$(CORE_DIR)/dsmil_driver_module.o := y
OBJECT_FILES_NON_STANDARD_$(SECURITY_DIR)/dsmil_mfa_auth.o := y
OBJECT_FILES_NON_STANDARD_$(SAFETY_DIR)/rust_stubs.o := y

# Expanded architecture driver (104 devices + 3 BIOS) without Rust safety
dsmil-104dev-objs := $(CORE_DIR)/dsmil-104dev.o
OBJECT_FILES_NON_STANDARD_dsmil-104dev.o := y
OBJECT_FILES_NON_STANDARD_$(CORE_DIR)/dsmil-104dev.o := y

$(info KBUILD: dsmil-84dev-objs = $(dsmil-84dev-objs))
$(info KBUILD: dsmil-104dev-objs = $(dsmil-104dev-objs))

# Note: OBJECT_FILES_NON_STANDARD and SKIP_STACK_VALIDATION are set per-object above
# DO NOT set them globally here as it may cause kbuild to skip compilation

# Kernel build directory
KDIR := /lib/modules/$(shell uname -r)/build

# ============================================================================
# Standalone build targets (only execute when NOT called from kbuild)
# ============================================================================
ifeq ($(KERNELRELEASE),)
# This section only runs when Makefile is called directly (standalone),
# NOT when called from kbuild

.PHONY: all modules clean install rust-lib

ifeq ($(ENABLE_RUST),1)
all: rust-lib $(RUST_LIB_OBJ)
	@echo "Building modules with Rust (ENABLE_RUST=$(ENABLE_RUST))"
	$(MAKE) -C $(KDIR) M=$(CURDIR) ENABLE_RUST=1 OBJTOOL=$(OBJTOOL_WRAPPER) SKIP_OBJTOOL=1 \
		CONFIG_STACK_VALIDATION=n CONFIG_OBJTOOL=n CONFIG_UNWINDER_ORC=n modules
else
all:
	@echo "Building modules without Rust (ENABLE_RUST=$(ENABLE_RUST))"
	$(MAKE) -C $(KDIR) M=$(CURDIR) ENABLE_RUST=0 OBJTOOL=$(OBJTOOL_WRAPPER) SKIP_OBJTOOL=1 \
		CONFIG_STACK_VALIDATION=n CONFIG_OBJTOOL=n CONFIG_UNWINDER_ORC=n modules
endif

# Build Rust library first
rust-lib:
ifeq ($(ENABLE_RUST),1)
	@echo "Building Rust safety layer..."
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust all
	@echo "Rust library built successfully"
else
	@echo "Rust safety layer disabled; skipping Rust build"
endif

check-rust:
	@if [ "$(ENABLE_RUST)" = "1" ] && [ ! -f $(RUST_LIB) ]; then \
		echo "Building missing Rust library..."; \
		$(MAKE) rust-lib; \
	fi

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust clean
	$(RM) $(RUST_LIB_OBJ) $(RUST_BUILTINS_STUBS) $(RUST_DIR)/.libdsmil_rust.o.cmd

# Install the module
install: all
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules_install
	depmod -a

# Rust-specific targets
rust-check:
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust check

rust-test:
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust test

rust-docs:
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust docs

# Debug build with Rust debug info
debug: rust-lib-debug
	$(MAKE) -C $(KDIR) M=$(PWD) modules

rust-lib-debug:
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust debug

# Show information about the build
info:
	@echo "DSMIL Kernel Module Build Information (Reorganized Structure)"
	@echo "============================================================="
	@echo "Directory Structure:"
	@echo "  Core:     $(CORE_DIR)/"
	@echo "  Security: $(SECURITY_DIR)/"
	@echo "  Safety:   $(SAFETY_DIR)/"
	@echo "  Debug:    $(DEBUG_DIR)/"
	@echo "  Enhanced: $(ENHANCED_DIR)/"
	@echo "  Rust:     $(RUST_DIR)/"
	@echo "  Build:    $(BUILD_DIR)/"
	@echo ""
	@echo "Rust Library: $(RUST_LIB)"
	@echo "Library exists: $(shell [ -f $(RUST_LIB) ] && echo 'Yes' || echo 'No')"
	@echo "Library size: $(shell [ -f $(RUST_LIB) ] && stat -c%s $(RUST_LIB) || echo '0') bytes"
	@echo "Kernel Dir: $(KDIR)"
	@echo "Build Dir: $(PWD)"
	@echo ""
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust info 2>/dev/null || true

# Structure information
structure:
	@echo "DSMIL Kernel Module Directory Structure"
	@echo "========================================"
	@echo ""
	@echo "$(CORE_DIR)/         - Core driver and HAL"
	@echo "$(SECURITY_DIR)/     - Security modules (access control, audit, MFA)"
	@echo "$(SAFETY_DIR)/       - Safety modules (Rust integration)"
	@echo "$(DEBUG_DIR)/        - Debug utilities"
	@echo "$(ENHANCED_DIR)/     - Enhanced features (threat engine, incident response)"
	@echo "$(RUST_DIR)/         - Rust safety layer"
	@echo "$(BUILD_DIR)/        - Build system and artifacts"
	@echo "docs/        - Documentation"
	@echo "scripts/     - Build scripts"
	@echo ""

# Help target
help:
	@echo "DSMIL Kernel Module Makefile (Reorganized Structure)"
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Build kernel module with Rust integration (default)"
	@echo "  clean        - Clean all build artifacts"
	@echo "  install      - Install kernel module"
	@echo "  info         - Show build information"
	@echo "  structure    - Show directory structure"
	@echo ""
	@echo "Rust-specific targets:"
	@echo "  rust-lib     - Build only the Rust library"
	@echo "  rust-check   - Check Rust code without building"
	@echo "  rust-test    - Run Rust unit tests"
	@echo "  rust-docs    - Generate Rust documentation"
	@echo "  debug        - Build with debug information"
	@echo ""
	@echo "Directory structure:"
	@echo "  core/        - Core driver and HAL"
	@echo "  security/    - Security modules"
	@echo "  safety/      - Safety modules"
	@echo "  debug/       - Debug utilities"
	@echo "  enhanced/    - Enhanced features"
	@echo "  rust/        - Rust safety layer"
	@echo "  build/       - Build system"
	@echo "  docs/        - Documentation"
	@echo ""
	@echo "For more Rust build options, see: make -C rust -f Makefile.rust help"

# End of standalone targets section
endif
# ============================================================================
# End of ifeq ($(KERNELRELEASE),) block
# When kbuild calls this Makefile, it only sees the obj-m and module-objs
# definitions at the top, NOT the standalone build targets above
# ============================================================================

# ============================================================================
# Dependency rules for Rust objects (must be OUTSIDE ifeq block)
# These rules must be visible to kbuild when it builds the modules
# ============================================================================

$(RUST_BUILTINS_STUBS): $(RUST_DIR)/compiler_builtins_stubs.c
	@echo "Compiling compiler builtins stubs..."
	$(CC) -c -o $@ $<

$(RUST_LIB):
ifeq ($(ENABLE_RUST),1)
	@echo "KBUILD: Building Rust library ($(RUST_LIB))..."
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust all
else
	@echo "KBUILD: Rust safety layer disabled; not building $(RUST_LIB)"
endif

$(RUST_LIB_OBJ): $(RUST_LIB) $(RUST_BUILTINS_STUBS)
ifeq ($(ENABLE_RUST),1)
	@echo "Linking Rust safety archive into relocatable object..."
	$(LD_CMD) -r -o $@ --whole-archive $< --no-whole-archive $(RUST_BUILTINS_STUBS)
	@printf 'cmd_rust/libdsmil_rust.o := %s -r -o %s --whole-archive %s --no-whole-archive %s\n' \
		"$(LD_CMD)" "$@" "$(RUST_LIB)" "$(RUST_BUILTINS_STUBS)" > $(RUST_DIR)/.libdsmil_rust.o.cmd
	@echo "Stripping unsupported FMA sections for legacy objtool..."
	@$(OBJCOPY) --remove-section='.text._ZN17compiler_builtins4math9libm_math4arch3x863fma13fma_with_fma4*' $@ 2>/dev/null || true
	@echo "Generated $@"
endif

# ============================================================================

.PHONY: all clean install rust-lib check-rust rust-check rust-test rust-docs \
        debug rust-lib-debug info structure help dsmil-104dev dsmil-84dev \
        rust-setup rust-lib-only

dsmil-104dev:
	$(MAKE) -C $(KDIR) M=$(CURDIR) ENABLE_RUST=0 ONLY_104DEV=1 modules

dsmil-84dev:
	$(MAKE) -C $(KDIR) M=$(CURDIR) ENABLE_RUST=0 ONLY_84DEV=1 modules

rust-setup:
	@echo "Configuring Rust toolchain for DSMIL kernel safety layer..."
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust setup

rust-lib-only:
	@echo "Building Rust safety library only (no kernel modules)..."
	$(MAKE) -C $(RUST_DIR) -f Makefile.rust all
