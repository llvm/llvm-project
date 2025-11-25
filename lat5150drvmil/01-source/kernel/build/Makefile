CC ?= gcc
LD_CMD := $(shell $(CC) -print-prog-name=ld)
OBJCOPY ?= $(shell $(CC) -print-prog-name=objcopy)

export SKIP_OBJTOOL := 1
export CONFIG_STACK_VALIDATION := n
OBJTOOL_WRAPPER := $(CURDIR)/scripts/skip-objtool.sh
export OBJTOOL := $(OBJTOOL_WRAPPER)

# Rust library integration
RUST_BUILD_DIR := rust
RUST_LIB := $(RUST_BUILD_DIR)/libdsmil_rust.a
RUST_LIB_OBJ := $(RUST_BUILD_DIR)/libdsmil_rust.o
RUST_BUILTINS_STUBS := $(RUST_BUILD_DIR)/compiler_builtins_stubs.o

ENABLE_RUST ?= 1

obj-m += dsmil-84dev.o
ifeq ($(ENABLE_RUST),1)
dsmil-84dev-objs := dsmil_driver_module.o $(RUST_LIB_OBJ)
# Mark both the Rust object and final module as non-standard to skip objtool
OBJECT_FILES_NON_STANDARD_rust/libdsmil_rust.o := y
OBJECT_FILES_NON_STANDARD_dsmil-84dev.o := y
OBJECT_FILES_NON_STANDARD_dsmil_driver_module.o := y
else
dsmil-84dev-objs := dsmil_driver_module.o rust_stubs.o
OBJECT_FILES_NON_STANDARD_dsmil-84dev.o := y
OBJECT_FILES_NON_STANDARD_dsmil_driver_module.o := y
OBJECT_FILES_NON_STANDARD_rust_stubs.o := y
endif
OBJECT_FILES_NON_STANDARD := y
SKIP_STACK_VALIDATION := y

# Simple single-object build (MODULE_LICENSE should be in main object)
# dsmil-72dev-objs := dsmil-72dev.o rust_stubs.o

# Single object build for now to avoid MODULE_LICENSE issues

# Kernel build directory
KDIR := /lib/modules/$(shell uname -r)/build

# Dependencies - ensure Rust library is built before kernel module

# Build the module
ifeq ($(ENABLE_RUST),1)
all: rust-lib $(RUST_LIB_OBJ)
else
all:
endif
	$(MAKE) -C $(KDIR) M=$(CURDIR) OBJTOOL=$(OBJTOOL_WRAPPER) SKIP_OBJTOOL=1 SKIP_STACK_VALIDATION=y OBJECT_FILES_NON_STANDARD=y CONFIG_STACK_VALIDATION= modules

# Build Rust library first
rust-lib:
ifeq ($(ENABLE_RUST),1)
	@echo "Building Rust safety layer..."
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust all
	@echo "Rust library built successfully"
else
	@echo "Rust safety layer disabled; skipping Rust build"
endif

$(RUST_BUILTINS_STUBS): $(RUST_BUILD_DIR)/compiler_builtins_stubs.c
	@echo "Compiling compiler builtins stubs..."
	$(CC) -c -o $@ $<

$(RUST_LIB_OBJ): $(RUST_LIB) $(RUST_BUILTINS_STUBS)
ifeq ($(ENABLE_RUST),1)
	@echo "Linking Rust safety archive into relocatable object..."
	$(LD_CMD) -r -o $@ --whole-archive $< --no-whole-archive $(RUST_BUILTINS_STUBS)
	@printf 'cmd_rust/libdsmil_rust.o := %s -r -o %s --whole-archive %s --no-whole-archive %s\n' "$(LD_CMD)" "$@" "$(RUST_LIB)" "$(RUST_BUILTINS_STUBS)" > $(RUST_BUILD_DIR)/.libdsmil_rust.o.cmd
	@echo "Stripping unsupported FMA sections for legacy objtool..."
	@$(OBJCOPY) --remove-section='.text._ZN17compiler_builtins4math9libm_math4arch3x863fma13fma_with_fma4*' $@ 2>/dev/null || true
	@echo "Generated $@"
endif

check-rust:
	@if [ "$(ENABLE_RUST)" = "1" ] && [ ! -f $(RUST_LIB) ]; then \
		echo "Building missing Rust library..."; \
		$(MAKE) rust-lib; \
	fi

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust clean
	$(RM) $(RUST_LIB_OBJ) $(RUST_BUILTINS_STUBS) $(RUST_BUILD_DIR)/.libdsmil_rust.o.cmd

# Install the module
install: all
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules_install
	depmod -a

# Rust-specific targets
rust-check:
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust check

rust-test:
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust test

rust-docs:
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust docs

# Debug build with Rust debug info
debug: rust-lib-debug
	$(MAKE) -C $(KDIR) M=$(PWD) modules

rust-lib-debug:
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust debug

# Show information about the build
info:
	@echo "DSMIL Kernel Module Build Information"
	@echo "====================================="
	@echo "Rust Library: $(RUST_LIB)"
	@echo "Library exists: $(shell [ -f $(RUST_LIB) ] && echo 'Yes' || echo 'No')"
	@echo "Library size: $(shell [ -f $(RUST_LIB) ] && stat -c%s $(RUST_LIB) || echo '0') bytes"
	@echo "Kernel Dir: $(KDIR)"
	@echo "Build Dir: $(PWD)"
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust info

# Help target
help:
	@echo "DSMIL Kernel Module Makefile"
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Build kernel module with Rust integration (default)"
	@echo "  clean        - Clean all build artifacts"
	@echo "  install      - Install kernel module"
	@echo "  info         - Show build information"
	@echo ""
	@echo "Rust-specific targets:"
	@echo "  rust-lib     - Build only the Rust library"
	@echo "  rust-check   - Check Rust code without building"
	@echo "  rust-test    - Run Rust unit tests"
	@echo "  rust-docs    - Generate Rust documentation"
	@echo "  debug        - Build with debug information"
	@echo ""
	@echo "For more Rust build options, see: make -C rust -f Makefile.rust help"

.PHONY: all clean install rust-lib check-rust rust-check rust-test rust-docs debug rust-lib-debug info help
