obj-m += dsmil-72dev.o

# Rust library integration
RUST_LIB := rust/libdsmil_rust.a
RUST_BUILD_DIR := rust

# Simple single-object build (MODULE_LICENSE should be in main object)
# dsmil-72dev-objs := dsmil-72dev.o rust_stubs.o

# Single object build for now to avoid MODULE_LICENSE issues

# Kernel build directory
KDIR := /lib/modules/$(shell uname -r)/build

# Dependencies - ensure Rust library is built before kernel module

# Build the module
all: rust-lib
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Build Rust library first
rust-lib:
	@echo "Building Rust safety layer..."
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust all
	@echo "Rust library built successfully"

# Check if Rust library exists
check-rust:
	@if [ ! -f $(RUST_LIB) ]; then \
		echo "Building missing Rust library..."; \
		$(MAKE) rust-lib; \
	fi

# Clean build artifacts
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	$(MAKE) -C $(RUST_BUILD_DIR) -f Makefile.rust clean

# Install the module
install: all
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
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