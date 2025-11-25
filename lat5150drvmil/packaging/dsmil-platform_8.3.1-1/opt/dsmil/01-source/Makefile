# Military Device Interface Makefile
# Dell Latitude 5450 MIL-SPEC DSMIL Device Interface
# 
# PHASE 1: Safe Foundation Build System for 0x8000-0x806B range
# - READ-ONLY operations only
# - Safety-first compilation flags
# - Integration with existing DSMIL kernel module
# 
# Copyright (C) 2025 JRTC1 Educational Development

# Project Configuration
PROJECT_NAME := military_device_interface
VERSION := 1.0.0
PHASE := Phase1

# Directories
SRCDIR := .
OBJDIR := obj
BINDIR := bin
KERNEL_DIR := 01-source/kernel
DOCS_DIR := docs

# Source Files
HEADERS := military_device_interface.h
SOURCES := military_device_lib.c
TEST_SOURCES := test_military_interface.c
OBJECTS := $(SOURCES:%.c=$(OBJDIR)/%.o)
TEST_OBJECTS := $(TEST_SOURCES:%.c=$(OBJDIR)/%.o)

# Output Files
LIBRARY := $(BINDIR)/libmilitary_device.a
SHARED_LIB := $(BINDIR)/libmilitary_device.so
TEST_PROGRAM := $(BINDIR)/test_military_interface

# Compiler Configuration
CC := gcc
AR := ar
STRIP := strip

# Safety-First Compiler Flags
CFLAGS := -std=c11 -Wall -Wextra -Werror -pedantic -fPIC
CFLAGS += -D_GNU_SOURCE -D_DEFAULT_SOURCE
CFLAGS += -DMILDEV_VERSION_STRING=\"$(VERSION)\"
CFLAGS += -DMILDEV_PHASE=\"$(PHASE)\"

# Debug/Release Configuration
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CFLAGS += -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
    BUILD_TYPE := Debug
else
    CFLAGS += -O2 -DNDEBUG -fstack-protector-strong
    LDFLAGS += -Wl,-z,relro,-z,now
    BUILD_TYPE := Release
endif

# Platform-Specific Flags
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M), x86_64)
    CFLAGS += -march=native -mtune=native
endif

# Include Directories
INCLUDES := -I$(SRCDIR) -I$(KERNEL_DIR)

# Library Dependencies
LIBS := -lpthread -lrt
TEST_LIBS := $(LIBS)

# Linker Flags
LDFLAGS += -Wl,--as-needed

# Make Configuration
MAKEFLAGS += --warn-undefined-variables
.DELETE_ON_ERROR:
.SUFFIXES:

# Default Target
.PHONY: all
all: info $(LIBRARY) $(SHARED_LIB) $(TEST_PROGRAM)

# Information Banner
.PHONY: info
info:
	@echo "======================================================="
	@echo "Military Device Interface Build System"
	@echo "Dell Latitude 5450 MIL-SPEC DSMIL Device Interface"
	@echo "======================================================="
	@echo "Version: $(VERSION) ($(PHASE))"
	@echo "Build Type: $(BUILD_TYPE)"
	@echo "Target Architecture: $(UNAME_M)"
	@echo "Compiler: $(CC) $(shell $(CC) --version | head -n1)"
	@echo "Build Directory: $(OBJDIR)"
	@echo "Output Directory: $(BINDIR)"
	@echo "======================================================="
	@echo ""

# Create Directories
$(OBJDIR) $(BINDIR):
	@mkdir -p $@

# Compile Source Files
$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) | $(OBJDIR)
	@echo "Compiling $< ..."
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Build Static Library
$(LIBRARY): $(OBJECTS) | $(BINDIR)
	@echo "Creating static library $@ ..."
	$(AR) rcs $@ $(OBJECTS)
	@echo "Static library created: $@"

# Build Shared Library
$(SHARED_LIB): $(OBJECTS) | $(BINDIR)
	@echo "Creating shared library $@ ..."
	$(CC) -shared $(LDFLAGS) -o $@ $(OBJECTS) $(LIBS)
	@echo "Shared library created: $@"

# Build Test Program
$(TEST_PROGRAM): $(TEST_OBJECTS) $(LIBRARY) | $(BINDIR)
	@echo "Building test program $@ ..."
	$(CC) $(LDFLAGS) -o $@ $(TEST_OBJECTS) -L$(BINDIR) -lmilitary_device $(TEST_LIBS)
	@echo "Test program created: $@"

# Install Targets
.PHONY: install install-dev install-runtime
install: install-dev install-runtime

install-dev: $(LIBRARY) $(SHARED_LIB)
	@echo "Installing development files..."
	install -d $(DESTDIR)/usr/local/lib
	install -d $(DESTDIR)/usr/local/include
	install -m 644 $(LIBRARY) $(DESTDIR)/usr/local/lib/
	install -m 755 $(SHARED_LIB) $(DESTDIR)/usr/local/lib/
	install -m 644 $(HEADERS) $(DESTDIR)/usr/local/include/
	ldconfig
	@echo "Development files installed."

install-runtime: $(TEST_PROGRAM)
	@echo "Installing runtime files..."
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 $(TEST_PROGRAM) $(DESTDIR)/usr/local/bin/
	@echo "Runtime files installed."

# Test Targets
.PHONY: test test-basic test-full test-continuous test-performance
test: test-basic

test-basic: $(TEST_PROGRAM)
	@echo "Running basic safety tests..."
	./$(TEST_PROGRAM) --verbose
	@echo "Basic tests completed."

test-full: $(TEST_PROGRAM)
	@echo "Running comprehensive test suite..."
	./$(TEST_PROGRAM) --verbose --all --performance
	@echo "Full test suite completed."

test-continuous: $(TEST_PROGRAM)
	@echo "Running continuous monitoring test (Ctrl+C to stop)..."
	./$(TEST_PROGRAM) --monitor --interval 5 --verbose

test-performance: $(TEST_PROGRAM)
	@echo "Running performance benchmark..."
	./$(TEST_PROGRAM) --performance --verbose

# Safety Validation
.PHONY: safety-check thermal-check quarantine-check
safety-check: thermal-check quarantine-check
	@echo "Safety validation completed."

thermal-check:
	@echo "Checking thermal conditions..."
	@TEMP=$$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | head -c 2 || echo "50"); \
	TEMP_C=$$((TEMP / 1000)); \
	if [ $$TEMP_C -lt 100 ]; then \
		echo "Thermal conditions safe: $${TEMP_C}°C"; \
	else \
		echo "WARNING: Thermal conditions unsafe: $${TEMP_C}°C"; \
		exit 1; \
	fi

quarantine-check:
	@echo "Verifying quarantine list integrity..."
	@grep -q "0x8009\|0x800A\|0x800B\|0x8019\|0x8029" $(HEADERS) && \
		echo "Quarantine list present in headers" || \
		(echo "ERROR: Quarantine list missing!" && exit 1)

# Kernel Module Integration
.PHONY: check-kernel load-kernel unload-kernel
check-kernel:
	@echo "Checking DSMIL kernel module status..."
	@if lsmod | grep -q dsmil; then \
		echo "DSMIL kernel module loaded"; \
	else \
		echo "DSMIL kernel module not loaded"; \
	fi
	@if [ -c /dev/dsmil-72dev ]; then \
		echo "DSMIL device node available: /dev/dsmil-72dev"; \
	else \
		echo "DSMIL device node not found"; \
	fi

load-kernel:
	@echo "Loading DSMIL kernel module..."
	@if [ -f "$(KERNEL_DIR)/dsmil-72dev.ko" ]; then \
		sudo insmod "$(KERNEL_DIR)/dsmil-72dev.ko" force_jrtc1_mode=true thermal_threshold=100; \
		echo "DSMIL kernel module loaded in JRTC1 safe mode"; \
	else \
		echo "ERROR: DSMIL kernel module not found at $(KERNEL_DIR)/dsmil-72dev.ko"; \
		exit 1; \
	fi

unload-kernel:
	@echo "Unloading DSMIL kernel module..."
	sudo rmmod dsmil-72dev || true
	@echo "DSMIL kernel module unloaded"

# Development Targets
.PHONY: debug release clean-debug clean-release
debug:
	$(MAKE) DEBUG=1 all

release:
	$(MAKE) DEBUG=0 all

clean-debug:
	$(MAKE) DEBUG=1 clean

clean-release:
	$(MAKE) DEBUG=0 clean

# Documentation Targets
.PHONY: docs docs-clean
docs:
	@echo "Generating documentation..."
	@mkdir -p $(DOCS_DIR)
	@echo "# Military Device Interface Documentation" > $(DOCS_DIR)/README.md
	@echo "" >> $(DOCS_DIR)/README.md
	@echo "Version: $(VERSION) ($(PHASE))" >> $(DOCS_DIR)/README.md
	@echo "Build Type: $(BUILD_TYPE)" >> $(DOCS_DIR)/README.md
	@echo "" >> $(DOCS_DIR)/README.md
	@echo "## API Reference" >> $(DOCS_DIR)/README.md
	@echo "" >> $(DOCS_DIR)/README.md
	@echo "See military_device_interface.h for complete API documentation." >> $(DOCS_DIR)/README.md
	@echo "Documentation generated in $(DOCS_DIR)/"

docs-clean:
	rm -rf $(DOCS_DIR)

# Clean Targets
.PHONY: clean distclean
clean:
	@echo "Cleaning build files..."
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)
	@echo "Clean completed."

distclean: clean docs-clean
	@echo "Performing deep clean..."
	find . -name "*.o" -delete
	find . -name "*.a" -delete
	find . -name "*.so" -delete
	find . -name "*~" -delete
	find . -name "core" -delete
	@echo "Deep clean completed."

# Package Targets
.PHONY: package package-source
package: all docs
	@echo "Creating deployment package..."
	@mkdir -p package/$(PROJECT_NAME)-$(VERSION)
	cp -r $(BINDIR) package/$(PROJECT_NAME)-$(VERSION)/
	cp -r $(DOCS_DIR) package/$(PROJECT_NAME)-$(VERSION)/
	cp $(HEADERS) package/$(PROJECT_NAME)-$(VERSION)/
	cp README* package/$(PROJECT_NAME)-$(VERSION)/ 2>/dev/null || true
	cd package && tar -czf $(PROJECT_NAME)-$(VERSION)-linux-$(UNAME_M).tar.gz $(PROJECT_NAME)-$(VERSION)
	@echo "Package created: package/$(PROJECT_NAME)-$(VERSION)-linux-$(UNAME_M).tar.gz"

package-source:
	@echo "Creating source package..."
	@mkdir -p package
	tar -czf package/$(PROJECT_NAME)-$(VERSION)-source.tar.gz \
		--exclude=package \
		--exclude=$(OBJDIR) \
		--exclude=$(BINDIR) \
		--exclude=.git \
		.
	@echo "Source package created: package/$(PROJECT_NAME)-$(VERSION)-source.tar.gz"

# Help Target
.PHONY: help
help:
	@echo "Military Device Interface Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all targets (default)"
	@echo "  debug            - Build with debug symbols and sanitizers"
	@echo "  release          - Build optimized release version"
	@echo "  test             - Run basic safety tests"
	@echo "  test-full        - Run comprehensive test suite"
	@echo "  test-continuous  - Run continuous monitoring test"
	@echo "  test-performance - Run performance benchmark"
	@echo "  safety-check     - Validate safety conditions"
	@echo "  check-kernel     - Check DSMIL kernel module status"
	@echo "  load-kernel      - Load DSMIL kernel module (requires sudo)"
	@echo "  unload-kernel    - Unload DSMIL kernel module (requires sudo)"
	@echo "  install          - Install library and runtime"
	@echo "  package          - Create deployment package"
	@echo "  docs             - Generate documentation"
	@echo "  clean            - Clean build files"
	@echo "  distclean        - Deep clean all generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  DEBUG=1          - Enable debug build"
	@echo "  CC=compiler      - Override compiler"
	@echo "  DESTDIR=path     - Override install destination"

# Dependency Management
-include $(OBJECTS:.o=.d)
-include $(TEST_OBJECTS:.o=.d)

$(OBJDIR)/%.d: $(SRCDIR)/%.c | $(OBJDIR)
	@$(CC) $(CFLAGS) $(INCLUDES) -MM -MT $(@:.d=.o) $< > $@

# Static Analysis (if available)
.PHONY: analyze
analyze:
	@if command -v cppcheck >/dev/null 2>&1; then \
		echo "Running static analysis..."; \
		cppcheck --enable=all --std=c11 --language=c $(SOURCES) $(TEST_SOURCES); \
	else \
		echo "cppcheck not available, skipping static analysis"; \
	fi

# Memory Check (if available)
.PHONY: memcheck
memcheck: $(TEST_PROGRAM)
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Running memory check..."; \
		valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TEST_PROGRAM); \
	else \
		echo "valgrind not available, skipping memory check"; \
	fi

# Special Targets
.PHONY: emergency-stop
emergency-stop:
	@echo "EMERGENCY STOP: Terminating all processes..."
	@pkill -f $(TEST_PROGRAM) || true
	@echo "Emergency stop completed."

# Version Information
.PHONY: version
version:
	@echo "$(PROJECT_NAME) version $(VERSION) ($(PHASE))"