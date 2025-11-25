# SPDX-License-Identifier: GPL-2.0
#
# Makefile for Dell Military Specification Subsystem Driver
#

# Module name
obj-m := dell-milspec.o

# Source files
dell-milspec-objs := dell-millspec-enhanced.o

# Kernel build directory (adjust as needed)
KDIR ?= /lib/modules/$(shell uname -r)/build

# Compiler flags
ccflags-y := -I$(src)
ccflags-$(CONFIG_DELL_MILSPEC_DEBUG) += -DDEBUG

# Native optimizations for Meteor Lake
ccflags-y += -march=alderlake -mtune=alderlake -O3
ccflags-y += -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl
ccflags-y += -mavx512vnni -mavx512bf16 -mavx512vbmi -mavx512vbmi2
ccflags-y += -mprefer-vector-width=512

# Module signing (if needed)
MODULE_SIG_ALL := 1

# Build targets
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

# Development targets
debug:
	$(MAKE) -C $(KDIR) M=$(PWD) ccflags-y+="-DDEBUG -g" modules

sparse:
	$(MAKE) -C $(KDIR) M=$(PWD) C=2 CF="-D__CHECK_ENDIAN__" modules

# Load/unload for testing
load:
	sudo modprobe dell-smbios
	sudo insmod dell-milspec.ko milspec_debug=0xFF

unload:
	sudo rmmod dell-milspec

reload: unload load

# Check for required kernel config
check-config:
	@echo "Checking kernel configuration..."
	@if ! grep -q "CONFIG_DELL_SMBIOS=y\|CONFIG_DELL_SMBIOS=m" $(KDIR)/.config; then \
		echo "ERROR: CONFIG_DELL_SMBIOS not enabled"; \
		exit 1; \
	fi
	@if ! grep -q "CONFIG_TPM=y\|CONFIG_TPM=m" $(KDIR)/.config; then \
		echo "WARNING: CONFIG_TPM not enabled (TPM features disabled)"; \
	fi
	@if ! grep -q "CONFIG_I2C=y\|CONFIG_I2C=m" $(KDIR)/.config; then \
		echo "WARNING: CONFIG_I2C not enabled (crypto chip support disabled)"; \
	fi
	@echo "Kernel configuration OK"

# Documentation
docs:
	@echo "Building documentation..."
	@mkdir -p Documentation/dell-milspec
	@echo "Dell Military Specification Subsystem" > Documentation/dell-milspec/README.md
	@echo "=====================================" >> Documentation/dell-milspec/README.md
	@echo "" >> Documentation/dell-milspec/README.md
	@echo "This driver provides support for Dell military specification features" >> Documentation/dell-milspec/README.md
	@echo "including Mode 5 security, DSMIL subsystems, and hardware crypto." >> Documentation/dell-milspec/README.md

# Testing
test: all
	@echo "Running basic module tests..."
	@sudo dmesg -c > /dev/null
	@sudo insmod dell-milspec.ko
	@sleep 1
	@if sudo dmesg | grep -q "MIL-SPEC: Driver loaded successfully"; then \
		echo "PASS: Module loaded"; \
	else \
		echo "FAIL: Module load failed"; \
		sudo dmesg; \
	fi
	@sudo rmmod dell-milspec

.PHONY: all clean install debug sparse load unload reload check-config docs test
