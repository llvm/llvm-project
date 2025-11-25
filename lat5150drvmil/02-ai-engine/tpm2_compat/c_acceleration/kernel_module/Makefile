obj-m += tpm2_accel_early.o

# Compiler flags for early boot module
ccflags-y += -DTPM2_ACCEL_EARLY_BOOT
ccflags-y += -DTPM2_ACCEL_INTEL_CORE_ULTRA_7_165H
ccflags-y += -DTPM2_ACCEL_NPU_34_TOPS
ccflags-y += -DTPM2_ACCEL_DELL_MILSPEC
ccflags-y += -DTPM2_ACCEL_VERSION=\"1.0.0\"

# Optimization flags
ccflags-y += -O2 -fno-strict-aliasing -fno-common
ccflags-y += -Wall -Wextra -Werror
ccflags-y += -Wno-unused-parameter -Wno-sign-compare

# Security hardening flags
ccflags-y += -fstack-protector-strong
ccflags-y += -D_FORTIFY_SOURCE=2

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.o *.ko *.mod.c *.mod.o *.order *.symvers
	rm -rf .tmp_versions

.PHONY: all clean
