# Building hard-float libm as MIPS16 actually produces larger code size,
# so avoid doing so.
ifeq ($(subdir),math)
sysdep-CFLAGS += -mno-mips16
endif
