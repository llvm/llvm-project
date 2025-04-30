ifeq ($(subdir),misc)
sysdep_headers += sys/elf.h
endif

ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

abi-variants := lp64
abi-variants += lp64_be

ifeq (,$(filter $(default-abi),$(abi-variants)))
Unknown ABI, must be one of $(abi-variants)
endif

abi-lp64-condition := !defined __AARCH64EB__
abi-lp64_be-condition := defined __AARCH64EB__
