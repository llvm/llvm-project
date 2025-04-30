ifeq ($(subdir),misc)
sysdep_headers += sys/cachectl.h
sysdep_routines += flush-icache
endif

ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

abi-variants := ilp32 ilp32d lp64 lp64d

ifeq (,$(filter $(default-abi),$(abi-variants)))
$(error Unknown ABI $(default-abi), must be one of $(abi-variants))
endif

abi-ilp32-condition   := __WORDSIZE == 32 && defined __riscv_float_abi_soft
abi-ilp32d-condition  := __WORDSIZE == 32 && defined __riscv_float_abi_double
abi-lp64-condition    := __WORDSIZE == 64 && defined __riscv_float_abi_soft
abi-lp64d-condition   := __WORDSIZE == 64 && defined __riscv_float_abi_double
