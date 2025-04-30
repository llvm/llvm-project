# See Makeconfig regarding the use of default-abi.
default-abi := 32

ifeq ($(subdir),csu)
ifeq (yes,$(build-shared))
sysdep_routines += divdi3
shared-only-routines += divdi3
CPPFLAGS-divdi3.c = -Din_divdi3_c
endif
endif

ifeq ($(subdir),conform)
# For bug 21635.
conformtest-xfail-conds += powerpc32-linux
endif
