# Linux/m68k uses Motorola asm syntax and the ELF format.

m68k-syntax-flag = -DMOTOROLA_SYNTAX

ifeq ($(subdir),csu)
sysdep_routines += m68k-helpers
ifeq (yes,$(build-shared))
sysdep_routines += divdi3
shared-only-routines += divdi3
CPPFLAGS-divdi3.c = -Din_divdi3_c
endif
endif

ifeq ($(subdir),misc)
sysdep_routines += mremap
sysdep_headers += sys/reg.h
endif

ifeq ($(subdir),elf)
sysdep-others += lddlibc4
install-bin += lddlibc4
endif
