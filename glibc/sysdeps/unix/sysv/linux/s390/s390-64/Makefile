# See Makeconfig regarding the use of default-abi.
default-abi := 64

ifeq ($(subdir),misc)
sysdep_headers += sys/elf.h
endif

ifeq ($(subdir),elf)
ifeq (yes,$(build-shared))
# This is needed to support g++ v2 and v3.
sysdep_routines += framestate
shared-only-routines += framestate
endif
endif

ifeq ($(subdir),stdlib)
sysdep_routines += __makecontext_ret
endif
