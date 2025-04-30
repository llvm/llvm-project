# The ia64 `long double' is a distinct type we support.
long-double-fcts = yes

ifeq ($(subdir),math)
# sqrtf128 requires soft-fp.
CPPFLAGS += -I../soft-fp
endif

ifeq ($(subdir),gmon)
sysdep_routines += _mcount
endif

ifeq ($(subdir), csu)
CPPFLAGS-start.S = -D__ASSEMBLY__

ifeq (yes,$(build-shared))
# Compatibility
sysdep_routines += ia64libgcc
shared-only-routines += ia64libgcc
endif
endif

ifeq ($(subdir),elf)
sysdep-dl-routines += dl-symaddr dl-fptr
endif
