ifeq ($(subdir),elf)
# Prevent the use of VSX registers and insns in _dl_start, which under -O3
# optimization may require a TOC reference before relocations are resolved.
CFLAGS-rtld.c += $(ibm128-abi-CFLAGS) -mno-vsx

# Likewise, remove options to build IEEE long double
$(foreach suf,$(all-object-suffixes),$(objpfx)rtld$(suf)): \
  sysdep-CFLAGS := $(filter-out -mabi=ieeelongdouble,$(sysdep-CFLAGS))
endif

ifeq ($(subdir),string)
sysdep_routines += strstr-ppc64
CFLAGS-strncase.c += -funroll-loops
CFLAGS-strncase_l.c += -funroll-loops
endif
