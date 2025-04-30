long-double-fcts = yes

ifeq (yes,$(aarch64-bti))
# Mark linker output BTI compatible, it warns on non-BTI inputs.
sysdep-LDFLAGS += -Wl,-z,force-bti
# Make warnings fatal outside the test system.
LDFLAGS-lib.so += -Wl,--fatal-warnings
LDFLAGS-rtld += -Wl,-z,force-bti,--fatal-warnings
endif

ifeq ($(subdir),elf)
sysdep-dl-routines += dl-bti
endif

ifeq ($(subdir),elf)
sysdep-dl-routines += tlsdesc dl-tlsdesc
gen-as-const-headers += dl-link.sym

tests-internal += tst-ifunc-arg-1 tst-ifunc-arg-2

ifeq (yes,$(aarch64-variant-pcs))
tests += tst-vpcs
modules-names += tst-vpcs-mod
LDFLAGS-tst-vpcs-mod.so = -Wl,-z,lazy
$(objpfx)tst-vpcs: $(objpfx)tst-vpcs-mod.so
endif
endif

ifeq ($(subdir),csu)
gen-as-const-headers += tlsdesc.sym
endif

ifeq ($(subdir),gmon)
CFLAGS-mcount.c += -mgeneral-regs-only
endif

ifeq ($(subdir),math)
CPPFLAGS += -I../soft-fp
endif

ifeq ($(subdir),misc)
sysdep_headers += sys/ifunc.h
sysdep_routines += __mtag_tag_zero_region \
		   __mtag_tag_region
endif

ifeq ($(subdir),malloc)
sysdep_malloc_debug_routines = __mtag_tag_zero_region __mtag_tag_region
endif
