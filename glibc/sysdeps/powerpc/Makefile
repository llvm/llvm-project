ifeq ($(subdir),string)
CFLAGS-memcmp.c += -Wno-uninitialized
endif

ifeq ($(subdir),elf)
# extra shared linker files to link into dl-allobjs.so and libc
sysdep-dl-routines += dl-machine hwcapinfo
sysdep_routines += dl-machine hwcapinfo
# extra shared linker files to link only into dl-allobjs.so
sysdep-rtld-routines += dl-machine hwcapinfo

modules-names += mod-tlsopt-powerpc
mod-tlsopt-powerpc.so-no-z-defs = yes
tests += tst-tlsopt-powerpc
$(objpfx)tst-tlsopt-powerpc: $(objpfx)mod-tlsopt-powerpc.so

tests-static += tst-cache-ppc-static
tests-internal += tst-cache-ppc-static

ifeq (yes,$(build-shared))
modules-names += mod-cache-ppc
tests += tst-cache-ppc tst-cache-ppc-static-dlopen
tests-static += tst-cache-ppc-static-dlopen
test-internal-extras += mod-cache-ppc

mod-cache-ppc.so-no-z-defs = yes
tst-cache-ppc-static-dlopen-ENV = LD_LIBRARY_PATH=$(objpfx):$(common-objpfx):$(common-objpfx)elf
$(objpfx)tst-cache-ppc-static-dlopen.out: $(objpfx)mod-cache-ppc.so

$(objpfx)tst-cache-ppc: $(objpfx)mod-cache-ppc.so
endif

ifneq (no,$(multi-arch))
tests-static += tst-tlsifunc-static
tests-internal += tst-tlsifunc-static
ifeq (yes,$(build-shared))
tests-internal += tst-tlsifunc
endif # build-shared
endif # multi-arch
endif # subdir = elf

ifeq ($(subdir),setjmp)
ifeq (yes,$(build-shared))
sysdep_routines += novmx-longjmp novmx-sigjmp
endif
endif

ifeq ($(subdir),csu)
# get offset to rtld_global._dl_hwcap and rtld_global._dl_hwcap2
gen-as-const-headers += rtld-global-offsets.sym
# get offset to __locale_struct.__ctype_tolower
gen-as-const-headers += locale-defines.sym
endif

ifeq ($(subdir),nptl)
tests-internal += test-get_hwcap test-get_hwcap-static
tests-static += test-get_hwcap-static
endif

ifeq ($(subdir),misc)
sysdep_headers += sys/platform/ppc.h
tests += test-gettimebase
tests += tst-set_ppr

# This test is expected to run and exit with EXIT_UNSUPPORTED on
# processors that do not implement the Power ISA 2.06 or greater.
# But the test makes use of instructions from Power ISA 2.06 and 2.07.
CFLAGS-tst-set_ppr.c += -Wa,-many
endif

ifeq ($(subdir),wcsmbs)
CFLAGS-wcscpy.c += -DUNROLL_NTIMES=8
CFLAGS-wcschr.c += -DUNROLL_NTIMES=8
CFLAGS-wcsrchr.c += -DUNROLL_NTIMES=8
endif
