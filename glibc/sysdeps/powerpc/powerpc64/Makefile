# Powerpc64 specific build options.
# this is ./sysdeps/powerpc/powerpc64/Makefile

# Each TOC entry takes 8 bytes and the TOC holds up to 2^16 bytes,
# or 8192 entries.
# If -fpic is not specified, the latest gcc-3.2.1 now generates
# different code for call stubs (without the TOC reload).
# Shared objects need the TOC reload so specify -fpic.
ifeq (yes,$(build-shared))
pic-ccflag = -fpic
endif

# These flags prevent FPU or Altivec registers from being used,
# for code called in contexts that is not allowed to touch those registers.
# Stupid GCC requires us to pass all these ridiculous switches.  We need to
# pass the -mno-* switches as well to prevent the compiler from attempting
# to emit altivec or vsx instructions, especially when the registers aren't
# available.
no-special-regs := $(sort $(foreach n,40 41 50 51 60 61 62 63 \
				      $(foreach m,2 3 4 5 6 7 8 9, \
						  3$m 4$m 5$m),\
				    -ffixed-$n)) \
		   $(sort $(foreach n,$(foreach m,0 1 2 3 4 5 6 7 8 9,\
						$m 1$m 2$m) 30 31,\
				    -ffixed-v$n)) \
		   -ffixed-vrsave -ffixed-vscr -mno-altivec -mno-vsx

# Likewise, we must disable IEEE long double support as some (all?)
# compilers will not accept the above options with IEEE long double.
ifeq ($(ibm128-fcts),yes)
no-special-regs += -mabi=ibmlongdouble
endif

# Need to prevent gcc from using fprs in code used during dynamic linking.

CFLAGS-dl-runtime.os = $(no-special-regs)
CFLAGS-dl-lookup.os = $(no-special-regs)
CFLAGS-dl-misc.os = $(no-special-regs)
CFLAGS-rtld-mempcpy.os = $(no-special-regs)
CFLAGS-rtld-memmove.os = $(no-special-regs)
CFLAGS-rtld-memchr.os = $(no-special-regs)
CFLAGS-rtld-strnlen.os = $(no-special-regs)

ifeq ($(subdir),elf)
# help gcc inline asm code from dl-machine.h
+cflags += -finline-limit=2000
endif

ifeq ($(subdir),gmon)
# The assembly functions assume that fp arg regs are not trashed.
# Compiling with -msoft-float ensures that fp regs are not used
# for moving memory around.
CFLAGS-mcount.c += $(no-special-regs)
sysdep_routines += ppc-mcount
endif

ifeq ($(subdir),setjmp)
tests += tst-setjmp-bug21895-static
tests-static += tst-setjmp-bug21895-static
modules-names += setjmp-bug21895

$(objpfx)tst-setjmp-bug21895-static.out: $(objpfx)setjmp-bug21895.so

tst-setjmp-bug21895-static-ENV = \
	LD_LIBRARY_PATH=$(objpfx):$(common-objpfx):$(common-objpfx)setjmp:$(common-objpfx)elf
endif

ifeq ($(subdir),stdlib)
CFLAGS-tst-ucontext-ppc64-vscr.c += -maltivec
tests += tst-ucontext-ppc64-vscr
endif
