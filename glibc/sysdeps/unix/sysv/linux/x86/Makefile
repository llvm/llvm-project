# We don't need any header files.
abi-includes :=

abi-variants := 32 64 x32

abi-32-condition := !defined __x86_64__
abi-64-condition := defined __x86_64__ && defined __LP64__
abi-x32-condition := defined __x86_64__ && defined __ILP32__

ifeq ($(subdir),misc)
sysdep_headers += sys/elf.h sys/perm.h sys/reg.h sys/vm86.h sys/debugreg.h sys/io.h
endif

ifeq ($(subdir),nptl)
CFLAGS-elision-lock.c += -mrtm
CFLAGS-elision-unlock.c += -mrtm
CFLAGS-elision-timed.c += -mrtm
CFLAGS-elision-trylock.c += -mrtm
endif

ifeq ($(subdir),setjmp)
tests += tst-saved_mask-1
endif

ifneq ($(enable-cet),no)
ifeq ($(subdir),elf)
tests += tst-cet-property-1 tst-cet-property-2

CFLAGS-tst-cet-property-1.o += -fcf-protection
ASFLAGS-tst-cet-property-dep-2.o += -fcf-protection

$(objpfx)tst-cet-property-2: $(objpfx)tst-cet-property-dep-2.o
$(objpfx)tst-cet-property-2.out: $(objpfx)tst-cet-property-2 \
				 $(objpfx)tst-cet-property-1.out
	env $(run-program-env) $(test-via-rtld-prefix) \
	  $(objpfx)tst-cet-property-2 \
	  < $(objpfx)tst-cet-property-1.out > $@; \
	  $(evaluate-test)
endif

ifeq ($(subdir),posix)
tests += tst-cet-vfork-1
CFLAGS-tst-cet-vfork-1.c += -mshstk
endif

ifeq ($(subdir),stdlib)
tests += tst-cet-setcontext-1
CFLAGS-tst-cet-setcontext-1.c += -mshstk
endif
endif
