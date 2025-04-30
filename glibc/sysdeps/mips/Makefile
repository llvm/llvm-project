ifeq ($(subdir),misc)
sysdep_headers += regdef.h fpregdef.h sys/regdef.h sys/fpregdef.h \
		  sys/asm.h sgidefs.h
endif

ifeq ($(subdir),setjmp)
sysdep_routines += setjmp_aux
endif

ifeq ($(subdir),rt)
librt-sysdep_routines += rt-sysdep
librt-shared-only-routines += rt-sysdep
endif

ifeq ($(subdir),csu)
CPPFLAGS-crti.S += $(pic-ccflag)
CPPFLAGS-crtn.S += $(pic-ccflag)
endif

ASFLAGS-.os += $(pic-ccflag)
# libc.a and libc_p.a must be compiled with -fPIE/-fpie for static PIE.
ASFLAGS-.o += $(pie-default)
ASFLAGS-.op += $(pie-default)

ifeq ($(subdir),elf)
ifneq ($(o32-fpabi),)
tests += tst-abi-interlink

fpabi-modules-names =
fpabi_list =
ifneq (,$(filter $(o32-fpabi),32 xx xxo))
fpabi-modules-names += tst-abi-fp32mod
CFLAGS-tst-abi-fp32mod.c += -mfp32
endif
ifneq (,$(filter $(o32-fpabi),xx))
fpabi-modules-names += tst-abi-fpxxmod
CFLAGS-tst-abi-fpxxmod.c += -mfpxx -mno-odd-spreg
endif
ifneq (,$(filter $(o32-fpabi),xx xxo))
ifeq ($(has-modd-spreg),yes)
fpabi-modules-names += tst-abi-fpxxomod
CFLAGS-tst-abi-fpxxomod.c += -mfpxx -modd-spreg
endif
endif
ifneq (,$(filter $(o32-fpabi),xx 64a))
ifeq ($(has-mpf64),yes)
fpabi-modules-names += tst-abi-fp64amod
CFLAGS-tst-abi-fp64amod.c += -mfp64 -mno-odd-spreg
endif
endif
ifneq (,$(filter $(o32-fpabi),xx xxo 64a 64))
ifeq ($(has-mpf64)-$(has-modd-spreg),yes-yes)
fpabi-modules-names += tst-abi-fp64mod
CFLAGS-tst-abi-fp64mod.c += -mfp64 -modd-spreg
endif
endif
modules-names += $(fpabi-modules-names)

comma:=,
empty:=
space:=$(empty) $(empty)
fpabi_list=$(subst $(space),$(comma),$(patsubst tst-abi-%mod,o_%,\
				     $(fpabi-modules-names)))
CPPFLAGS-tst-abi-interlink.c += -DFPABI_LIST=$(fpabi_list)
CPPFLAGS-tst-abi-interlink.c += -DFPABI_COUNT=$(words $(fpabi-modules-names))
CPPFLAGS-tst-abi-interlink.c += -DFPABI_NATIVE=o_fp$(o32-fpabi)
$(objpfx)tst-abi-interlink.out: $(patsubst %,$(objpfx)%.so,\
					   $(fpabi-modules-names))
endif

ifeq ($(mips-mode-switch),yes)
ifeq ($(o32-fpabi),xx)
tests += tst-mode-switch-1 tst-mode-switch-2 tst-mode-switch-3
endif
endif
endif

ifeq ($(subdir),elf)
tests += tst-undefined-weak
modules-names += tst-undefined-weak-lib

$(objpfx)tst-undefined-weak: $(objpfx)tst-undefined-weak-lib.so
endif
