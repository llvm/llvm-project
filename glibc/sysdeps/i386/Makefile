# The mpn functions need a #define for asm syntax flavor.
# Every i386 port in use uses gas syntax (I think).
asm-CPPFLAGS += -DGAS_SYNTAX

# The i386 `long double' is a distinct type we support.
long-double-fcts = yes

ifeq ($(subdir),math)
# These functions change the rounding mode internally and need to
# update both the SSE2 rounding mode and the 387 rounding mode.  See
# the handling of MATH_SET_BOTH_ROUNDING_MODES in
# sysdeps/i386/fpu/fenv_private.h.
CFLAGS-e_gamma_r.c += -DMATH_SET_BOTH_ROUNDING_MODES
endif

ifeq ($(subdir),gmon)
sysdep_routines += i386-mcount
endif

ifeq ($(subdir),elf)
CFLAGS-rtld.c += -Wno-uninitialized -Wno-unused
CFLAGS-dl-load.c += -Wno-unused
CFLAGS-dl-reloc.c += -Wno-unused
endif

ifeq ($(subdir),csu)
gen-as-const-headers += link-defines.sym
else
stack-align-test-flags += -malign-double
endif

ifeq ($(subdir),elf)
sysdep-dl-routines += tlsdesc dl-tlsdesc

tests += tst-audit3
modules-names += tst-auditmod3a tst-auditmod3b

$(objpfx)tst-audit3: $(objpfx)tst-auditmod3a.so
$(objpfx)tst-audit3.out: $(objpfx)tst-auditmod3b.so
tst-audit3-ENV = LD_AUDIT=$(objpfx)tst-auditmod3b.so
endif

ifeq ($(subdir),csu)
gen-as-const-headers += tlsdesc.sym
endif

# Make sure no code in ld.so uses mm/xmm/ymm/zmm registers on i386 since
# the first 3 mm/xmm/ymm/zmm registers are used to pass vector parameters
# which must be preserved.
# With SSE disabled, ensure -fpmath is not set to use sse either.
rtld-CFLAGS += -mno-sse -mno-mmx -mfpmath=387
ifeq ($(subdir),elf)
CFLAGS-.os += $(if $(filter $(@F),$(patsubst %,%.os,$(all-rtld-routines))),\
		   $(rtld-CFLAGS))

tests-special += $(objpfx)tst-ld-sse-use.out
$(objpfx)tst-ld-sse-use.out: ../sysdeps/i386/tst-ld-sse-use.sh $(objpfx)ld.so
	@echo "Checking ld.so for SSE register use.  This will take a few seconds..."
	$(BASH) $< $(objpfx) '$(NM)' '$(OBJDUMP)' '$(READELF)' > $@; \
	$(evaluate-test)
else
CFLAGS-.os += $(if $(filter rtld-%.os,$(@F)), $(rtld-CFLAGS))
endif
