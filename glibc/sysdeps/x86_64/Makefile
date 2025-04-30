# The i387 `long double' is a distinct type we support.
long-double-fcts = yes

ifeq ($(subdir),csu)
gen-as-const-headers += link-defines.sym
endif

ifeq ($(subdir),gmon)
sysdep_routines += _mcount
# We cannot compile _mcount.S with -pg because that would create
# recursive calls when ENTRY is used.  Just copy the normal static
# object.
sysdep_noprof += _mcount
endif

ifeq ($(subdir),string)
sysdep_routines += strcasecmp_l-nonascii strncase_l-nonascii
gen-as-const-headers += locale-defines.sym
tests += \
  tst-rsi-strlen
endif

ifeq ($(subdir),elf)
# There is no good reason to use MMX in x86-64 ld.so with GCC.
CFLAGS-.os += $(if $(filter $(@F),$(patsubst %,%.os,$(all-rtld-routines))),\
		   -mno-mmx)

sysdep-dl-routines += tlsdesc dl-tlsdesc tls_get_addr

tests += ifuncmain8
modules-names += ifuncmod8

$(objpfx)ifuncmain8: $(objpfx)ifuncmod8.so

tests += tst-quad1 tst-quad2
modules-names += tst-quadmod1 tst-quadmod2

$(objpfx)tst-quad1: $(objpfx)tst-quadmod1.so
$(objpfx)tst-quad2: $(objpfx)tst-quadmod2.so

quad-pie-test += tst-quad1pie tst-quad2pie
tests += $(quad-pie-test)
tests-pie += $(quad-pie-test)
test-extras += tst-quadmod1pie tst-quadmod2pie
extra-test-objs += tst-quadmod1pie.o tst-quadmod2pie.o

$(objpfx)tst-quad1pie: $(objpfx)tst-quadmod1pie.o
$(objpfx)tst-quad2pie: $(objpfx)tst-quadmod2pie.o

CFLAGS-tst-quad1pie.c = $(PIE-ccflag)
CFLAGS-tst-quad2pie.c = $(PIE-ccflag)

tests += tst-x86_64-1
modules-names += x86_64/tst-x86_64mod-1
LDFLAGS-tst-x86_64mod-1.so = -Wl,-soname,tst-x86_64mod-1.so
ifneq (no,$(have-tunables))
# Test the state size for XSAVE when XSAVEC is disabled.
tst-x86_64-1-ENV = GLIBC_TUNABLES=glibc.cpu.hwcaps=-XSAVEC
endif

$(objpfx)tst-x86_64-1: $(objpfx)x86_64/tst-x86_64mod-1.so

ifneq (no,$(have-tunables))
tests += tst-platform-1
modules-names += tst-platformmod-1 x86_64/tst-platformmod-2
CFLAGS-tst-platform-1.c = -mno-avx
CFLAGS-tst-platformmod-1.c = -mno-avx
CFLAGS-tst-platformmod-2.c = -mno-avx
LDFLAGS-tst-platformmod-2.so = -Wl,-soname,tst-platformmod-2.so
$(objpfx)tst-platform-1: $(objpfx)tst-platformmod-1.so
$(objpfx)tst-platform-1.out: $(objpfx)x86_64/tst-platformmod-2.so
# Turn off AVX512F and AVX2 so that GLRO(dl_platform) is
# always set to x86_64.
tst-platform-1-ENV = LD_PRELOAD=$(objpfx)\$$PLATFORM/tst-platformmod-2.so \
	GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX512F,-AVX2
endif

tests += tst-audit3 tst-audit4 tst-audit5 tst-audit6 tst-audit7 \
	 tst-audit10 tst-sse tst-avx tst-avx512
test-extras += tst-audit4-aux tst-audit10-aux \
	       tst-avx-aux tst-avx512-aux
extra-test-objs += tst-audit4-aux.o tst-audit10-aux.o \
		   tst-avx-aux.o tst-avx512-aux.o

ifeq ($(have-insert),yes)
tests += tst-split-dynreloc
LDFLAGS-tst-split-dynreloc = -Wl,-T,$(..)sysdeps/x86_64/tst-split-dynreloc.lds
tst-split-dynreloc-ENV = LD_BIND_NOW=1
endif

modules-names += tst-auditmod3a tst-auditmod3b \
		tst-auditmod4a tst-auditmod4b \
		tst-auditmod5a tst-auditmod5b \
		tst-auditmod6a tst-auditmod6b tst-auditmod6c \
		tst-auditmod7a tst-auditmod7b \
		tst-auditmod10a tst-auditmod10b \
		tst-ssemod tst-avxmod tst-avx512mod

$(objpfx)tst-audit3: $(objpfx)tst-auditmod3a.so
$(objpfx)tst-audit3.out: $(objpfx)tst-auditmod3b.so
tst-audit3-ENV = LD_AUDIT=$(objpfx)tst-auditmod3b.so

$(objpfx)tst-audit4: $(objpfx)tst-audit4-aux.o $(objpfx)tst-auditmod4a.so
$(objpfx)tst-audit4.out: $(objpfx)tst-auditmod4b.so
tst-audit4-ENV = LD_AUDIT=$(objpfx)tst-auditmod4b.so

$(objpfx)tst-audit5: $(objpfx)tst-auditmod5a.so
$(objpfx)tst-audit5.out: $(objpfx)tst-auditmod5b.so
tst-audit5-ENV = LD_AUDIT=$(objpfx)tst-auditmod5b.so

$(objpfx)tst-audit6: $(objpfx)tst-auditmod6a.so
$(objpfx)tst-audit6.out: $(objpfx)tst-auditmod6b.so \
			 $(objpfx)tst-auditmod6c.so
tst-audit6-ENV = LD_AUDIT=$(objpfx)tst-auditmod6b.so:$(objpfx)tst-auditmod6c.so

$(objpfx)tst-audit7: $(objpfx)tst-auditmod7a.so
$(objpfx)tst-audit7.out: $(objpfx)tst-auditmod7b.so
tst-audit7-ENV = LD_AUDIT=$(objpfx)tst-auditmod7b.so

$(objpfx)tst-audit10: $(objpfx)tst-audit10-aux.o $(objpfx)tst-auditmod10a.so
$(objpfx)tst-audit10.out: $(objpfx)tst-auditmod10b.so
tst-audit10-ENV = LD_AUDIT=$(objpfx)tst-auditmod10b.so

$(objpfx)tst-sse: $(objpfx)tst-ssemod.so
$(objpfx)tst-avx: $(objpfx)tst-avx-aux.o $(objpfx)tst-avxmod.so
$(objpfx)tst-avx512: $(objpfx)tst-avx512-aux.o $(objpfx)tst-avx512mod.so

AVX-CFLAGS = -mavx
ifneq ($(with-clang),yes)
AVX-CFLAGS += -mno-vzeroupper
endif
CFLAGS-tst-audit4-aux.c += $(AVX-CFLAGS)
CFLAGS-tst-auditmod4a.c += $(AVX-CFLAGS)
CFLAGS-tst-auditmod4b.c += $(AVX-CFLAGS)
CFLAGS-tst-auditmod6b.c += $(AVX-CFLAGS)
CFLAGS-tst-auditmod6c.c += $(AVX-CFLAGS)
CFLAGS-tst-auditmod7b.c += $(AVX-CFLAGS)
CFLAGS-tst-avx-aux.c += $(AVX-CFLAGS)
CFLAGS-tst-avxmod.c += $(AVX-CFLAGS)
ifeq (yes,$(config-cflags-avx512))
AVX512-CFLAGS = -mavx512f
CFLAGS-tst-audit10-aux.c += $(AVX512-CFLAGS)
CFLAGS-tst-auditmod10a.c += $(AVX512-CFLAGS)
CFLAGS-tst-auditmod10b.c += $(AVX512-CFLAGS)
CFLAGS-tst-avx512-aux.c += $(AVX512-CFLAGS)
CFLAGS-tst-avx512mod.c += $(AVX512-CFLAGS)
endif

$(objpfx)tst-glibc-hwcaps: $(objpfx)libmarkermod2-1.so \
  $(objpfx)libmarkermod3-1.so $(objpfx)libmarkermod4-1.so
$(objpfx)tst-glibc-hwcaps.out: \
  $(objpfx)libmarkermod2.so \
    $(objpfx)glibc-hwcaps/x86-64-v2/libmarkermod2.so \
  $(objpfx)libmarkermod3.so \
    $(objpfx)glibc-hwcaps/x86-64-v2/libmarkermod3.so \
    $(objpfx)glibc-hwcaps/x86-64-v3/libmarkermod3.so \
  $(objpfx)libmarkermod4.so \
    $(objpfx)glibc-hwcaps/x86-64-v2/libmarkermod4.so \
    $(objpfx)glibc-hwcaps/x86-64-v3/libmarkermod4.so \
    $(objpfx)glibc-hwcaps/x86-64-v4/libmarkermod4.so \

$(objpfx)glibc-hwcaps/x86-64-v2/libmarkermod2.so: $(objpfx)libmarkermod2-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v2/libmarkermod3.so: $(objpfx)libmarkermod3-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v3/libmarkermod3.so: $(objpfx)libmarkermod3-3.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v2/libmarkermod4.so: $(objpfx)libmarkermod4-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v3/libmarkermod4.so: $(objpfx)libmarkermod4-3.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v4/libmarkermod4.so: $(objpfx)libmarkermod4-4.so
	$(make-target-directory)
	cp $< $@

ifeq (no,$(build-hardcoded-path-in-tests))
# This is an ld.so.cache test, and RPATH/RUNPATH in the executable
# interferes with its test objectives.
tests-container += tst-glibc-hwcaps-cache
endif

tests-internal += tst-x86-64-tls-1

endif # $(subdir) == elf

ifeq ($(subdir),csu)
gen-as-const-headers += tlsdesc.sym rtld-offsets.sym
endif

ifeq ($(subdir),wcsmbs)
tests += \
  tst-rsi-wcslen
endif

$(objpfx)x86_64/tst-x86_64mod-1.os: $(objpfx)tst-x86_64mod-1.os
	$(make-target-directory)
	rm -f $@
	ln $< $@

do-tests-clean common-mostlyclean: tst-x86_64-1-clean

.PHONY: tst-x86_64-1-clean
tst-x86_64-1-clean:
	-rm -rf $(objpfx)x86_64

$(objpfx)x86_64/tst-platformmod-2.os: $(objpfx)tst-platformmod-2.os
	$(make-target-directory)
	rm -f $@
	ln $< $@
