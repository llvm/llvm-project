# When building float128 we need to ensure -mfloat128 is
# passed to all such object files.
type-float128-CFLAGS := -mfloat128

# Bootstrapping code for enabling IEEE 128.  This can be removed and
# any indirections simplified once IEEE 128 long double is enabled.
type-ldouble-CFLAGS =
no-gnu-attribute-CFLAGS =
ifeq ($(ibm128-fcts),yes)
ibm128-abi-CFLAGS := -mabi=ibmlongdouble
type-ldouble-CFLAGS += $(ibm128-abi-CFLAGS)
no-gnu-attribute-CFLAGS = -mno-gnu-attribute

# This is a very gnarly workaround to static libgcc providing a .gnu.attribute
# in ibm128 functions.  This is not a bug in static libgcc as normal users
# should only ever link ibm128 or ieee128 routines, never both.  We are the
# exception which allows the decision to be deferred to user libraries or
# applications.
#
# We built an object file and extract the desired .gnu.attributes section
# to be inserted into lib{c,m}.so.
ifeq ($(build-shared),yes)

# Build everything with IEEE 128-bit long double.
sysdep-CFLAGS += -mabi=ieeelongdouble -Wno-psabi $(no-gnu-attribute-CFLAGS)

before-compile += $(common-objpfx)no_ldbl_gnu_attribute.bin

$(common-objpfx)no_ldbl_gnu_attribute.bin: $(..)sysdeps/powerpc/powerpc64/le/no_ldbl_gnu_attribute.c
	$(filter-out $(no-gnu-attribute-CFLAGS),$(compile.c)) -o $(basename $@).o
	$(OBJCOPY) --dump-section=.gnu.attributes=$@.tmp $(basename $@).o
	mv $@.tmp $@

define after-link
if [ ! -z "$(filter libm.so libc.so,$(notdir $1))" ]; then \
  echo "Replacing .gnu.attributes in $1"; \
  $(OBJCOPY) --update-section=.gnu.attributes=$(common-objpfx)no_ldbl_gnu_attribute.bin $1; \
fi
endef

# Ensure a .gnu.attributes section is present by building an ibm128 file with
# -mgnu-attribute in both libm and libc shared libraries.  Prior to GCC 11 this
# section was implicitly included via static libgcc ibm128 routines.
ifeq ($(subdir),math)
$(objpfx)m_ldexpl.os: CFLAGS += -mgnu-attribute
$(objpfx)s_ldexpl.os: CFLAGS += -mgnu-attribute
endif

endif # ifeq ($(build-shared),yes)
endif # ifeq ($(ibm128-fcts),yes)

# All _Float128 files should be built assuming an ibm128 long double.
# Likewise, this avoids some old GCC 7 compiler bugs whereby calls
# to __mulkc3 are swapped with __multc3, or worse.  This is noted in
# GCC BZ 84914, and marked as will-not-fix.
type-float128-CFLAGS += $(type-ldouble-CFLAGS)

ifeq ($(subdir),math)
# sqrtf128 requires emulation before POWER9.
CPPFLAGS += -I../soft-fp

# float128 requires adding a handful of extra flags.
# Similarly, disable
$(foreach suf,$(all-object-suffixes),%f128$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
$(foreach suf,$(all-object-suffixes),%f128_r$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
$(foreach suf,$(all-object-suffixes),$(objpfx)test-float128%$(suf)): CFLAGS += $(type-float128-CFLAGS)
$(foreach suf,$(all-object-suffixes),$(objpfx)test-float64x%$(suf)): CFLAGS += $(type-float128-CFLAGS)
# Pairs of types with _Float128 / _Float64x as the wider type but not
# the narrower one.
f128-pairs = float32-float64x float32-float128 float64-float64x \
	     float64-float128 float32x-float64x float32x-float128
$(foreach suf,$(all-object-suffixes),$(foreach pair,$(f128-pairs),$(objpfx)test-$(pair)%$(suf))): CFLAGS += $(type-float128-CFLAGS)
CFLAGS-libm-test-support-float128.c += $(type-float128-CFLAGS)
CFLAGS-libm-test-support-float64x.c += $(type-float128-CFLAGS)
CFLAGS-test-math-iscanonical.cc += $(type-float128-CFLAGS)
CFLAGS-test-math-iseqsig.cc += $(type-float128-CFLAGS)
CFLAGS-test-math-issignaling.cc += $(type-float128-CFLAGS)
CFLAGS-test-math-iszero.cc += $(type-float128-CFLAGS)

CFLAGS-s_logbl-power7.c += $(type-ldouble-CFLAGS)
CFLAGS-s_logbl-ppc64.c += $(type-ldouble-CFLAGS)

$(foreach suf,$(all-object-suffixes),\
          $(objpfx)s_copysignl$(suf) \
          $(objpfx)s_fabsl$(suf)): \
  ASFLAGS += $(type-ldouble-CFLAGS)

$(foreach suf,$(all-object-suffixes),\
         $(objpfx)libm-test-%ibm128$(suf) \
         $(objpfx)test-iibm128%$(suf) $(objpfx)test-ibm128%$(suf)): \
  CFLAGS += $(type-ldouble-CFLAGS)

# Newer GCC (>7) doesn't like -mabi=* and -mlong-double-64
$(foreach suf,$(all-object-suffixes),\
         $(objpfx)test-narrow-macros-ldbl-64$(suf) \
         $(objpfx)test-nldbl-redirect$(suf) \
         $(objpfx)test-redirection-ldbl-64$(suf) \
         ): sysdep-CFLAGS := $(filter-out -mabi=ieeelongdouble,$(sysdep-CFLAGS))

endif

# Newer GCC (>7) doesn't like -mabi=* and -mlong-double-64
ifeq ($(subdir),misc)
$(foreach suf,$(all-object-suffixes),\
         $(objpfx)tst-nldbl-warn$(suf) \
         $(objpfx)tst-nldbl-error$(suf) \
         ): sysdep-CFLAGS := $(filter-out -mabi=ieeelongdouble,$(sysdep-CFLAGS))
endif

# Newer GCC (>7) doesn't like -mabi=* and -mlong-double-64
ifeq ($(subdir),argp)
$(foreach suf,$(all-object-suffixes),\
         $(objpfx)tst-nldbl-argp$(suf) \
         ): sysdep-CFLAGS := $(filter-out -mabi=ieeelongdouble,$(sysdep-CFLAGS))
endif


# Append flags to string <-> _Float128 routines.
ifneq ($(filter $(subdir),wcsmbs stdlib),)
$(foreach suf,$(all-object-suffixes),%f128$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
$(foreach suf,$(all-object-suffixes),%f128_l$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
$(foreach suf,$(all-object-suffixes),%f128_nan$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
$(foreach suf,$(all-object-suffixes),%float1282mpn$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
$(foreach suf,$(all-object-suffixes),%mpn2float128$(suf)): CFLAGS += $(type-float128-CFLAGS) $(no-gnu-attribute-CFLAGS)
CFLAGS-bug-strtod.c += $(type-float128-CFLAGS)
CFLAGS-bug-strtod2.c += $(type-float128-CFLAGS)
CFLAGS-tst-strtod-round.c += $(type-float128-CFLAGS)
CFLAGS-tst-wcstod-round.c += $(type-float128-CFLAGS)
CFLAGS-tst-strtod-nan-locale.c += $(type-float128-CFLAGS)
CFLAGS-tst-wcstod-nan-locale.c += $(type-float128-CFLAGS)
CFLAGS-tst-strtod6.c += $(type-float128-CFLAGS)
CFLAGS-tst-strfrom.c += $(type-float128-CFLAGS)
CFLAGS-tst-strfrom-locale.c += $(type-float128-CFLAGS)
CFLAGS-strfrom-skeleton.c += $(type-float128-CFLAGS)
CFLAGS-tst-strtod-nan-sign.c += $(type-float128-CFLAGS)
CFLAGS-tst-wcstod-nan-sign.c += $(type-float128-CFLAGS)

# When building glibc with support for _Float128, the powers of ten tables in
# fpioconst.c and in the string conversion functions must be extended.  Some
# Makefiles (e.g.: wcsmbs/Makefile) override CFLAGS defined by the Makefiles in
# sysdeps.  This is avoided with the use sysdep-CFLAGS instead of CFLAGS.
sysdep-CFLAGS += $(sysdep-CFLAGS-$(<F))
sysdep-CFLAGS-fpioconst.c += $(type-float128-CFLAGS)
sysdep-CFLAGS-strtod_l.c += $(type-float128-CFLAGS)
sysdep-CFLAGS-strtof_l.c += $(type-float128-CFLAGS)
sysdep-CFLAGS-strtold_l.c += $(type-float128-CFLAGS)
sysdep-CFLAGS-wcstod_l.c += $(type-float128-CFLAGS)
sysdep-CFLAGS-wcstof_l.c += $(type-float128-CFLAGS)
sysdep-CFLAGS-wcstold_l.c += $(type-float128-CFLAGS)

# Build any ibm128 specific stdlib tests with ibm128 ABI.
$(foreach suf,$(all-object-suffixes),%128ibm$(suf)): CFLAGS += $(type-ldouble-CFLAGS)
endif

# Append flags to printf routines.
ifeq ($(subdir),stdio-common)
CFLAGS-printf_fp.c = $(type-float128-CFLAGS)
CFLAGS-printf_fphex.c = $(type-float128-CFLAGS)
CFLAGS-printf_size.c = $(type-float128-CFLAGS)
endif

# Disable linker noise on files using ieee128 long double internally
ifeq ($(subdir),time)
CFLAGS-difftime.c += $(no-gnu-attribute-CFLAGS)
endif

$(foreach suf,$(all-object-suffixes),$(objpfx)nldbl-%$(suf)): \
  CFLAGS += $(type-ldouble-CFLAGS)

ldbl-ibm128-files = $(objpfx)nldbl-%$(suf) \
		    $(objpfx)libm-test-%ibm128$(suf) \
		    $(objpfx)test-iibm128%$(suf) $(objpfx)test-ibm128%$(suf) \
		    $(objpfx)test-tgmath3-%$(suf)

# Remove -mabi=ieeelongdouble from ldbl-ibm128 files.
$(foreach suf,$(all-object-suffixes), $(ldbl-ibm128-files)) \
	  $(foreach r,$(ldbl-128ibm-routines) $(ldbl-tests), \
		    $(objpfx)$(r)$(suf)): \
  sysdep-CFLAGS := $(filter-out -mabi=ieeelongdouble,$(sysdep-CFLAGS))

# TODO: a bug in stdc++ will fail if -mabi=ieeelongdouble and -mfloat128
# are both enabled.  The latter is enabled by default in GCC 8+.  This is
# tracked via GCC BZ 94080.
ifeq ($(subdir),support)
CFLAGS-links-dso-program.cc += -mno-float128
endif
ifeq ($(subdir),nptl)
CFLAGS-tst-thread_local1.cc += -mno-float128
CFLAGS-tst-minstack-throw.cc += -mno-float128
endif

ifeq ($(subdir),elf)
$(objpfx)tst-glibc-hwcaps: \
  $(objpfx)libmarkermod2-1.so $(objpfx)libmarkermod3-1.so
$(objpfx)tst-glibc-hwcaps.out: \
  $(objpfx)libmarkermod2.so \
    $(objpfx)glibc-hwcaps/power9/libmarkermod2.so \
  $(objpfx)libmarkermod3.so \
    $(objpfx)glibc-hwcaps/power9/libmarkermod3.so \
    $(objpfx)glibc-hwcaps/power10/libmarkermod3.so \

$(objpfx)glibc-hwcaps/power9/libmarkermod2.so: $(objpfx)libmarkermod2-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/power9/libmarkermod3.so: $(objpfx)libmarkermod3-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/power10/libmarkermod3.so: $(objpfx)libmarkermod3-3.so
	$(make-target-directory)
	cp $< $@

ifeq (no,$(build-hardcoded-path-in-tests))
# This is an ld.so.cache test, and RPATH/RUNPATH in the executable
# interferes with its test objectives.
tests-container += tst-glibc-hwcaps-cache
endif

endif # $(subdir) == elf
