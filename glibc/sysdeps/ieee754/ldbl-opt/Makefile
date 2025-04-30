# The`long double' type is a distinct type we support if
# -mlong-double-128 option is used (or when it becomes a default
# when -mlong-double-64 is not used).
long-double-fcts = yes
ifeq (,$(filter -mlong-double-128,$(sysdep-CFLAGS)))
sysdep-CFLAGS += -mlong-double-128
endif

ifeq ($(subdir),math)
libm-routines += s_nexttowardfd
routines += nldbl-compat

extra-libs += libnldbl
libnldbl-calls = asprintf dprintf fprintf fscanf fwprintf fwscanf iovfscanf \
		 obstack_printf obstack_vprintf printf scanf snprintf \
		 sprintf sscanf swprintf swscanf vasprintf vdprintf vfprintf \
		 vfscanf vfwprintf vfwscanf vprintf vscanf vsnprintf \
		 vsprintf vsscanf vswprintf vswscanf vwprintf vwscanf \
		 wprintf wscanf printf_fp printf_size \
		 fprintf_chk fwprintf_chk printf_chk snprintf_chk sprintf_chk \
		 swprintf_chk vfprintf_chk vfwprintf_chk vprintf_chk \
		 vsnprintf_chk vsprintf_chk vswprintf_chk vwprintf_chk \
		 wprintf_chk asprintf_chk vasprintf_chk dprintf_chk \
		 vdprintf_chk obstack_printf_chk obstack_vprintf_chk \
		 syslog syslog_chk vsyslog vsyslog_chk \
		 strfmon strfmon_l \
		 strfroml \
		 strtold strtold_l strtoldint wcstold wcstold_l wcstoldint \
		 qecvt qfcvt qgcvt qecvt_r qfcvt_r \
		 isinf isnan finite signbit scalb log2 lgamma_r ceil \
		 significand acos asin atan atan2 cos sin tan cosh sinh \
		 tanh acosh asinh atanh exp log log10 exp10 expm1 \
		 log1p logb exp2 sqrt cbrt fabs floor j0 j1 y0 y1 erf erfc \
		 lgamma tgamma gamma rint nearbyint round trunc \
		 copysign fdim fmax fmin nextafter pow hypot fmod \
		 remainder ldexp scalbn frexp modf scalbln fma nan sincos \
		 jn yn ilogb remquo lrint lround llrint llround nexttowardf \
		 nexttoward conj cacos cacosh casin catan catanh ccos ccosh \
		 casinh cexp clog cproj csin csinh csqrt ctan ctanh cpow \
		 cabs carg cimag creal clog10 \
		 isoc99_scanf isoc99_fscanf isoc99_sscanf \
		 isoc99_vscanf isoc99_vfscanf isoc99_vsscanf \
		 isoc99_wscanf isoc99_fwscanf isoc99_swscanf \
		 isoc99_vwscanf isoc99_vfwscanf isoc99_vswscanf \
		 nextup nextdown totalorder totalordermag getpayload \
		 canonicalize setpayload setpayloadsig llogb fmaxmag fminmag \
		 roundeven fromfp ufromfp fromfpx ufromfpx fadd dadd \
		 fdiv ddiv fmul dmul fsub dsub
libnldbl-routines = $(libnldbl-calls:%=nldbl-%)
libnldbl-inhibit-o = $(object-suffixes)
libnldbl-static-only-routines = $(libnldbl-routines)
extra-objs += $(addsuffix .oS, $(libnldbl-routines))

CFLAGS-nldbl-acos.c = -fno-builtin-acosl
CFLAGS-nldbl-acosh.c = -fno-builtin-acoshl
CFLAGS-nldbl-asin.c = -fno-builtin-asinl
CFLAGS-nldbl-asinh.c = -fno-builtin-asinhl
CFLAGS-nldbl-atan.c = -fno-builtin-atanl
CFLAGS-nldbl-atan2.c = -fno-builtin-atan2l
CFLAGS-nldbl-atanh.c = -fno-builtin-atanhl
CFLAGS-nldbl-cabs.c = -fno-builtin-cabsl
CFLAGS-nldbl-cacos.c = -fno-builtin-cacosl
CFLAGS-nldbl-cacosh.c = -fno-builtin-cacoshl
CFLAGS-nldbl-canonicalize.c = -fno-builtin-canonicalizel
CFLAGS-nldbl-carg.c = -fno-builtin-cargl
CFLAGS-nldbl-casin.c = -fno-builtin-casinl
CFLAGS-nldbl-casinh.c = -fno-builtin-casinhl
CFLAGS-nldbl-catan.c = -fno-builtin-catanl
CFLAGS-nldbl-catanh.c = -fno-builtin-catanhl
CFLAGS-nldbl-cbrt.c = -fno-builtin-cbrtl
CFLAGS-nldbl-ccos.c = -fno-builtin-ccosl
CFLAGS-nldbl-ccosh.c = -fno-builtin-ccoshl
CFLAGS-nldbl-ceil.c = -fno-builtin-ceill
CFLAGS-nldbl-cexp.c = -fno-builtin-cexpl
CFLAGS-nldbl-cimag.c = -fno-builtin-cimagl
CFLAGS-nldbl-clog.c = -fno-builtin-clogl
CFLAGS-nldbl-clog10.c = -fno-builtin-clog10l
CFLAGS-nldbl-conj.c = -fno-builtin-conjl
CFLAGS-nldbl-copysign.c = -fno-builtin-copysignl
CFLAGS-nldbl-cos.c = -fno-builtin-cosl
CFLAGS-nldbl-cosh.c = -fno-builtin-coshl
CFLAGS-nldbl-cpow.c = -fno-builtin-cpowl
CFLAGS-nldbl-cproj.c = -fno-builtin-cprojl
CFLAGS-nldbl-creal.c = -fno-builtin-creall
CFLAGS-nldbl-csin.c = -fno-builtin-csinl
CFLAGS-nldbl-csinh.c = -fno-builtin-csinhl
CFLAGS-nldbl-csqrt.c = -fno-builtin-csqrtl
CFLAGS-nldbl-ctan.c = -fno-builtin-ctanl
CFLAGS-nldbl-ctanh.c = -fno-builtin-ctanhl
CFLAGS-nldbl-dadd.c = -fno-builtin-daddl
CFLAGS-nldbl-ddiv.c = -fno-builtin-ddivl
CFLAGS-nldbl-dmul.c = -fno-builtin-dmull
CFLAGS-nldbl-dsub.c = -fno-builtin-dsubl
CFLAGS-nldbl-erf.c = -fno-builtin-erfl
CFLAGS-nldbl-erfc.c = -fno-builtin-erfcl
CFLAGS-nldbl-exp.c = -fno-builtin-expl
CFLAGS-nldbl-exp10.c = -fno-builtin-exp10l
CFLAGS-nldbl-exp2.c = -fno-builtin-exp2l
CFLAGS-nldbl-expm1.c = -fno-builtin-expm1l
CFLAGS-nldbl-fabs.c = -fno-builtin-fabsl
CFLAGS-nldbl-fadd.c = -fno-builtin-faddl
CFLAGS-nldbl-fdim.c = -fno-builtin-fdiml
CFLAGS-nldbl-fdiv.c = -fno-builtin-fdivl
CFLAGS-nldbl-finite.c = -fno-builtin-finitel
CFLAGS-nldbl-floor.c = -fno-builtin-floorl
CFLAGS-nldbl-fma.c = -fno-builtin-fmal
CFLAGS-nldbl-fmax.c = -fno-builtin-fmaxl
CFLAGS-nldbl-fmaxmag.c = -fno-builtin-fmaxmagl
CFLAGS-nldbl-fmin.c = -fno-builtin-fminl
CFLAGS-nldbl-fminmag.c = -fno-builtin-fminmagl
CFLAGS-nldbl-fmod.c = -fno-builtin-fmodl
CFLAGS-nldbl-fmul.c = -fno-builtin-fmull
CFLAGS-nldbl-frexp.c = -fno-builtin-frexpl
CFLAGS-nldbl-fromfp.c = -fno-builtin-fromfpl
CFLAGS-nldbl-fromfpx.c = -fno-builtin-fromfpxl
CFLAGS-nldbl-fsub.c = -fno-builtin-fsubl
CFLAGS-nldbl-gamma.c = -fno-builtin-gammal
CFLAGS-nldbl-getpayload.c = -fno-builtin-getpayloadl
CFLAGS-nldbl-hypot.c = -fno-builtin-hypotl
CFLAGS-nldbl-ilogb.c = -fno-builtin-ilogbl
CFLAGS-nldbl-isinf.c = -fno-builtin-isinfl
CFLAGS-nldbl-isnan.c = -fno-builtin-isnanl
CFLAGS-nldbl-j0.c = -fno-builtin-j0l
CFLAGS-nldbl-j1.c = -fno-builtin-j1l
CFLAGS-nldbl-jn.c = -fno-builtin-jnl
CFLAGS-nldbl-ldexp.c = -fno-builtin-ldexpl
CFLAGS-nldbl-lgamma.c = -fno-builtin-lgammal
CFLAGS-nldbl-lgamma_r.c = -fno-builtin-lgammal_r
CFLAGS-nldbl-llogb.c = -fno-builtin-llogbl
CFLAGS-nldbl-llrint.c = -fno-builtin-llrintl
CFLAGS-nldbl-llround.c = -fno-builtin-llroundl
CFLAGS-nldbl-log.c = -fno-builtin-logl
CFLAGS-nldbl-log10.c = -fno-builtin-log10l
CFLAGS-nldbl-log1p.c = -fno-builtin-log1pl
CFLAGS-nldbl-log2.c = -fno-builtin-log2l
CFLAGS-nldbl-logb.c = -fno-builtin-logbl
CFLAGS-nldbl-lrint.c = -fno-builtin-lrintl
CFLAGS-nldbl-lround.c = -fno-builtin-lroundl
CFLAGS-nldbl-modf.c = -fno-builtin-modfl
CFLAGS-nldbl-nan.c = -fno-builtin-nanl
CFLAGS-nldbl-nearbyint.c = -fno-builtin-nearbyintl
CFLAGS-nldbl-nextafter.c = -fno-builtin-nextafterl
CFLAGS-nldbl-nextdown.c = -fno-builtin-nextdownl
CFLAGS-nldbl-nexttoward.c = -fno-builtin-nexttoward -fno-builtin-nexttowardl
CFLAGS-nldbl-nexttowardf.c = -fno-builtin-nexttowardf
CFLAGS-nldbl-nextup.c = -fno-builtin-nextupl
CFLAGS-nldbl-pow.c = -fno-builtin-powl
CFLAGS-nldbl-remainder.c = -fno-builtin-remainderl -fno-builtin-dreml
CFLAGS-nldbl-remquo.c = -fno-builtin-remquol
CFLAGS-nldbl-rint.c = -fno-builtin-rintl
CFLAGS-nldbl-round.c = -fno-builtin-roundl
CFLAGS-nldbl-roundeven.c = -fno-builtin-roundevenl
CFLAGS-nldbl-scalb.c = -fno-builtin-scalbl
CFLAGS-nldbl-scalbln.c = -fno-builtin-scalblnl
CFLAGS-nldbl-scalbn.c = -fno-builtin-scalbnl
CFLAGS-nldbl-setpayload.c = -fno-builtin-setpayloadl
CFLAGS-nldbl-setpayloadsig.c = -fno-builtin-setpayloadsigl
CFLAGS-nldbl-significand.c = -fno-builtin-significandl
CFLAGS-nldbl-sin.c = -fno-builtin-sinl
CFLAGS-nldbl-sincos.c = -fno-builtin-sincosl
CFLAGS-nldbl-sinh.c = -fno-builtin-sinhl
CFLAGS-nldbl-sqrt.c = -fno-builtin-sqrtl
CFLAGS-nldbl-tan.c = -fno-builtin-tanl
CFLAGS-nldbl-tanh.c = -fno-builtin-tanhl
CFLAGS-nldbl-tgamma.c = -fno-builtin-tgammal
CFLAGS-nldbl-totalorder.c = -fno-builtin-totalorderl
CFLAGS-nldbl-totalordermag.c = -fno-builtin-totalordermagl
CFLAGS-nldbl-trunc.c = -fno-builtin-truncl
CFLAGS-nldbl-ufromfp.c = -fno-builtin-ufromfpl
CFLAGS-nldbl-ufromfpx.c = -fno-builtin-ufromfpxl
CFLAGS-nldbl-y0.c = -fno-builtin-y0l
CFLAGS-nldbl-y1.c = -fno-builtin-y1l
CFLAGS-nldbl-yn.c = -fno-builtin-ynl

tests += test-narrow-macros-ldbl-64 test-nldbl-redirect \
	 test-redirection-ldbl-64
CFLAGS-test-narrow-macros-ldbl-64.c += -mlong-double-64
CFLAGS-test-nldbl-redirect.c += -mlong-double-64
CFLAGS-test-redirection-ldbl-64.c += -mlong-double-64

$(objpfx)test-nldbl-redirect: $(objpfx)libnldbl_nonshared.a

endif

# Tests for argp.h functions (reusing the relevant tests from argp/)
ifeq ($(subdir),argp)
tests-internal += tst-nldbl-argp

$(objpfx)tst-nldbl-argp.c: tst-ldbl-argp.c
	cp $< $@

CFLAGS-tst-nldbl-argp.c += -mlong-double-64
endif

# Tests for err.h and error.h functions (reusing the relevant tests from
# misc/).
ifeq ($(subdir), misc)
tests-internal += tst-nldbl-warn
tests-internal += tst-nldbl-error

$(objpfx)tst-nldbl-warn.c: tst-ldbl-warn.c
	cp $< $@

$(objpfx)tst-nldbl-error.c: tst-ldbl-error.c
	cp $< $@

CFLAGS-tst-nldbl-warn.c += -mlong-double-64
CFLAGS-tst-nldbl-error.c += -mlong-double-64
endif
