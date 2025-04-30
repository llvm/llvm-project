/* _Float128 overrides for building ldbl-128 as _Float128.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* This must be included before the function renames below.  */
#include <gmp.h>
#include <math.h>
#undef HUGE_VALL
#define HUGE_VALL HUGE_VAL_F128
#include <math/mul_splitl.h>

/* This must be included before the renames of types and macros from
   it.  */
#include <ieee754.h>

/* Renames derived from math_private.h.  */
#include <math_private.h>
#include <fenv_private.h>
#include <ieee754_float128.h>
#define ieee854_long_double_shape_type ieee854_float128_shape_type
#define ieee854_long_double ieee854_float128

#undef GET_LDOUBLE_LSW64
#undef GET_LDOUBLE_MSW64
#undef GET_LDOUBLE_WORDS64
#undef SET_LDOUBLE_LSW64
#undef SET_LDOUBLE_MSW64
#undef SET_LDOUBLE_WORDS64
#define GET_LDOUBLE_LSW64(x,y) GET_FLOAT128_LSW64 (x, y)
#define GET_LDOUBLE_MSW64(x,y) GET_FLOAT128_MSW64 (x, y)
#define GET_LDOUBLE_WORDS64(x,y,z) GET_FLOAT128_WORDS64 (x, y, z)
#define SET_LDOUBLE_LSW64(x,y) SET_FLOAT128_LSW64 (x, y)
#define SET_LDOUBLE_MSW64(x,y) SET_FLOAT128_MSW64 (x, y)
#define SET_LDOUBLE_WORDS64(x,y,z) SET_FLOAT128_WORDS64 (x, y, z)

#undef IEEE854_LONG_DOUBLE_BIAS
#define IEEE854_LONG_DOUBLE_BIAS IEEE854_FLOAT128_BIAS

#ifdef SET_RESTORE_ROUNDF128
# undef SET_RESTORE_ROUNDL
# define SET_RESTORE_ROUNDL(RM) SET_RESTORE_ROUNDF128 (RM)
#endif

#ifdef libc_feholdexcept_setroundf128
# undef libc_feholdexcept_setroundl
# define libc_feholdexcept_setroundl(ENV, RM)	\
  libc_feholdexcept_setroundf128 (ENV, RM)
#endif

#ifdef libc_feupdateenv_testf128
# undef libc_feupdateenv_testl
# define libc_feupdateenv_testl(ENV, EX) libc_feupdateenv_testf128 (ENV, EX)
#endif

/* misc macros from the header below.  */
#include <fix-fp-int-convert-overflow.h>
#undef FIX_LDBL_LONG_CONVERT_OVERFLOW
#undef FIX_LDBL_LLONG_CONVERT_OVERFLOW
#define FIX_LDBL_LONG_CONVERT_OVERFLOW FIX_FLT128_LONG_CONVERT_OVERFLOW
#define FIX_LDBL_LLONG_CONVERT_OVERFLOW FIX_FLT128_LLONG_CONVERT_OVERFLOW


/* float.h constants.  */
#include <float.h>
#undef LDBL_DIG
#undef LDBL_EPSILON
#undef LDBL_MANT_DIG
#undef LDBL_MAX
#undef LDBL_MAX_10_EXP
#undef LDBL_MAX_EXP
#undef LDBL_MIN
#undef LDBL_MIN_10_EXP
#undef LDBL_MIN_EXP
#undef LDBL_TRUE_MIN
#define LDBL_DIG FLT128_DIG
#define LDBL_EPSILON FLT128_EPSILON
#define LDBL_MANT_DIG FLT128_MANT_DIG
#define LDBL_MAX FLT128_MAX
#define LDBL_MAX_10_EXP FLT128_MAX_10_EXP
#define LDBL_MAX_EXP FLT128_MAX_EXP
#define LDBL_MIN FLT128_MIN
#define LDBL_MIN_10_EXP FLT128_MIN_10_EXP
#define LDBL_MIN_EXP FLT128_MIN_EXP
#define LDBL_TRUE_MIN FLT128_TRUE_MIN


/* math.h GNU constants.  */
#undef M_El
#undef M_LOG2El
#undef M_LOG10El
#undef M_LN2l
#undef M_LN10l
#undef M_PIl
#undef M_PI_2l
#undef M_PI_4l
#undef M_1_PIl
#undef M_2_PIl
#undef M_2_SQRTPIl
#undef M_SQRT2l
#undef M_SQRT1_2l
#define M_El M_Ef128
#define M_LOG2El M_LOG2Ef128
#define M_LOG10El M_LOG10Ef128
#define M_LN2l M_LN2f128
#define M_LN10l M_LN10f128
#define M_PIl M_PIf128
#define M_PI_2l M_PI_2f128
#define M_PI_4l M_PI_4f128
#define M_1_PIl M_1_PIf128
#define M_2_PIl M_2_PIf128
#define M_2_SQRTPIl M_2_SQRTPIf128
#define M_SQRT2l M_SQRT2f128
#define M_SQRT1_2l M_SQRT1_2f128


#include <libm-alias-ldouble.h>
#include <libm-alias-float128.h>
#undef libm_alias_ldouble_r
#define libm_alias_ldouble_r(from, to, r) libm_alias_float128_r (from, to, r)


#include <math/math-narrow.h>
#undef libm_alias_float_ldouble
#define libm_alias_float_ldouble(func) libm_alias_float32_float128 (func)
#undef libm_alias_double_ldouble
#define libm_alias_double_ldouble(func) libm_alias_float64_float128 (func)

#include <math-use-builtins.h>
#undef USE_NEARBYINTL_BUILTIN
#define USE_NEARBYINTL_BUILTIN USE_NEARBYINTF128_BUILTIN
#undef USE_RINTL_BUILTIN
#define USE_RINTL_BUILTIN USE_RINTF128_BUILTIN
#undef USE_FLOORL_BUILTIN
#define USE_FLOORL_BUILTIN USE_FLOORF128_BUILTIN
#undef USE_CEILL_BUILTIN
#define USE_CEILL_BUILTIN USE_CEILF128_BUILTIN
#undef USE_TRUNCL_BUILTIN
#define USE_TRUNCL_BUILTIN USE_TRUNCF128_BUILTIN
#undef USE_ROUNDL_BUILTIN
#define USE_ROUNDL_BUILTIN USE_ROUNDF128_BUILTIN
#undef USE_ROUNDEVENL_BUILTIN
#define USE_ROUNDEVENL_BUILTIN USE_ROUNDEVENF128_BUILTIN
#undef USE_COPYSIGNL_BUILTIN
#define USE_COPYSIGNL_BUILTIN USE_COPYSIGNF128_BUILTIN
#undef USE_FMAL_BUILTIN
#define USE_FMAL_BUILTIN USE_FMAF128_BUILTIN

/* IEEE function renames.  */
#define __ieee754_acoshl __ieee754_acoshf128
#define __ieee754_acosl __ieee754_acosf128
#define __ieee754_asinhl __ieee754_asinhf128
#define __ieee754_asinl __ieee754_asinf128
#define __ieee754_atan2l __ieee754_atan2f128
#define __ieee754_atanhl __ieee754_atanhf128
#define __ieee754_coshl __ieee754_coshf128
#define __ieee754_cosl __ieee754_cosf128
#define __ieee754_exp10l __ieee754_exp10f128
#define __ieee754_exp2l __ieee754_exp2f128
#define __ieee754_expl __ieee754_expf128
#define __ieee754_fmodl __ieee754_fmodf128
#define __ieee754_gammal_r __ieee754_gammaf128_r
#define __ieee754_hypotl __ieee754_hypotf128
#define __ieee754_ilogbl __ieee754_ilogbf128
#define __ieee754_j0l __ieee754_j0f128
#define __ieee754_j1l __ieee754_j1f128
#define __ieee754_jnl __ieee754_jnf128
#define __ieee754_lgammal_r __ieee754_lgammaf128_r
#define __ieee754_log10l __ieee754_log10f128
#define __ieee754_log2l __ieee754_log2f128
#define __ieee754_logl __ieee754_logf128
#define __ieee754_powl __ieee754_powf128
#define __ieee754_rem_pio2l __ieee754_rem_pio2f128
#define __ieee754_remainderl __ieee754_remainderf128
#define __ieee754_sinhl __ieee754_sinhf128
#define __ieee754_sqrtl __ieee754_sqrtf128
#define __ieee754_y0l __ieee754_y0f128
#define __ieee754_y1l __ieee754_y1f128
#define __ieee754_ynl __ieee754_ynf128


/* finite math entry points.  */
#define __acoshl_finite __acoshf128_finite
#define __acosl_finite __acosf128_finite
#define __asinl_finite __asinf128_finite
#define __atan2l_finite __atan2f128_finite
#define __atanhl_finite __atanhf128_finite
#define __coshl_finite __coshf128_finite
#define __cosl_finite __cosf128_finite
#define __exp10l_finite __exp10f128_finite
#define __exp2l_finite __exp2f128_finite
#define __expl_finite __expf128_finite
#define __fmodl_finite __fmodf128_finite
#define __hypotl_finite __hypotf128_finite
#define __ilogbl_finite __ilogbf128_finite
#define __j0l_finite __j0f128_finite
#define __j1l_finite __j1f128_finite
#define __jnl_finite __jnf128_finite
#define __lgammal_r_finite __lgammaf128_r_finite
#define __log10l_finite __log10f128_finite
#define __log2l_finite __log2f128_finite
#define __logl_finite __logf128_finite
#define __powl_finite __powf128_finite
#define __remainderl_finite __remainderf128_finite
#define __sinhl_finite __sinhf128_finite
#define __y0l_finite __y0f128_finite
#define __y1l_finite __y1f128_finite
#define __ynl_finite __ynf128_finite


/* internal function names.  */
#define __asinhl __asinhf128
#define __atanl __atanf128
#define __cbrtl __cbrtf128
#define __ceill __ceilf128
#define __copysignl __copysignf128
#define __cosl __cosf128
#define __erfcl __erfcf128
#define __erfl __erff128
#define __expl __expf128
#define __expm1l __expm1f128
#define __fabsl __fabsf128
#define __fdiml __fdimf128
#define __finitel __finitef128
#define __floorl __floorf128
#define __fmal __fmaf128
#define __fmaxl __fmaxf128
#define __fminl __fminf128
#define __fpclassifyl __fpclassifyf128
#define __frexpl __frexpf128
#define __gammal_r_finite __gammaf128_r_finite
#define __getpayloadl __getpayloadf128
#define __isinfl __isinff128
#define __isnanl __isnanf128
#define __issignalingl __issignalingf128
#define __ldexpl __ldexpf128
#define __llrintl __llrintf128
#define __llroundl __llroundf128
#define __log1pl __log1pf128
#define __logbl __logbf128
#define __logl __logf128
#define __lrintl __lrintf128
#define __lroundl __lroundf128
#define __modfl __modff128
#define __nearbyintl __nearbyintf128
#define __nextafterl __nextafterf128
#define __nextdownl __nextdownf128
#define __nextupl __nextupf128
#define __remquol __remquof128
#define __rintl __rintf128
#define __roundevenl __roundevenf128
#define __roundl __roundf128
#define __scalblnl __scalblnf128
#define __scalbnl __scalbnf128
#define __signbitl __signbitf128
#define __sincosl __sincosf128
#define __sinl __sinf128
#define __sqrtl __sqrtf128
#define __tanhl __tanhf128
#define __tanl __tanf128
#define __totalorderl __totalorderf128
#define __totalorder_compatl __totalorder_compatf128
#define __totalordermagl __totalordermagf128
#define __totalordermag_compatl __totalordermag_compatf128
#define __truncl __truncf128
#define __x2y2m1l __x2y2m1f128

#define __faddl __f32addf128
#define __daddl __f64addf128
#define __fdivl __f32divf128
#define __ddivl __f64divf128
#define __fmull __f32mulf128
#define __dmull __f64mulf128
#define __fsubl __f32subf128
#define __dsubl __f64subf128

/* Used on __finite compat alias.  */
#define __acosl __acosf128
#define __acoshl __acoshf128
#define __asinl __asinf128
#define __atan2l __atan2f128
#define __atanhl __atanhf128
#define __coshl __coshf128
#define __exp10l __exp10f128
#define __expl __expf128
#define __fmodl __fmodf128
#define __gammal_r __gammaf128_r
#define __hypotl __hypotf128
#define __j0l __j0f128
#define __j1l __j1f128
#define __jnl __jnf128
#define __lgammal_r __lgammaf128_r
#define __log10l __log10f128
#define __log2l __log2f128
#define __logl __logf128
#define __powl __powf128
#define __remainderl __remainderf128
#define __sinhl __sinhf128
#define __y0l __y0f128
#define __y1l __y1f128
#define __ynl __ynf128

/* __nexttowardf128 is not _Float128 API. */
#define __nexttowardl __nexttowardf128_do_not_use
#define nexttowardl nexttowardf128_do_not_use


/* public entry points.  */
#define asinhl asinhf128
#define atanl atanf128
#define cbrtl cbrtf128
#define ceill ceilf128
#define copysignl copysignf128
#define cosl cosf128
#define erfcl erfcf128
#define erfl erff128
#define expl expf128
#define expm1l expm1f128
#define fabsl fabsf128
#define fdiml fdimf128
#define finitel finitef128_do_not_use
#define floorl floorf128
#define fmal fmaf128
#define fmaxl fmaxf128
#define fminl fminf128
#define frexpl frexpf128
#define getpayloadl getpayloadf128
#define isinfl isinff128_do_not_use
#define isnanl isnanf128_do_not_use
#define ldexpl ldexpf128
#define llrintl llrintf128
#define llroundl llroundf128
#define log1pl log1pf128
#define logbl logbf128
#define logl logf128
#define lrintl lrintf128
#define lroundl lroundf128
#define modfl modff128
#define nanl nanf128
#define nearbyintl nearbyintf128
#define nextafterl nextafterf128
#define nextdownl nextdownf128
#define nextupl nextupf128
#define remquol remquof128
#define rintl rintf128
#define roundevenl roundevenf128
#define roundl roundf128
#define scalbnl scalbnf128
#define sincosl sincosf128
#define sinl sinf128
#define sqrtl sqrtf128
#define tanhl tanhf128
#define tanl tanf128
#define totalorderl totalorderf128
#define totalordermagl totalordermagf128
#define truncl truncf128


/* misc internal renames.  */
#define __builtin_fmal __builtin_fmaf128
#define __expl_table __expf128_table
#define __gamma_productl __gamma_productf128
#define __kernel_cosl __kernel_cosf128
#define __kernel_rem_pio2l __kernel_rem_pio2f128
#define __kernel_sincosl __kernel_sincosf128
#define __kernel_sinl __kernel_sinf128
#define __kernel_tanl __kernel_tanf128
#define __lgamma_negl __lgamma_negf128
#define __lgamma_productl __lgamma_productf128
#define __mpn_extract_long_double __mpn_extract_float128
#define __sincosl_table __sincosf128_table
#define mul_splitl mul_splitf128

/* Builtin renames.  */
#define __builtin_copysignl __builtin_copysignf128
#define __builtin_signbitl __builtin_signbit
#define __builtin_nearbyintl __builtin_nearbyintf128
#define __builtin_rintl __builtin_rintf128
#define __builtin_floorl __builtin_floorf128
#define __builtin_ceill __builtin_ceilf128
#define __builtin_truncl __builtin_truncf128
#define __builtin_roundl __builtin_roundf128
#define __builtin_copysignl __builtin_copysignf128

/* Get the constant suffix from bits/floatn-compat.h.  */
#define L(x) __f128 (x)

static inline void
mul_splitf128 (_Float128 *hi, _Float128 *lo, _Float128 x, _Float128 y)
{
#ifdef __FP_FAST_FMAF128
  /* Fast built-in fused multiply-add.  */
  *hi = x * y;
  *lo = __builtin_fmal (x, y, -*hi);
#else
  /* Apply Dekker's algorithm.  */
  *hi = x * y;
# define C ((1LL << (FLT128_MANT_DIG + 1) / 2) + 1)
  _Float128 x1 = x * C;
  _Float128 y1 = y * C;
# undef C
  x1 = (x - x1) + x1;
  y1 = (y - y1) + y1;
  _Float128 x2 = x - x1;
  _Float128 y2 = y - y1;
  *lo = (((x1 * y1 - *hi) + x1 * y2) + x2 * y1) + x2 * y2;
#endif
}
