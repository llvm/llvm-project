/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#define ATTR __attribute__((always_inline))

#define FUNC1D(root) \
  ATTR double __nv_##root(double x) { return __ocml_##root##_f64(x); }
#define FUNC1F(root) \
  ATTR float __nv_##root##f(float x) { return __ocml_##root##_f32(x); }
#define FUNC1(root) FUNC1D(root) FUNC1F(root)

#define FUNC2D(root) \
  ATTR double __nv_##root(double x, double y) { return __ocml_##root##_f64(x, y); }
#define FUNC2F(root) \
  ATTR float __nv_##root##f(float x, float y) { return __ocml_##root##_f32(x, y); }
#define FUNC2(root) FUNC2D(root) FUNC2F(root)

#define FUNC3D(root) \
  ATTR double __nv_##root(double x, double y, double z) { return __ocml_##root##_f64(x, y, z); }
#define FUNC3F(root) \
  ATTR float __nv_##root##f(float x, float y, float z) { return __ocml_##root##_f32(x, y, z); }
#define FUNC3(root) FUNC3D(root) FUNC3F(root)

//-------- T __nv_acos
//-------- T __nv_acosf
FUNC1(acos)

//-------- T __nv_acosh
//-------- T __nv_acoshf
FUNC1(acosh)

//-------- T __nv_asin
//-------- T __nv_asinf
FUNC1(asin)

//-------- T __nv_asinh
//-------- T __nv_asinhf
FUNC1(asinh)

//-------- T __nv_atan
//-------- T __nv_atanf
FUNC1(atan)

//-------- T __nv_atan2
//-------- T __nv_atan2f
FUNC2(atan2)

//-------- T __nv_atanh
//-------- T __nv_atanhf
FUNC1(atanh)

//-------- T __nv_cbrt
//-------- T __nv_cbrtf
FUNC1(cbrt)

//-------- T __nv_ceil
//-------- T __nv_ceilf
FUNC1(ceil)

//-------- T __nv_copysign
//-------- T __nv_copysignf
FUNC2(copysign)

//-------- T __nv_cos
//-------- T __nv_cosf
FUNC1(cos)

//-------- T __nv_cosh
//-------- T __nv_coshf
FUNC1(cosh)

//-------- T __nv_cospi
//-------- T __nv_cospif
FUNC1(cospi)

//-------- T __nv_erf
//-------- T __nv_erff
FUNC1(erf)

//-------- T __nv_erfc
//-------- T __nv_erfcf
FUNC1(erfc)

//-------- T __nv_erfcinv
//-------- T __nv_erfcinvf
FUNC1(erfcinv)

//-------- T __nv_erfcx
//-------- T __nv_erfcxf
FUNC1(erfcx)

//-------- T __nv_erfinv
//-------- T __nv_erfinvf
FUNC1(erfinv)

//-------- T __nv_exp
//-------- T __nv_expf
FUNC1(exp)

//-------- T __nv_exp10
//-------- T __nv_exp10f
FUNC1(exp10)

//-------- T __nv_exp2
//-------- T __nv_exp2f
FUNC1(exp2)

//-------- T __nv_expm1
//-------- T __nv_expm1f
FUNC1(expm1)

//-------- T __nv_fabs
//-------- T __nv_fabsf
FUNC1(fabs)

//-------- T __nv_fdim
//-------- T __nv_fdimf
FUNC2(fdim)

//-------- T __nv_floor
//-------- T __nv_floorf
FUNC1(floor)

//-------- T __nv_fma
//-------- T __nv_fmaf
FUNC3(fma)

//-------- T __nv_fmax
//-------- T __nv_fmaxf
FUNC2(fmax)

//-------- T __nv_fmin
//-------- T __nv_fminf
FUNC2(fmin)

//-------- T __nv_fmod
//-------- T __nv_fmodf
FUNC2(fmod)

//-------- T __nv_hypot
//-------- T __nv_hypotf
FUNC2(hypot)

//-------- T __nv_j0
//-------- T __nv_j0f
FUNC1(j0)

//-------- T __nv_j1
//-------- T __nv_j1f
FUNC1(j1)

//-------- T __nv_lgamma
//-------- T __nv_lgammaf
FUNC1(lgamma)

//-------- T __nv_log
//-------- T __nv_logf
FUNC1(log)

//-------- T __nv_log10
//-------- T __nv_log10f
FUNC1(log10)

//-------- T __nv_log1p
//-------- T __nv_log1pf
FUNC1(log1p)

//-------- T __nv_log2
//-------- T __nv_log2f
FUNC1(log2)

//-------- T __nv_logb
//-------- T __nv_logbf
FUNC1(logb)

//-------- T __nv_pow
//-------- T __nv_powf
FUNC2(pow)

//-------- T __nv_rcbrt
//-------- T __nv_rcbrtf
FUNC1(rcbrt)

//-------- T __nv_remainder
//-------- T __nv_remainderf
FUNC2(remainder)

//-------- T __nv_rhypot
//-------- T __nv_rhypotf
FUNC2(rhypot)

//-------- T __nv_nearbyint
//-------- T __nv_nearbyintf
FUNC1(nearbyint)

//-------- T __nv_nextafter
//-------- T __nv_nextafterf
FUNC2(nextafter)

//-------- T __nv_rint
//-------- T __nv_rintf
FUNC1(rint)

//-------- T __nv_round
//-------- T __nv_roundf
FUNC1(round)

//-------- T __nv_rsqrt
//-------- T __nv_rsqrtf
FUNC1(rsqrt)

//-------- T __nv_scalbn
//-------- T __nv_scalbnf
FUNC2(scalbn)

//-------- T __nv_sin
//-------- T __nv_sinf
FUNC1(sin)

//-------- T __nv_sinh
//-------- T __nv_sinhf
FUNC1(sinh)

//-------- T __nv_sinpi
//-------- T __nv_sinpif
FUNC1(sinpi)

//-------- T __nv_sqrt
//-------- T __nv_sqrtf
FUNC1(sqrt)

//-------- T __nv_tan
//-------- T __nv_tanf
FUNC1(tan)

//-------- T __nv_tanh
//-------- T __nv_tanhf
FUNC1(tanh)

//-------- T __nv_tgamma
//-------- T __nv_tgammaf
FUNC1(tgamma)

//-------- T __nv_trunc
//-------- T __nv_truncf
FUNC1(trunc)

//-------- T __nv_y0
//-------- T __nv_y0f
FUNC1(y0)

//-------- T __nv_y1
//-------- T __nv_y1f
FUNC1(y1)

//-------- T __nv_cyl_bessel_i0
ATTR double __nv_cyl_bessel_i0(double x) { return __ocml_i0_f64(x); }

//-------- T __nv_cyl_bessel_i0f
ATTR float __nv_cyl_bessel_i0f(float x) { return __ocml_i0_f32(x); }

//-------- T __nv_cyl_bessel_i1
ATTR double __nv_cyl_bessel_i1(double x) { return __ocml_i1_f64(x); }

//-------- T __nv_cyl_bessel_i1f
ATTR float __nv_cyl_bessel_i1f(float x) { return __ocml_i1_f32(x); }

//-------- T __nv_frexp
ATTR double __nv_frexp(double x, __private int *ptr) { return __ocml_frexp_f64(x, ptr); }

//-------- T __nv_frexpf
ATTR float __nv_frexpf(float x, __private int *ptr) { return __ocml_frexp_f32(x, ptr); }

//-------- T __nv_ilogb
ATTR int __nv_ilogb(double x) { return __ocml_ilogb_f64(x); }

//-------- T __nv_ilogbf
ATTR int __nv_ilogbf(float x) { return __ocml_ilogb_f32(x); }

//-------- T __nv_ldexp
ATTR double __nv_ldexp(double x, int i) { return __ocml_ldexp_f64(x, i); }

//-------- T __nv_ldexpf
ATTR float __nv_ldexpf(float x, int i) { return __ocml_ldexp_f32(x, i); }

//-------- T __nv_modf
ATTR double __nv_modf(double x, __private double *ptr) { return __ocml_modf_f64(x, ptr); }

//-------- T __nv_modff
ATTR float __nv_modff(float x, __private float *ptr) { return __ocml_modf_f32(x, ptr); }

//-------- T __nv_norm3d
ATTR double __nv_norm3d(double x, double y, double z) { return __ocml_len3_f64(x,y,z); }

//-------- T __nv_norm3df
ATTR float __nv_norm3df(float x, float y, float z) { return __ocml_len3_f32(x,y,z); }

//-------- T __nv_norm4d
ATTR double __nv_norm4d(double a, double b, double c, double d) { return __ocml_len4_f64(a,b,c,d); }

//-------- T __nv_norm4df
ATTR float __nv_norm4df(float a, float b, float c, float d) { return __ocml_len4_f32(a,b,c,d); }

//-------- T __nv_normcdf
ATTR double __nv_normcdf(double x) { return __ocml_ncdf_f64(x); }

//-------- T __nv_normcdff
ATTR float __nv_normcdff(float x) { return __ocml_ncdf_f32(x); }

//-------- T __nv_normcdfinv
ATTR double __nv_normcdfinv(double x) { return __ocml_ncdfinv_f64(x); }

//-------- T __nv_normcdfinvf
ATTR float __nv_normcdfinvf(float x) { return __ocml_ncdfinv_f32(x); }

//-------- T __nv_powi
ATTR double __nv_powi(double x, int n) { return __ocml_pown_f64(x, n); }

//-------- T __nv_powi
ATTR float __nv_powif(float x, int n) { return __ocml_pown_f32(x, n); }

//-------- T __nv_remquo
ATTR double __nv_remquo(double x, double y, __private int *ptr) { return __ocml_remquo_f64(x, y, ptr); }

//-------- T __nv_remquof
ATTR float __nv_remquof(float x, float y, __private int *ptr) { return __ocml_remquo_f32(x, y, ptr); }

//-------- T __nv_saturatef
ATTR float __nv_saturatef(float x) { return __ocml_min_f32(__ocml_max_f32(x, 0.0f), 1.0f); }

//-------- T __nv_signbitd
ATTR int __nv_signbitd(double x) { return __ocml_signbit_f64(x); }

//-------- T __nv_signbitf
ATTR int __nv_signbitf(float x) { return __ocml_signbit_f32(x); }

//-------- T __nv_sincos
ATTR void __nv_sincos(double x, __private double * sptr, __private double *cptr) { (*sptr)=__ocml_sincos_f64(x, cptr); }

//-------- T __nv_sincosf
ATTR void __nv_sincosf(float x, __private float * sptr, __private float *cptr) { (*sptr)=__ocml_sincos_f32(x, cptr); }

//-------- T __nv_sincospi
ATTR void __nv_sincospi(double x, __private double * sptr, __private double *cptr) { (*sptr)=__ocml_sincospi_f64(x, cptr); }

//-------- T __nv_sincospif
ATTR void __nv_sincosfpif(float x, __private float * sptr, __private float *cptr) { (*sptr)=__ocml_sincospi_f32(x, cptr); }

