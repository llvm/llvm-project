/*===---- __clang_spirv_math.h - Device-side SPIRV math support ------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __CLANG_SPIRV_MATH_H__
#define __CLANG_SPIRV_MATH_H__

#if !defined(__SPIRV__) && !defined(__OPENMP_SPIRV__)
#error "This file is for SPIRV and OpenMP SPIRV device compilation only."
#endif

// The __CLANG_GPU_DISABLE_MATH_WRAPPERS macro provides a way to let standard
// libcalls reach the link step instead of being eagerly replaced.
#ifndef __CLANG_GPU_DISABLE_MATH_WRAPPERS

// __DEVICE__ is a helper macro with common set of attributes for the wrappers
// we implement in this file. We need static in order to avoid emitting unused
// functions and __forceinline__ helps inlining these wrappers at -O1.
#pragma push_macro("__DEVICE__")
#ifdef __OPENMP_SPIRV__
#if defined(__cplusplus)
#define __DEVICE__ static constexpr __attribute__((always_inline, nothrow))
#else
#define __DEVICE__ static __attribute__((always_inline, nothrow))
#endif
#else
#define __DEVICE__ static __device__ __forceinline__
#endif

__DEVICE__
float __cosf(float __x) { return __spirv_ocl_cos(__x); }
__DEVICE__
float __exp10f(float __x) { return __spirv_ocl_exp10(__x); }
__DEVICE__
float __expf(float __x) { return __spirv_ocl_exp(__x); }

__DEVICE__
float __fadd_rd(float __x, float __y) {
  float sum = __x + __y;
  float rounded = __spirv_ocl_floor(sum);
  if (rounded > sum)
    rounded -= 1.0f;
  return rounded;
}

__DEVICE__
float __fadd_rn(float __x, float __y) { return __spirv_ocl_rint(__x + __y); }
__DEVICE__
float __fadd_ru(float __x, float __y) { return __spirv_ocl_ceil(__x + __y); }
__DEVICE__
float __fadd_rz(float __x, float __y) { return __spirv_ocl_trunc(__x + __y); }

__DEVICE__
float __fdiv_rd(float __x, float __y) {
  float res = __x / __y;
  float rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0f;
  return rounded;
}
__DEVICE__
float __fdiv_rn(float __x, float __y) { return __spirv_ocl_rint(__x / __y); }
__DEVICE__
float __fdiv_ru(float __x, float __y) { return __spirv_ocl_ceil(__x / __y); }
__DEVICE__
float __fdiv_rz(float __x, float __y) { return __spirv_ocl_trunc(__x / __y); }
__DEVICE__
float __fdividef(float __x, float __y) { return __x / __y; }

__DEVICE__
float __fmaf_rd(float __x, float __y, float __z) {
  float res = __x * __y + __z;
  float rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0f;
  return rounded;
}
__DEVICE__
float __fmaf_rn(float __x, float __y, float __z) {
  return __spirv_ocl_rint(__x * __y + __z);
}
__DEVICE__
float __fmaf_ru(float __x, float __y, float __z) {
  return __spirv_ocl_ceil(__x * __y + __z);
}
__DEVICE__
float __fmaf_rz(float __x, float __y, float __z) {
  return __spirv_ocl_trunc(__x * __y + __z);
}

__DEVICE__
float __fmul_rd(float __x, float __y) {
  float res = __x * __y;
  float rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0f;
  return rounded;
}
__DEVICE__
float __fmul_rn(float __x, float __y) { return __spirv_ocl_rint(__x * __y); }
__DEVICE__
float __fmul_ru(float __x, float __y) { return __spirv_ocl_ceil(__x * __y); }
__DEVICE__
float __fmul_rz(float __x, float __y) { return __spirv_ocl_trunc(__x * __y); }

__DEVICE__
float __frcp_rd(float __x) { return __fdiv_rd(1.0f, __x); }
__DEVICE__
float __frcp_rn(float __x) { return __fdiv_rn(1.0f, __x); }
__DEVICE__
float __frcp_ru(float __x) { return __fdiv_ru(1.0f, __x); }
__DEVICE__
float __frcp_rz(float __x) { return __fdiv_rz(1.0f, __x); }
__DEVICE__

float __frsqrt_rn(float __x) {
  return __spirv_ocl_rint(__spirv_ocl_rsqrt(__x));
}

__DEVICE__
float __fsqrt_rd(float __x) {
  float res = __spirv_ocl_sqrt(__x);
  float rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0f;
  return rounded;
}
__DEVICE__
float __fsqrt_rn(float __x) { return __spirv_ocl_rint(__spirv_ocl_sqrt(__x)); }
__DEVICE__
float __fsqrt_ru(float __x) { return __spirv_ocl_ceil(__spirv_ocl_sqrt(__x)); }
__DEVICE__
float __fsqrt_rz(float __x) { return __spirv_ocl_trunc(__spirv_ocl_sqrt(__x)); }

__DEVICE__
float __fsub_rd(float __x, float __y) {
  float res = __x - __y;
  float rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0f;
  return rounded;
}
__DEVICE__
float __fsub_rn(float __x, float __y) { return __spirv_ocl_rint(__x - __y); }
__DEVICE__
float __fsub_ru(float __x, float __y) { return __spirv_ocl_ceil(__x - __y); }
__DEVICE__
float __fsub_rz(float __x, float __y) { return __spirv_ocl_trunc(__x - __y); }
__DEVICE__
float __log10f(float __x) { return __spirv_ocl_log10(__x); }
__DEVICE__
float __log2f(float __x) { return __spirv_ocl_log2(__x); }
__DEVICE__
float __logf(float __x) { return __spirv_ocl_log(__x); }
__DEVICE__
float __powf(float __x, float __y) { return __spirv_ocl_pow(__x, __y); }

__DEVICE__
float __saturatef(float __x) { return __spirv_ocl_fclamp(__x, 0.0f, 1.0f); }

__DEVICE__
void __sincosf(float __x, float *__sinptr, float *__cosptr) {
  *__sinptr = __spirv_ocl_sincos(__x, __cosptr);
}

__DEVICE__
float __sinf(float __x) { return __spirv_ocl_sin(__x); }

__DEVICE__
float __tanf(float __x) { return __spirv_ocl_tan(__x); }

__DEVICE__
int __finitef(float __x) { return !__spirv_IsInf(__x) && !__spirv_IsNan(__x); }
__DEVICE__
int __isinff(float __x) { return __spirv_IsInf(__x); }
__DEVICE__
int __isnanf(float __x) { return __spirv_IsNan(__x); }
__DEVICE__
int __signbitf(float __x) { return __builtin_signbitf(__x); }

__DEVICE__
int __finite(double __x) { return !__spirv_IsInf(__x) && !__spirv_IsNan(__x); }

__DEVICE__
int __isinf(double __x) { return __spirv_IsInf(__x); }

__DEVICE__
int __isnan(double __x) { return __spirv_IsNan(__x); }
__DEVICE__
int __signbit(double __x) { return __builtin_signbit(__x); }

__DEVICE__
double __dadd_rd(double __x, double __y) {
  double sum = __x + __y;
  double rounded = __spirv_ocl_floor(sum);
  if (rounded > sum)
    rounded -= 1.0;
  return rounded;
}
__DEVICE__
double __dadd_rn(double __x, double __y) { return __spirv_ocl_rint(__x + __y); }
__DEVICE__
double __dadd_ru(double __x, double __y) { return __spirv_ocl_ceil(__x + __y); }
__DEVICE__
double __dadd_rz(double __x, double __y) {
  return __spirv_ocl_trunc(__x + __y);
}
__DEVICE__
double __ddiv_rd(double __x, double __y) {
  double res = __x / __y;
  double rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0;
  return rounded;
}
__DEVICE__
double __ddiv_rn(double __x, double __y) { return __spirv_ocl_rint(__x / __y); }
__DEVICE__
double __ddiv_ru(double __x, double __y) { return __spirv_ocl_ceil(__x / __y); }
__DEVICE__
double __ddiv_rz(double __x, double __y) {
  return __spirv_ocl_trunc(__x / __y);
}

__DEVICE__
double __dmul_rd(double __x, double __y) {
  double res = __x * __y;
  double rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0;
  return rounded;
}
__DEVICE__
double __dmul_rn(double __x, double __y) { return __spirv_ocl_rint(__x * __y); }
__DEVICE__
double __dmul_ru(double __x, double __y) { return __spirv_ocl_ceil(__x * __y); }
__DEVICE__
double __dmul_rz(double __x, double __y) {
  return __spirv_ocl_trunc(__x * __y);
}

__DEVICE__
double __drcp_rd(double __x) { return __ddiv_rd(1.0, __x); }
__DEVICE__
double __drcp_rn(double __x) { return __ddiv_rn(1.0, __x); }
__DEVICE__
double __drcp_ru(double __x) { return __ddiv_ru(1.0, __x); }
__DEVICE__
double __drcp_rz(double __x) { return __ddiv_rz(1.0, __x); }

__DEVICE__
double __dsqrt_rd(double __x) {
  double res = __spirv_ocl_sqrt(__x);
  double rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0;
  return rounded;
}
__DEVICE__
double __dsqrt_rn(double __x) {
  return __spirv_ocl_rint(__spirv_ocl_sqrt(__x));
}
__DEVICE__
double __dsqrt_ru(double __x) {
  return __spirv_ocl_ceil(__spirv_ocl_sqrt(__x));
}
__DEVICE__
double __dsqrt_rz(double __x) {
  return __spirv_ocl_trunc(__spirv_ocl_sqrt(__x));
}

__DEVICE__
double __dsub_rd(double __x, double __y) {
  double res = __x - __y;
  double rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0;
  return rounded;
}
__DEVICE__
double __dsub_rn(double __x, double __y) { return __spirv_ocl_rint(__x - __y); }
__DEVICE__
double __dsub_ru(double __x, double __y) { return __spirv_ocl_ceil(__x - __y); }
__DEVICE__
double __dsub_rz(double __x, double __y) {
  return __spirv_ocl_trunc(__x - __y);
}

__DEVICE__
double __fma_rd(double __x, double __y, double __z) {
  double res = __x * __y + __z;
  double rounded = __spirv_ocl_floor(res);
  if (rounded > res)
    rounded -= 1.0;
  return rounded;
}
__DEVICE__
double __fma_rn(double __x, double __y, double __z) {
  return __spirv_ocl_rint(__x * __y + __z);
}
__DEVICE__
double __fma_ru(double __x, double __y, double __z) {
  return __spirv_ocl_ceil(__x * __y + __z);
}
__DEVICE__
double __fma_rz(double __x, double __y, double __z) {
  return __spirv_ocl_trunc(__x * __y + __z);
}

__DEVICE__ int abs(int __a) { return __spirv_ocl_s_abs(__a); }
__DEVICE__ double fabs(double __a) { return __spirv_ocl_fabs(__a); }
__DEVICE__ double acos(double __a) { return __spirv_ocl_acos(__a); }
__DEVICE__ float acosf(float __a) { return __spirv_ocl_acos(__a); }
__DEVICE__ double acosh(double __a) { return __spirv_ocl_acosh(__a); }
__DEVICE__ float acoshf(float __a) { return __spirv_ocl_acosh(__a); }
__DEVICE__ double asin(double __a) { return __spirv_ocl_asin(__a); }
__DEVICE__ float asinf(float __a) { return __spirv_ocl_asin(__a); }
__DEVICE__ double asinh(double __a) { return __spirv_ocl_asinh(__a); }
__DEVICE__ float asinhf(float __a) { return __spirv_ocl_asinh(__a); }
__DEVICE__ double atan(double __a) { return __spirv_ocl_atan(__a); }
__DEVICE__ double atan2(double __a, double __b) {
  return __spirv_ocl_atan2(__a, __b);
}
__DEVICE__ float atan2f(float __a, float __b) {
  return __spirv_ocl_atan2(__a, __b);
}
__DEVICE__ float atanf(float __a) { return __spirv_ocl_atan(__a); }
__DEVICE__ double atanh(double __a) { return __spirv_ocl_atanh(__a); }
__DEVICE__ float atanhf(float __a) { return __spirv_ocl_atanh(__a); }
__DEVICE__ double cbrt(double __a) { return __spirv_ocl_cbrt(__a); }
__DEVICE__ float cbrtf(float __a) { return __spirv_ocl_cbrt(__a); }
__DEVICE__ double ceil(double __a) { return __spirv_ocl_ceil(__a); }
__DEVICE__ float ceilf(float __a) { return __spirv_ocl_ceil(__a); }
__DEVICE__ double copysign(double __a, double __b) {
  return __spirv_ocl_copysign(__a, __b);
}
__DEVICE__ float copysignf(float __a, float __b) {
  return __spirv_ocl_copysign(__a, __b);
}
__DEVICE__ double cos(double __a) { return __spirv_ocl_cos(__a); }
__DEVICE__ float cosf(float __a) { return __spirv_ocl_cos(__a); }
__DEVICE__ double cosh(double __a) { return __spirv_ocl_cosh(__a); }
__DEVICE__ float coshf(float __a) { return __spirv_ocl_cosh(__a); }
__DEVICE__ double cospi(double __a) { return __spirv_ocl_cospi(__a); }
__DEVICE__ float cospif(float __a) { return __spirv_ocl_cospi(__a); }
__DEVICE__ double erf(double __a) { return __spirv_ocl_erf(__a); }
__DEVICE__ double erfc(double __a) { return __spirv_ocl_erfc(__a); }
__DEVICE__ float erfcf(float __a) { return __spirv_ocl_erfc(__a); }
__DEVICE__ double erfcx(double __a) {
  return __spirv_ocl_exp(__a * __a) * __spirv_ocl_erfc(__a);
}
__DEVICE__ float erfcxf(float __a) {
  return __spirv_ocl_exp(__a * __a) * __spirv_ocl_erfc(__a);
}
__DEVICE__ float erff(float __a) { return __spirv_ocl_erf(__a); }
__DEVICE__ double exp(double __a) { return __spirv_ocl_exp(__a); }
__DEVICE__ double exp10(double __a) { return __spirv_ocl_exp10(__a); }
__DEVICE__ float exp10f(float __a) { return __spirv_ocl_exp10(__a); }
__DEVICE__ double exp2(double __a) { return __spirv_ocl_exp2(__a); }
__DEVICE__ float exp2f(float __a) { return __spirv_ocl_exp2(__a); }
__DEVICE__ float expf(float __a) { return __spirv_ocl_exp(__a); }
__DEVICE__ double expm1(double __a) { return __spirv_ocl_expm1(__a); }
__DEVICE__ float expm1f(float __a) { return __spirv_ocl_expm1(__a); }
__DEVICE__ float fabsf(float __a) { return __spirv_ocl_fabs(__a); }
__DEVICE__ double fdim(double __a, double __b) {
  return __spirv_ocl_fdim(__a, __b);
}
__DEVICE__ float fdimf(float __a, float __b) {
  return __spirv_ocl_fdim(__a, __b);
}
__DEVICE__ double fdivide(double __a, double __b) { return __a / __b; }
__DEVICE__ float fdividef(float __a, float __b) { return __a / __b; }
__DEVICE__ double floor(double __f) { return __spirv_ocl_floor(__f); }
__DEVICE__ float floorf(float __f) { return __spirv_ocl_floor(__f); }
__DEVICE__ double fma(double __a, double __b, double __c) {
  return __spirv_ocl_fma(__a, __b, __c);
}
__DEVICE__ float fmaf(float __a, float __b, float __c) {
  return __spirv_ocl_fma(__a, __b, __c);
}
__DEVICE__ double fmax(double __a, double __b) {
  return __spirv_ocl_fmax(__a, __b);
}
__DEVICE__ float fmaxf(float __a, float __b) {
  return __spirv_ocl_fmax(__a, __b);
}
__DEVICE__ double fmin(double __a, double __b) {
  return __spirv_ocl_fmin(__a, __b);
}
__DEVICE__ float fminf(float __a, float __b) {
  return __spirv_ocl_fmin(__a, __b);
}
__DEVICE__ double fmod(double __a, double __b) {
  return __spirv_ocl_fmod(__a, __b);
}
__DEVICE__ float fmodf(float __a, float __b) {
  return __spirv_ocl_fmod(__a, __b);
}
__DEVICE__ double frexp(double __a, int *__b) {
  return __spirv_ocl_frexp(__a, __b);
}
__DEVICE__ float frexpf(float __a, int *__b) {
  return __spirv_ocl_frexp(__a, __b);
}
__DEVICE__ double hypot(double __a, double __b) {
  return __spirv_ocl_hypot(__a, __b);
}
__DEVICE__ float hypotf(float __a, float __b) {
  return __spirv_ocl_hypot(__a, __b);
}
__DEVICE__ int ilogb(double __a) { return __spirv_ocl_ilogb(__a); }
__DEVICE__ int ilogbf(float __a) { return __spirv_ocl_ilogb(__a); }
__DEVICE__ long labs(long __a) { return __spirv_ocl_s_abs(__a); };
__DEVICE__ double ldexp(double __a, int __b) {
  return __spirv_ocl_ldexp(__a, __b);
}
__DEVICE__ float ldexpf(float __a, int __b) {
  return __spirv_ocl_ldexp(__a, __b);
}
__DEVICE__ double lgamma(double __a) { return __spirv_ocl_lgamma(__a); }
__DEVICE__ float lgammaf(float __a) { return __spirv_ocl_lgamma(__a); }
__DEVICE__ long long llabs(long long __a) { return __spirv_ocl_s_abs(__a); }
__DEVICE__ long long llmax(long long __a, long long __b) {
  return __spirv_ocl_s_max(__a, __b);
}
__DEVICE__ long long llmin(long long __a, long long __b) {
  return __spirv_ocl_s_min(__a, __b);
}
__DEVICE__ long long llrint(double __a) { return __builtin_rint(__a); }
__DEVICE__ long long llrintf(float __a) { return __builtin_rintf(__a); }
__DEVICE__ long long llround(double __a) { return __builtin_round(__a); }
__DEVICE__ long long llroundf(float __a) { return __builtin_roundf(__a); }
__DEVICE__ double round(double __a) { return __spirv_ocl_round(__a); }
__DEVICE__ float roundf(float __a) { return __spirv_ocl_round(__a); }
__DEVICE__ double log(double __a) { return __spirv_ocl_log(__a); }
__DEVICE__ double log10(double __a) { return __spirv_ocl_log10(__a); }
__DEVICE__ float log10f(float __a) { return __spirv_ocl_log10(__a); }
__DEVICE__ double log1p(double __a) { return __spirv_ocl_log1p(__a); }
__DEVICE__ float log1pf(float __a) { return __spirv_ocl_log1p(__a); }
__DEVICE__ double log2(double __a) { return __spirv_ocl_log2(__a); }
__DEVICE__ float log2f(float __a) { return __spirv_ocl_log2(__a); }
__DEVICE__ double logb(double __a) { return __spirv_ocl_logb(__a); }
__DEVICE__ float logbf(float __a) { return __spirv_ocl_logb(__a); }
__DEVICE__ float logf(float __a) { return __spirv_ocl_log(__a); }
__DEVICE__ long lrint(double __a) { return __builtin_rint(__a); }
__DEVICE__ long lrintf(float __a) { return __builtin_rintf(__a); }
__DEVICE__ long lround(double __a) { return __builtin_round(__a); }
__DEVICE__ long lroundf(float __a) { return __builtin_roundf(__a); }
__DEVICE__ int max(int __a, int __b) { return __spirv_ocl_s_max(__a, __b); }
__DEVICE__ int min(int __a, int __b) { return __spirv_ocl_s_min(__a, __b); }
__DEVICE__ double modf(double __a, double *__b) {
  return __spirv_ocl_modf(__a, __b);
}
__DEVICE__ float modff(float __a, float *__b) {
  return __spirv_ocl_modf(__a, __b);
}
__DEVICE__ double nearbyint(double __a) { return __spirv_ocl_rint(__a); }
__DEVICE__ float nearbyintf(float __a) { return __spirv_ocl_rint(__a); }
__DEVICE__ double nextafter(double __a, double __b) {
  return __spirv_ocl_nextafter(__a, __b);
}
__DEVICE__ float nextafterf(float __a, float __b) {
  return __spirv_ocl_nextafter(__a, __b);
}

__DEVICE__ double norm(int __dim, const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __spirv_ocl_sqrt(__r);
}
__DEVICE__ double norm3d(double __a, double __b, double __c) {
  return __spirv_ocl_sqrt(__a * __a + __b * __b + __c * __c);
}
__DEVICE__ float norm3df(float __a, float __b, float __c) {
  return __spirv_ocl_sqrt(__a * __a + __b * __b + __c * __c);
}
__DEVICE__ double norm4d(double __a, double __b, double __c, double __d) {
  return __spirv_ocl_sqrt(__a * __a + __b * __b + __c * __c + __d * __d);
}
__DEVICE__ float norm4df(float __a, float __b, float __c, float __d) {
  return __spirv_ocl_sqrt(__a * __a + __b * __b + __c * __c + __d * __d);
}
__DEVICE__ double normcdf(double __a) {
  return 0.5 * (1.0 + __spirv_ocl_erf(__a * __spirv_ocl_rsqrt(2.0)));
}
__DEVICE__ float normcdff(float __a) {
  return 0.5f * (1.0f + __spirv_ocl_erf(__a * __spirv_ocl_rsqrt(2.0f)));
}
__DEVICE__ float normf(int __dim, const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __spirv_ocl_sqrt(__r);
}
__DEVICE__ double pow(double __a, double __b) {
  return __spirv_ocl_pow(__a, __b);
}
__DEVICE__ float powf(float __a, float __b) {
  return __spirv_ocl_pow(__a, __b);
}
__DEVICE__ double powi(double __a, int __b) { return pow(__a, (double)__b); }
__DEVICE__ float powif(float __a, int __b) { return pow(__a, (float)__b); }
__DEVICE__ double rcbrt(double __a) { return 1.0 / __spirv_ocl_cbrt(__a); }
__DEVICE__ float rcbrtf(float __a) { return 1.0f / __spirv_ocl_cbrt(__a); }
__DEVICE__ double remainder(double __a, double __b) {
  return __spirv_ocl_remainder(__a, __b);
}
__DEVICE__ float remainderf(float __a, float __b) {
  return __spirv_ocl_remainder(__a, __b);
}
__DEVICE__ double remquo(double __a, double __b, int *__c) {
  return __spirv_ocl_remquo(__a, __b, __c);
}
__DEVICE__ float remquof(float __a, float __b, int *__c) {
  return __spirv_ocl_remquo(__a, __b, __c);
}
__DEVICE__ double rhypot(double __a, double __b) {
  return __spirv_ocl_hypot(__a, __b);
}
__DEVICE__ float rhypotf(float __a, float __b) {
  return __spirv_ocl_hypot(__a, __b);
}
__DEVICE__ double rint(double __a) { return __spirv_ocl_rint(__a); }
__DEVICE__ float rintf(float __a) { return __spirv_ocl_rint(__a); }
__DEVICE__ double rnorm(int __dim, const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __spirv_ocl_rsqrt(__r);
}
__DEVICE__ double rnorm3d(double __a, double __b, double __c) {
  return __spirv_ocl_rsqrt(__a * __a + __b * __b + __c * __c);
}
__DEVICE__ float rnorm3df(float __a, float __b, float __c) {
  return __spirv_ocl_rsqrt(__a * __a + __b * __b + __c * __c);
}
__DEVICE__ double rnorm4d(double __a, double __b, double __c, double __d) {
  return __spirv_ocl_rsqrt(__a * __a + __b * __b + __c * __c + __d * __d);
}
__DEVICE__ float rnorm4df(float __a, float __b, float __c, float __d) {
  return __spirv_ocl_rsqrt(__a * __a + __b * __b + __c * __c + __d * __d);
}
__DEVICE__ float rnormf(int __dim, const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __spirv_ocl_rsqrt(__r);
}
__DEVICE__ double rsqrt(double __a) { return __spirv_ocl_rsqrt(__a); }
__DEVICE__ float rsqrtf(float __a) { return __spirv_ocl_rsqrt(__a); }
__DEVICE__ double scalbn(double __a, int __b) {
  return __spirv_ocl_ldexp(__a, __b);
}
__DEVICE__ float scalbnf(float __a, int __b) {
  return __spirv_ocl_ldexp(__a, __b);
}
__DEVICE__ double scalbln(double __a, long __b) {
  if (__b > INT_MAX)
    return __a > 0 ? HUGE_VAL : -HUGE_VAL;
  if (__b < INT_MIN)
    return __a > 0 ? 0.0 : -0.0;
  return scalbn(__a, (int)__b);
}
__DEVICE__ float scalblnf(float __a, long __b) {
  if (__b > INT_MAX)
    return __a > 0 ? HUGE_VALF : -HUGE_VALF;
  if (__b < INT_MIN)
    return __a > 0 ? 0.f : -0.f;
  return scalbnf(__a, (int)__b);
}
__DEVICE__ double sin(double __a) { return __spirv_ocl_sin(__a); }
__DEVICE__ void sincos(double __a, double *__s, double *__c) {
  *__s = __spirv_ocl_sincos(__a, __c);
}
__DEVICE__ void sincosf(float __a, float *__s, float *__c) {
  *__s = __spirv_ocl_sincos(__a, __c);
}
__DEVICE__ void sincospi(double __a, double *__s, double *__c) {
  *__s = __spirv_ocl_sinpi(__a);
  *__c = __spirv_ocl_cospi(__a);
}
__DEVICE__ void sincospif(float __a, float *__s, float *__c) {
  *__s = __spirv_ocl_sinpi(__a);
  *__c = __spirv_ocl_cospi(__a);
}
__DEVICE__ float sinf(float __a) { return __spirv_ocl_sin(__a); }
__DEVICE__ double sinh(double __a) { return __spirv_ocl_sinh(__a); }
__DEVICE__ float sinhf(float __a) { return __spirv_ocl_sinh(__a); }
__DEVICE__ double sinpi(double __a) { return __spirv_ocl_sinpi(__a); }
__DEVICE__ float sinpif(float __a) { return __spirv_ocl_sinpi(__a); }
__DEVICE__ double sqrt(double __a) { return __spirv_ocl_sqrt(__a); }
__DEVICE__ float sqrtf(float __a) { return __spirv_ocl_sqrt(__a); }
__DEVICE__ double tan(double __a) { return __spirv_ocl_tan(__a); }
__DEVICE__ float tanf(float __a) { return __spirv_ocl_tan(__a); }
__DEVICE__ double tanh(double __a) { return __spirv_ocl_tanh(__a); }
__DEVICE__ float tanhf(float __a) { return __spirv_ocl_tanh(__a); }
__DEVICE__ double tgamma(double __a) { return __spirv_ocl_tgamma(__a); }
__DEVICE__ float tgammaf(float __a) { return __spirv_ocl_tgamma(__a); }
__DEVICE__ double trunc(double __a) { return __spirv_ocl_trunc(__a); }
__DEVICE__ float truncf(float __a) { return __spirv_ocl_trunc(__a); }
__DEVICE__ unsigned long long ullmax(unsigned long long __a,
                                     unsigned long long __b) {
  return __spirv_ocl_u_max(__a, __b);
}
__DEVICE__ unsigned long long ullmin(unsigned long long __a,
                                     unsigned long long __b) {
  return __spirv_ocl_u_min(__a, __b);
}
__DEVICE__ unsigned int umax(unsigned int __a, unsigned int __b) {
  return __spirv_ocl_u_max(__a, __b);
}
__DEVICE__ unsigned int umin(unsigned int __a, unsigned int __b) {
  return __spirv_ocl_u_min(__a, __b);
}

#if !defined(__cplusplus) && __STDC_VERSION__ >= 201112L
#define isfinite(__x) _Generic((__x), float: __finitef, double: __finite)(__x)
#define isinf(__x) _Generic((__x), float: __isinff, double: __isinf)(__x)
#define isnan(__x) _Generic((__x), float: __isnanf, double: __isnan)(__x)
#define signbit(__x) _Generic((__x), float: __signbitf, double: __signbit)(__x)
#endif // !defined(__cplusplus) && __STDC_VERSION__ >= 201112L

__DEVICE__
unsigned long __make_mantissa_base8(const char *__tagp) {
  unsigned long __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '7')
      __r = (__r * 8u) + __tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

__DEVICE__
unsigned long __make_mantissa_base10(const char *__tagp) {
  unsigned long __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 10u) + __tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

__DEVICE__
unsigned long __make_mantissa_base16(const char *__tagp) {
  unsigned long __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 16u) + __tmp - '0';
    else if (__tmp >= 'a' && __tmp <= 'f')
      __r = (__r * 16u) + __tmp - 'a' + 10;
    else if (__tmp >= 'A' && __tmp <= 'F')
      __r = (__r * 16u) + __tmp - 'A' + 10;
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

__DEVICE__
unsigned long __make_mantissa(const char *__tagp) {
  if (!__tagp)
    return 0;
  if (*__tagp == '0') {
    ++__tagp;

    if (*__tagp == 'x' || *__tagp == 'X')
      return __make_mantissa_base16(++__tagp);
    else
      return __make_mantissa_base8(__tagp);
  }

  return __make_mantissa_base10(__tagp);
}

float nanf(const char *__tagp) {
  return __spirv_ocl_nan((unsigned int)__make_mantissa(__tagp));
}
double nan(const char *__tagp) {
  return __spirv_ocl_nan(__make_mantissa(__tagp));
}

#pragma pop_macro("__DEVICE__")
#endif // __CLANG_GPU_DISABLE_MATH_WRAPPERS
#endif // __CLANG_SPIRV_MATH_H__
