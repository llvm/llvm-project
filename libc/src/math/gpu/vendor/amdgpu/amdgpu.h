//===-- AMDGPU specific definitions for math support ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_AMDGPU_H
#define LLVM_LIBC_SRC_MATH_GPU_AMDGPU_H

#include "declarations.h"
#include "platform.h"

#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE {
namespace internal {
LIBC_INLINE double acos(double x) { return __ocml_acos_f64(x); }
LIBC_INLINE float acosf(float x) { return __ocml_acos_f32(x); }
LIBC_INLINE double acosh(double x) { return __ocml_acosh_f64(x); }
LIBC_INLINE float acoshf(float x) { return __ocml_acosh_f32(x); }
LIBC_INLINE double asin(double x) { return __ocml_asin_f64(x); }
LIBC_INLINE float asinf(float x) { return __ocml_asin_f32(x); }
LIBC_INLINE double asinh(double x) { return __ocml_asinh_f64(x); }
LIBC_INLINE float asinhf(float x) { return __ocml_asinh_f32(x); }
LIBC_INLINE double atan(double x) { return __ocml_atan_f64(x); }
LIBC_INLINE float atanf(float x) { return __ocml_atan_f32(x); }
LIBC_INLINE double atan2(double x, double y) { return __ocml_atan2_f64(x, y); }
LIBC_INLINE float atan2f(float x, float y) { return __ocml_atan2_f32(x, y); }
LIBC_INLINE double atanh(double x) { return __ocml_atanh_f64(x); }
LIBC_INLINE float atanhf(float x) { return __ocml_atanh_f32(x); }
LIBC_INLINE double cos(double x) { return __ocml_cos_f64(x); }
LIBC_INLINE float cosf(float x) { return __ocml_cos_f32(x); }
LIBC_INLINE double cosh(double x) { return __ocml_cosh_f64(x); }
LIBC_INLINE float coshf(float x) { return __ocml_cosh_f32(x); }
LIBC_INLINE double erf(double x) { return __ocml_erf_f64(x); }
LIBC_INLINE float erff(float x) { return __ocml_erf_f32(x); }
LIBC_INLINE double exp(double x) { return __builtin_exp(x); }
LIBC_INLINE float expf(float x) { return __builtin_expf(x); }
LIBC_INLINE double exp2(double x) { return __ocml_exp2_f64(x); }
LIBC_INLINE float exp2f(float x) { return __ocml_exp2_f32(x); }
LIBC_INLINE double exp10(double x) { return __ocml_exp10_f64(x); }
LIBC_INLINE float exp10f(float x) { return __ocml_exp10_f32(x); }
LIBC_INLINE double expm1(double x) { return __ocml_expm1_f64(x); }
LIBC_INLINE float expm1f(float x) { return __ocml_expm1_f32(x); }
LIBC_INLINE double fdim(double x, double y) { return __ocml_fdim_f64(x, y); }
LIBC_INLINE float fdimf(float x, float y) { return __ocml_fdim_f32(x, y); }
LIBC_INLINE double hypot(double x, double y) { return __ocml_hypot_f64(x, y); }
LIBC_INLINE float hypotf(float x, float y) { return __ocml_hypot_f32(x, y); }
LIBC_INLINE int ilogb(double x) { return __ocml_ilogb_f64(x); }
LIBC_INLINE int ilogbf(float x) { return __ocml_ilogb_f32(x); }
LIBC_INLINE double ldexp(double x, int i) { return __builtin_ldexp(x, i); }
LIBC_INLINE float ldexpf(float x, int i) { return __builtin_ldexpf(x, i); }
LIBC_INLINE long long llrint(double x) {
  return static_cast<long long>(__builtin_rint(x));
}
LIBC_INLINE long long llrintf(float x) {
  return static_cast<long long>(__builtin_rintf(x));
}
LIBC_INLINE double log10(double x) { return __ocml_log10_f64(x); }
LIBC_INLINE float log10f(float x) { return __ocml_log10_f32(x); }
LIBC_INLINE double log1p(double x) { return __ocml_log1p_f64(x); }
LIBC_INLINE float log1pf(float x) { return __ocml_log1p_f32(x); }
LIBC_INLINE double log2(double x) { return __ocml_log2_f64(x); }
LIBC_INLINE float log2f(float x) { return __ocml_log2_f32(x); }
LIBC_INLINE double log(double x) { return __ocml_log_f64(x); }
LIBC_INLINE float logf(float x) { return __ocml_log_f32(x); }
LIBC_INLINE long lrint(double x) {
  return static_cast<long>(__builtin_rint(x));
}
LIBC_INLINE long lrintf(float x) {
  return static_cast<long>(__builtin_rintf(x));
}
LIBC_INLINE double nextafter(double x, double y) {
  return __ocml_nextafter_f64(x, y);
}
LIBC_INLINE float nextafterf(float x, float y) {
  return __ocml_nextafter_f32(x, y);
}
LIBC_INLINE double pow(double x, double y) { return __ocml_pow_f64(x, y); }
LIBC_INLINE float powf(float x, float y) { return __ocml_pow_f32(x, y); }
LIBC_INLINE double sin(double x) { return __ocml_sin_f64(x); }
LIBC_INLINE float sinf(float x) { return __ocml_sin_f32(x); }
LIBC_INLINE void sincos(double x, double *sinptr, double *cosptr) {
  *sinptr = __ocml_sincos_f64(x, cosptr);
}
LIBC_INLINE void sincosf(float x, float *sinptr, float *cosptr) {
  *sinptr = __ocml_sincos_f32(x, cosptr);
}
LIBC_INLINE double sinh(double x) { return __ocml_sinh_f64(x); }
LIBC_INLINE float sinhf(float x) { return __ocml_sinh_f32(x); }
LIBC_INLINE double tan(double x) { return __ocml_tan_f64(x); }
LIBC_INLINE float tanf(float x) { return __ocml_tan_f32(x); }
LIBC_INLINE double tanh(double x) { return __ocml_tanh_f64(x); }
LIBC_INLINE float tanhf(float x) { return __ocml_tanh_f32(x); }
LIBC_INLINE double scalbn(double x, int i) {
  return __builtin_amdgcn_ldexp(x, i);
}
LIBC_INLINE float scalbnf(float x, int i) {
  return __builtin_amdgcn_ldexpf(x, i);
}
LIBC_INLINE double frexp(double x, int *nptr) {
  return __builtin_frexp(x, nptr);
}
LIBC_INLINE float frexpf(float x, int *nptr) {
  return __builtin_frexpf(x, nptr);
}
LIBC_INLINE double remquo(double x, double y, int *q) {
  int tmp;
  double r = __ocml_remquo_f64(x, y, (gpu::Private<int> *)&tmp);
  *q = tmp;
  return r;
}
LIBC_INLINE float remquof(float x, float y, int *q) {
  int tmp;
  float r = __ocml_remquo_f32(x, y, (gpu::Private<int> *)&tmp);
  *q = tmp;
  return r;
}
LIBC_INLINE double tgamma(double x) { return __ocml_tgamma_f64(x); }
LIBC_INLINE float tgammaf(float x) { return __ocml_tgamma_f32(x); }

} // namespace internal
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_GPU_AMDGPU_H
