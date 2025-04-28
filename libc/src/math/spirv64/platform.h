//===-- SPIR-V specific definitions for math support ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_SPIRV64_SPIRV_H
#define LLVM_LIBC_SRC_MATH_SPIRV64_SPIRV_H

#include "declarations.h"

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {
LIBC_INLINE double acos(double x) { return __builtin_acos(x); }
LIBC_INLINE float acosf(float x) { return __builtin_acosf(x); }
LIBC_INLINE double acosh(double x) { return __builtin_acosh(x); }
LIBC_INLINE float acoshf(float x) { return __builtin_acoshf(x); }
LIBC_INLINE double asin(double x) { return __builtin_asin(x); }
LIBC_INLINE float asinf(float x) { return __builtin_asinf(x); }
LIBC_INLINE double asinh(double x) { return __builtin_asinh(x); }
LIBC_INLINE float asinhf(float x) { return __builtin_asinhf(x); }
LIBC_INLINE double atan2(double x, double y) { return __builtin_atan2(x, y); }
LIBC_INLINE float atan2f(float x, float y) { return __builtin_atan2f(x, y); }
LIBC_INLINE double atan(double x) { return __builtin_atan(x); }
LIBC_INLINE float atanf(float x) { return __builtin_atanf(x); }
LIBC_INLINE double atanh(double x) { return __builtin_atanh(x); }
LIBC_INLINE float atanhf(float x) { return __builtin_atanhf(x); }
LIBC_INLINE double cos(double x) { return __builtin_cos(x); }
LIBC_INLINE float cosf(float x) { return __builtin_cosf(x); }
LIBC_INLINE double cosh(double x) { return __builtin_cosh(x); }
LIBC_INLINE float coshf(float x) { return __builtin_coshf(x); }
LIBC_INLINE double erf(double x) { return __builtin_erf(x); }
LIBC_INLINE float erff(float x) { return __builtin_erff(x); }
LIBC_INLINE double exp(double x) { return __builtin_exp(x); }
LIBC_INLINE float expf(float x) { return __builtin_expf(x); }
LIBC_INLINE double exp2(double x) { return __builtin_exp2(x); }
LIBC_INLINE float exp2f(float x) { return __builtin_exp2f(x); }
LIBC_INLINE double exp10(double x) { return __builtin_exp10(x); }
LIBC_INLINE float exp10f(float x) { return __builtin_exp10f(x); }
LIBC_INLINE double expm1(double x) { return __builtin_expm1(x); }
LIBC_INLINE float expm1f(float x) { return __builtin_expm1f(x); }
LIBC_INLINE double fdim(double x, double y) { return __builtin_fdim(x, y); }
LIBC_INLINE float fdimf(float x, float y) { return __builtin_fdimf(x, y); }
LIBC_INLINE double hypot(double x, double y) { return __builtin_hypot(x, y); }
LIBC_INLINE float hypotf(float x, float y) { return __builtin_hypotf(x, y); }
LIBC_INLINE int ilogb(double x) { return __builtin_ilogb(x); }
LIBC_INLINE int ilogbf(float x) { return __builtin_ilogbf(x); }
LIBC_INLINE double ldexp(double x, int i) { return __builtin_ldexp(x, i); }
LIBC_INLINE float ldexpf(float x, int i) { return __builtin_ldexpf(x, i); }
LIBC_INLINE long long llrint(double x) { return __builtin_llrint(x); }
LIBC_INLINE long long llrintf(float x) { return __builtin_llrintf(x); }
LIBC_INLINE double log10(double x) { return __builtin_log10(x); }
LIBC_INLINE float log10f(float x) { return __builtin_log10f(x); }
LIBC_INLINE double log1p(double x) { return __builtin_log1p(x); }
LIBC_INLINE float log1pf(float x) { return __builtin_log1pf(x); }
LIBC_INLINE double log2(double x) { return __builtin_log2(x); }
LIBC_INLINE float log2f(float x) { return __builtin_log2f(x); }
LIBC_INLINE double log(double x) { return __builtin_log(x); }
LIBC_INLINE float logf(float x) { return __builtin_logf(x); }
LIBC_INLINE long lrint(double x) { return __builtin_lrint(x); }
LIBC_INLINE long lrintf(float x) { return __builtin_lrintf(x); }
LIBC_INLINE double nextafter(double x, double y) {
  return __builtin_nextafter(x, y);
}
LIBC_INLINE float nextafterf(float x, float y) { return __builtin_nextafterf(x, y); }
LIBC_INLINE double pow(double x, double y) { return __builtin_pow(x, y); }
LIBC_INLINE float powf(float x, float y) { return __builtin_powf(x, y); }
LIBC_INLINE double sin(double x) { return __builtin_sin(x); }
LIBC_INLINE float sinf(float x) { return __builtin_sinf(x); }
LIBC_INLINE void sincos(double x, double *sinptr, double *cosptr) {
  return __builtin_sincos(x, sinptr, cosptr);
}
LIBC_INLINE void sincosf(float x, float *sinptr, float *cosptr) {
  return __builtin_sincosf(x, sinptr, cosptr);
}
LIBC_INLINE double sinh(double x) { return __builtin_sinh(x); }
LIBC_INLINE float sinhf(float x) { return __builtin_sinhf(x); }
LIBC_INLINE double tan(double x) { return __builtin_tan(x); }
LIBC_INLINE float tanf(float x) { return __builtin_tanf(x); }
LIBC_INLINE double tanh(double x) { return __builtin_tanh(x); }
LIBC_INLINE float tanhf(float x) { return __builtin_tanhf(x); }
LIBC_INLINE double scalbn(double x, int i) { return __builtin_scalbn(x, i); }
LIBC_INLINE float scalbnf(float x, int i) { return __builtin_scalbnf(x, i); }
LIBC_INLINE double frexp(double x, int *i) { return __builtin_frexp(x, i); }
LIBC_INLINE float frexpf(float x, int *i) { return __builtin_frexpf(x, i); }
LIBC_INLINE double remquo(double x, double y, int *i) {
  return __builtin_remquo(x, y, i);
}
LIBC_INLINE float remquof(float x, float y, int *i) {
  return __builtin_remquof(x, y, i);
}
LIBC_INLINE double tgamma(double x) { return __builtin_tgamma(x); }
LIBC_INLINE float tgammaf(float x) { return __builtin_tgammaf(x); }

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_SPIRV64_SPIRV_H
