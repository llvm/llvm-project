//===---- __clang_gpu_math.h - Generic GPU math functions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_GPU_MATH_H__
#define __CLANG_GPU_MATH_H__

#if defined(__HIP__) || defined(__CUDA__)

#include <__clang_gpu_libclc_declares.h>

#pragma push_macro("__GPU_MATH__")
#define __GPU_MATH__ static __inline__ __attribute__((device, always_inline))

#define __GPU_MATH_UNARY(__type, __name, __builtin)                            \
  __GPU_MATH__ __type __name(__type __x) { return __builtin(__x); }

#define __GPU_MATH_BINARY(__type, __name, __builtin)                           \
  __GPU_MATH__ __type __name(__type __x, __type __y) {                         \
    return __builtin(__x, __y);                                                \
  }

#define __GPU_MATH_TERNARY(__type, __name, __builtin)                          \
  __GPU_MATH__ __type __name(__type __x, __type __y, __type __z) {             \
    return __builtin(__x, __y, __z);                                           \
  }

#define __GPU_MATH_UNARY_RESULT(__result, __type, __name, __builtin)           \
  __GPU_MATH__ __result __name(__type __x) { return __builtin(__x); }

#define __GPU_MATH_BINARY_RESULT(__result, __type, __name, __builtin)          \
  __GPU_MATH__ __result __name(__type __x, __type __y) {                       \
    return __builtin(__x, __y);                                                \
  }

#define __GPU_MATH_UNARY_PTR(__type, __name, __builtin)                        \
  __GPU_MATH__ __type __name(__type __x, int *__exp) {                         \
    return __builtin(__x, __exp);                                              \
  }

#define __GPU_MATH_UNARY_OUT_PTR(__type, __name, __builtin)                    \
  __GPU_MATH__ __type __name(__type __x, __type *__out) {                      \
    return __builtin(__x, __out);                                              \
  }

#define __GPU_MATH_UNARY_INT_PTR(__type, __name, __builtin)                    \
  __GPU_MATH__ __type __name(__type __x, int *__out) {                         \
    return __builtin(__x, __out);                                              \
  }

#define __GPU_MATH_BINARY_OUT_PTR(__type, __name, __builtin)                   \
  __GPU_MATH__ __type __name(__type __x, __type __y, int *__out) {             \
    return __builtin(__x, __y, __out);                                         \
  }

#define __GPU_MATH_EXPONENT(__type, __exponent, __name, __builtin)             \
  __GPU_MATH__ __type __name(__type __x, __exponent __n) {                     \
    return __builtin(__x, __n);                                                \
  }

__GPU_MATH_UNARY(double, acos, __gpu_libclc_acos)
__GPU_MATH_UNARY(float, acosf, __gpu_libclc_acos)
__GPU_MATH_UNARY(double, acosh, __gpu_libclc_acosh)
__GPU_MATH_UNARY(float, acoshf, __gpu_libclc_acosh)
__GPU_MATH_UNARY(double, asin, __gpu_libclc_asin)
__GPU_MATH_UNARY(float, asinf, __gpu_libclc_asin)
__GPU_MATH_UNARY(double, asinh, __gpu_libclc_asinh)
__GPU_MATH_UNARY(float, asinhf, __gpu_libclc_asinh)
__GPU_MATH_UNARY(double, atan, __gpu_libclc_atan)
__GPU_MATH_UNARY(float, atanf, __gpu_libclc_atan)
__GPU_MATH_BINARY(double, atan2, __gpu_libclc_atan2)
__GPU_MATH_BINARY(float, atan2f, __gpu_libclc_atan2)
__GPU_MATH_UNARY(double, atanh, __gpu_libclc_atanh)
__GPU_MATH_UNARY(float, atanhf, __gpu_libclc_atanh)
__GPU_MATH_UNARY(double, cbrt, __gpu_libclc_cbrt)
__GPU_MATH_UNARY(float, cbrtf, __gpu_libclc_cbrt)
__GPU_MATH_UNARY(double, ceil, __gpu_libclc_ceil)
__GPU_MATH_UNARY(float, ceilf, __gpu_libclc_ceil)
__GPU_MATH_BINARY(double, copysign, __gpu_libclc_copysign)
__GPU_MATH_BINARY(float, copysignf, __gpu_libclc_copysign)
__GPU_MATH_UNARY(double, cos, __gpu_libclc_cos)
__GPU_MATH_UNARY(float, cosf, __gpu_libclc_cos)
__GPU_MATH_UNARY(double, cosh, __gpu_libclc_cosh)
__GPU_MATH_UNARY(float, coshf, __gpu_libclc_cosh)
__GPU_MATH_UNARY(double, erf, __gpu_libclc_erf)
__GPU_MATH_UNARY(float, erff, __gpu_libclc_erf)
__GPU_MATH_UNARY(double, erfc, __gpu_libclc_erfc)
__GPU_MATH_UNARY(float, erfcf, __gpu_libclc_erfc)
__GPU_MATH_UNARY(double, exp, __gpu_libclc_exp)
__GPU_MATH_UNARY(float, expf, __gpu_libclc_exp)
__GPU_MATH_UNARY(double, exp2, __gpu_libclc_exp2)
__GPU_MATH_UNARY(float, exp2f, __gpu_libclc_exp2)
__GPU_MATH_UNARY(double, exp10, __gpu_libclc_exp10)
__GPU_MATH_UNARY(float, exp10f, __gpu_libclc_exp10)
__GPU_MATH_UNARY(double, expm1, __gpu_libclc_expm1)
__GPU_MATH_UNARY(float, expm1f, __gpu_libclc_expm1)
__GPU_MATH_UNARY(double, fabs, __gpu_libclc_fabs)
__GPU_MATH_UNARY(float, fabsf, __gpu_libclc_fabs)
__GPU_MATH_BINARY(double, fdim, __gpu_libclc_fdim)
__GPU_MATH_BINARY(float, fdimf, __gpu_libclc_fdim)
__GPU_MATH_UNARY(double, floor, __gpu_libclc_floor)
__GPU_MATH_UNARY(float, floorf, __gpu_libclc_floor)
__GPU_MATH_TERNARY(double, fma, __gpu_libclc_fma)
__GPU_MATH_TERNARY(float, fmaf, __gpu_libclc_fma)
__GPU_MATH_BINARY(double, fmax, __gpu_libclc_fmax)
__GPU_MATH_BINARY(float, fmaxf, __gpu_libclc_fmax)
__GPU_MATH_BINARY(double, fmin, __gpu_libclc_fmin)
__GPU_MATH_BINARY(float, fminf, __gpu_libclc_fmin)
__GPU_MATH_BINARY(double, fmod, __gpu_libclc_fmod)
__GPU_MATH_BINARY(float, fmodf, __gpu_libclc_fmod)
__GPU_MATH_UNARY_INT_PTR(double, frexp, __gpu_libclc_frexp)
__GPU_MATH_UNARY_INT_PTR(float, frexpf, __gpu_libclc_frexp)
__GPU_MATH_BINARY(double, hypot, __gpu_libclc_hypot)
__GPU_MATH_BINARY(float, hypotf, __gpu_libclc_hypot)
__GPU_MATH_UNARY_RESULT(int, double, ilogb, __gpu_libclc_ilogb)
__GPU_MATH_UNARY_RESULT(int, float, ilogbf, __gpu_libclc_ilogb)
__GPU_MATH_EXPONENT(double, int, ldexp, __gpu_libclc_ldexp)
__GPU_MATH_EXPONENT(float, int, ldexpf, __gpu_libclc_ldexp)
__GPU_MATH_UNARY(double, lgamma, __gpu_libclc_lgamma)
__GPU_MATH_UNARY(float, lgammaf, __gpu_libclc_lgamma)
__GPU_MATH_UNARY_RESULT(long long, double, llrint, __gpu_libclc_rint)
__GPU_MATH_UNARY_RESULT(long long, float, llrintf, __gpu_libclc_rint)
__GPU_MATH_UNARY_RESULT(long long, double, llround, __gpu_libclc_round)
__GPU_MATH_UNARY_RESULT(long long, float, llroundf, __gpu_libclc_round)
__GPU_MATH_UNARY(double, log, __gpu_libclc_log)
__GPU_MATH_UNARY(float, logf, __gpu_libclc_log)
__GPU_MATH_UNARY(double, log10, __gpu_libclc_log10)
__GPU_MATH_UNARY(float, log10f, __gpu_libclc_log10)
__GPU_MATH_UNARY(double, log1p, __gpu_libclc_log1p)
__GPU_MATH_UNARY(float, log1pf, __gpu_libclc_log1p)
__GPU_MATH_UNARY(double, log2, __gpu_libclc_log2)
__GPU_MATH_UNARY(float, log2f, __gpu_libclc_log2)
__GPU_MATH_UNARY(double, logb, __gpu_libclc_logb)
__GPU_MATH_UNARY(float, logbf, __gpu_libclc_logb)
__GPU_MATH_UNARY_RESULT(long, double, lrint, __gpu_libclc_rint)
__GPU_MATH_UNARY_RESULT(long, float, lrintf, __gpu_libclc_rint)
__GPU_MATH_UNARY_RESULT(long, double, lround, __gpu_libclc_round)
__GPU_MATH_UNARY_RESULT(long, float, lroundf, __gpu_libclc_round)
__GPU_MATH_UNARY_OUT_PTR(double, modf, __gpu_libclc_modf)
__GPU_MATH_UNARY_OUT_PTR(float, modff, __gpu_libclc_modf)
__GPU_MATH_UNARY(double, nearbyint, __gpu_libclc_rint)
__GPU_MATH_UNARY(float, nearbyintf, __gpu_libclc_rint)
__GPU_MATH_BINARY(double, nextafter, __gpu_libclc_nextafter)
__GPU_MATH_BINARY(float, nextafterf, __gpu_libclc_nextafter)
__GPU_MATH_BINARY(double, pow, __gpu_libclc_pow)
__GPU_MATH_BINARY(float, powf, __gpu_libclc_pow)
__GPU_MATH_BINARY(double, remainder, __gpu_libclc_remainder)
__GPU_MATH_BINARY(float, remainderf, __gpu_libclc_remainder)
__GPU_MATH_BINARY_OUT_PTR(double, remquo, __gpu_libclc_remquo)
__GPU_MATH_BINARY_OUT_PTR(float, remquof, __gpu_libclc_remquo)
__GPU_MATH_UNARY(double, rint, __gpu_libclc_rint)
__GPU_MATH_UNARY(float, rintf, __gpu_libclc_rint)
__GPU_MATH_UNARY(double, round, __gpu_libclc_round)
__GPU_MATH_UNARY(float, roundf, __gpu_libclc_round)
__GPU_MATH_UNARY(double, roundeven, __gpu_libclc_rint)
__GPU_MATH_UNARY(float, roundevenf, __gpu_libclc_rint)
__GPU_MATH__ double scalbln(double __x, long __n) {
  return __gpu_libclc_ldexp(__x, (int)__n);
}
__GPU_MATH__ float scalblnf(float __x, long __n) {
  return __gpu_libclc_ldexp(__x, (int)__n);
}
__GPU_MATH_EXPONENT(double, int, scalbn, __gpu_libclc_ldexp)
__GPU_MATH_EXPONENT(float, int, scalbnf, __gpu_libclc_ldexp)
__GPU_MATH_UNARY(double, sin, __gpu_libclc_sin)
__GPU_MATH_UNARY(float, sinf, __gpu_libclc_sin)
__GPU_MATH_UNARY(double, sinh, __gpu_libclc_sinh)
__GPU_MATH_UNARY(float, sinhf, __gpu_libclc_sinh)
__GPU_MATH_UNARY(double, sqrt, __gpu_libclc_sqrt)
__GPU_MATH_UNARY(float, sqrtf, __gpu_libclc_sqrt)
__GPU_MATH_UNARY(double, tan, __gpu_libclc_tan)
__GPU_MATH_UNARY(float, tanf, __gpu_libclc_tan)
__GPU_MATH_UNARY(double, tanh, __gpu_libclc_tanh)
__GPU_MATH_UNARY(float, tanhf, __gpu_libclc_tanh)
__GPU_MATH_UNARY(double, tgamma, __gpu_libclc_tgamma)
__GPU_MATH_UNARY(float, tgammaf, __gpu_libclc_tgamma)
__GPU_MATH_UNARY(double, trunc, __gpu_libclc_trunc)
__GPU_MATH_UNARY(float, truncf, __gpu_libclc_trunc)

__GPU_MATH__ void sincos(double __x, double *__sin, double *__cos) {
  *__sin = __gpu_libclc_sincos(__x, __cos);
}
__GPU_MATH__ void sincosf(float __x, float *__sin, float *__cos) {
  *__sin = __gpu_libclc_sincos(__x, __cos);
}
__GPU_MATH__ void sincospi(double __x, double *__sin, double *__cos) {
  *__sin = __gpu_libclc_sinpi(__x);
  *__cos = __gpu_libclc_cospi(__x);
}
__GPU_MATH__ void sincospif(float __x, float *__sin, float *__cos) {
  *__sin = __gpu_libclc_sinpi(__x);
  *__cos = __gpu_libclc_cospi(__x);
}

#undef __GPU_MATH_EXPONENT
#undef __GPU_MATH_BINARY_OUT_PTR
#undef __GPU_MATH_UNARY_INT_PTR
#undef __GPU_MATH_UNARY_OUT_PTR
#undef __GPU_MATH_BINARY_RESULT
#undef __GPU_MATH_UNARY_RESULT
#undef __GPU_MATH_TERNARY
#undef __GPU_MATH_BINARY
#undef __GPU_MATH_UNARY

#pragma pop_macro("__GPU_MATH__")

#endif // defined(__HIP__) || defined(__CUDA__)
#endif // __CLANG_GPU_MATH_H__
