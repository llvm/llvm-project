//===-- __clang_gpu_libclc_declares.h - libclc device declarations --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_GPU_LIBCLC_DECLARES_H__
#define __CLANG_GPU_LIBCLC_DECLARES_H__

#if defined(__HIP__) || defined(__CUDA__)

#pragma push_macro("__GPU_LIBCLC_DECL__")
#define __GPU_LIBCLC_DECL__ __attribute__((device, overloadable))

#define __GPU_LIBCLC_UNARY(__type, __name, __symbol)                          \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type) __asm(__symbol);

#define __GPU_LIBCLC_BINARY(__type, __name, __symbol)                         \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type, __type)            \
      __asm(__symbol);

#define __GPU_LIBCLC_TERNARY(__type, __name, __symbol)                        \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type, __type, __type)    \
      __asm(__symbol);

#define __GPU_LIBCLC_UNARY_RESULT(__result, __type, __name, __symbol)         \
  __GPU_LIBCLC_DECL__ __result __gpu_libclc_##__name(__type) __asm(__symbol);

#define __GPU_LIBCLC_UNARY_PTR(__type, __name, __symbol)                      \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type, __type *)          \
      __asm(__symbol);

#define __GPU_LIBCLC_UNARY_INT_PTR(__type, __name, __symbol)                  \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type, int *)             \
      __asm(__symbol);

#define __GPU_LIBCLC_BINARY_OUT_PTR(__type, __name, __symbol)                 \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type, __type, int *)     \
      __asm(__symbol);

#define __GPU_LIBCLC_EXPONENT(__type, __exponent, __name, __symbol)           \
  __GPU_LIBCLC_DECL__ __type __gpu_libclc_##__name(__type, __exponent)        \
      __asm(__symbol);

__GPU_LIBCLC_UNARY(double, acos, "_Z4acosd")
__GPU_LIBCLC_UNARY(float, acos, "_Z4acosf")
__GPU_LIBCLC_UNARY(double, acosh, "_Z5acoshd")
__GPU_LIBCLC_UNARY(float, acosh, "_Z5acoshf")
__GPU_LIBCLC_UNARY(double, asin, "_Z4asind")
__GPU_LIBCLC_UNARY(float, asin, "_Z4asinf")
__GPU_LIBCLC_UNARY(double, asinh, "_Z5asinhd")
__GPU_LIBCLC_UNARY(float, asinh, "_Z5asinhf")
__GPU_LIBCLC_UNARY(double, atan, "_Z4atand")
__GPU_LIBCLC_UNARY(float, atan, "_Z4atanf")
__GPU_LIBCLC_BINARY(double, atan2, "_Z5atan2dd")
__GPU_LIBCLC_BINARY(float, atan2, "_Z5atan2ff")
__GPU_LIBCLC_UNARY(double, atanh, "_Z5atanhd")
__GPU_LIBCLC_UNARY(float, atanh, "_Z5atanhf")
__GPU_LIBCLC_UNARY(double, cbrt, "_Z4cbrtd")
__GPU_LIBCLC_UNARY(float, cbrt, "_Z4cbrtf")
__GPU_LIBCLC_UNARY(double, ceil, "_Z4ceild")
__GPU_LIBCLC_UNARY(float, ceil, "_Z4ceilf")
__GPU_LIBCLC_BINARY(double, copysign, "_Z8copysigndd")
__GPU_LIBCLC_BINARY(float, copysign, "_Z8copysignff")
__GPU_LIBCLC_UNARY(double, cos, "_Z3cosd")
__GPU_LIBCLC_UNARY(float, cos, "_Z3cosf")
__GPU_LIBCLC_UNARY(double, cosh, "_Z4coshd")
__GPU_LIBCLC_UNARY(float, cosh, "_Z4coshf")
__GPU_LIBCLC_UNARY(double, cospi, "_Z5cospid")
__GPU_LIBCLC_UNARY(float, cospi, "_Z5cospif")
__GPU_LIBCLC_UNARY(double, erf, "_Z3erfd")
__GPU_LIBCLC_UNARY(float, erf, "_Z3erff")
__GPU_LIBCLC_UNARY(double, erfc, "_Z4erfcd")
__GPU_LIBCLC_UNARY(float, erfc, "_Z4erfcf")
__GPU_LIBCLC_UNARY(double, exp, "_Z3expd")
__GPU_LIBCLC_UNARY(float, exp, "_Z3expf")
__GPU_LIBCLC_UNARY(double, exp2, "_Z4exp2d")
__GPU_LIBCLC_UNARY(float, exp2, "_Z4exp2f")
__GPU_LIBCLC_UNARY(double, exp10, "_Z5exp10d")
__GPU_LIBCLC_UNARY(float, exp10, "_Z5exp10f")
__GPU_LIBCLC_UNARY(double, expm1, "_Z5expm1d")
__GPU_LIBCLC_UNARY(float, expm1, "_Z5expm1f")
__GPU_LIBCLC_UNARY(double, fabs, "_Z4fabsd")
__GPU_LIBCLC_UNARY(float, fabs, "_Z4fabsf")
__GPU_LIBCLC_BINARY(double, fdim, "_Z4fdimdd")
__GPU_LIBCLC_BINARY(float, fdim, "_Z4fdimff")
__GPU_LIBCLC_UNARY(double, floor, "_Z5floord")
__GPU_LIBCLC_UNARY(float, floor, "_Z5floorf")
__GPU_LIBCLC_TERNARY(double, fma, "_Z3fmaddd")
__GPU_LIBCLC_TERNARY(float, fma, "_Z3fmafff")
__GPU_LIBCLC_BINARY(double, fmax, "_Z4fmaxdd")
__GPU_LIBCLC_BINARY(float, fmax, "_Z4fmaxff")
__GPU_LIBCLC_BINARY(double, fmin, "_Z4fmindd")
__GPU_LIBCLC_BINARY(float, fmin, "_Z4fminff")
__GPU_LIBCLC_BINARY(double, fmod, "_Z4fmoddd")
__GPU_LIBCLC_BINARY(float, fmod, "_Z4fmodff")
__GPU_LIBCLC_UNARY_INT_PTR(double, frexp, "_Z5frexpdPi")
__GPU_LIBCLC_UNARY_INT_PTR(float, frexp, "_Z5frexpfPi")
__GPU_LIBCLC_BINARY(double, hypot, "_Z5hypotdd")
__GPU_LIBCLC_BINARY(float, hypot, "_Z5hypotff")
__GPU_LIBCLC_UNARY_RESULT(int, double, ilogb, "_Z5ilogbd")
__GPU_LIBCLC_UNARY_RESULT(int, float, ilogb, "_Z5ilogbf")
__GPU_LIBCLC_EXPONENT(double, int, ldexp, "_Z5ldexpdi")
__GPU_LIBCLC_EXPONENT(float, int, ldexp, "_Z5ldexpfi")
__GPU_LIBCLC_UNARY(double, lgamma, "_Z6lgammad")
__GPU_LIBCLC_UNARY(float, lgamma, "_Z6lgammaf")
__GPU_LIBCLC_UNARY(double, log, "_Z3logd")
__GPU_LIBCLC_UNARY(float, log, "_Z3logf")
__GPU_LIBCLC_UNARY(double, log10, "_Z5log10d")
__GPU_LIBCLC_UNARY(float, log10, "_Z5log10f")
__GPU_LIBCLC_UNARY(double, log1p, "_Z5log1pd")
__GPU_LIBCLC_UNARY(float, log1p, "_Z5log1pf")
__GPU_LIBCLC_UNARY(double, log2, "_Z4log2d")
__GPU_LIBCLC_UNARY(float, log2, "_Z4log2f")
__GPU_LIBCLC_UNARY(double, logb, "_Z4logbd")
__GPU_LIBCLC_UNARY(float, logb, "_Z4logbf")
__GPU_LIBCLC_UNARY_PTR(double, modf, "_Z4modfdPd")
__GPU_LIBCLC_UNARY_PTR(float, modf, "_Z4modffPf")
__GPU_LIBCLC_BINARY(double, nextafter, "_Z9nextafterdd")
__GPU_LIBCLC_BINARY(float, nextafter, "_Z9nextafterff")
__GPU_LIBCLC_BINARY(double, pow, "_Z3powdd")
__GPU_LIBCLC_BINARY(float, pow, "_Z3powff")
__GPU_LIBCLC_BINARY(double, remainder, "_Z9remainderdd")
__GPU_LIBCLC_BINARY(float, remainder, "_Z9remainderff")
__GPU_LIBCLC_BINARY_OUT_PTR(double, remquo, "_Z6remquoddPi")
__GPU_LIBCLC_BINARY_OUT_PTR(float, remquo, "_Z6remquoffPi")
__GPU_LIBCLC_UNARY(double, rint, "_Z4rintd")
__GPU_LIBCLC_UNARY(float, rint, "_Z4rintf")
__GPU_LIBCLC_UNARY(double, round, "_Z5roundd")
__GPU_LIBCLC_UNARY(float, round, "_Z5roundf")
__GPU_LIBCLC_UNARY(double, sin, "_Z3sind")
__GPU_LIBCLC_UNARY(float, sin, "_Z3sinf")
__GPU_LIBCLC_UNARY_PTR(double, sincos, "_Z6sincosdPd")
__GPU_LIBCLC_UNARY_PTR(float, sincos, "_Z6sincosfPf")
__GPU_LIBCLC_UNARY(double, sinpi, "_Z5sinpid")
__GPU_LIBCLC_UNARY(float, sinpi, "_Z5sinpif")
__GPU_LIBCLC_UNARY(double, sinh, "_Z4sinhd")
__GPU_LIBCLC_UNARY(float, sinh, "_Z4sinhf")
__GPU_LIBCLC_UNARY(double, sqrt, "_Z4sqrtd")
__GPU_LIBCLC_UNARY(float, sqrt, "_Z4sqrtf")
__GPU_LIBCLC_UNARY(double, tan, "_Z3tand")
__GPU_LIBCLC_UNARY(float, tan, "_Z3tanf")
__GPU_LIBCLC_UNARY(double, tanh, "_Z4tanhd")
__GPU_LIBCLC_UNARY(float, tanh, "_Z4tanhf")
__GPU_LIBCLC_UNARY(double, tgamma, "_Z6tgammad")
__GPU_LIBCLC_UNARY(float, tgamma, "_Z6tgammaf")
__GPU_LIBCLC_UNARY(double, trunc, "_Z5truncd")
__GPU_LIBCLC_UNARY(float, trunc, "_Z5truncf")

#undef __GPU_LIBCLC_EXPONENT
#undef __GPU_LIBCLC_BINARY_OUT_PTR
#undef __GPU_LIBCLC_UNARY_INT_PTR
#undef __GPU_LIBCLC_UNARY_PTR
#undef __GPU_LIBCLC_UNARY_RESULT
#undef __GPU_LIBCLC_TERNARY
#undef __GPU_LIBCLC_BINARY
#undef __GPU_LIBCLC_UNARY

#pragma pop_macro("__GPU_LIBCLC_DECL__")

#endif // defined(__HIP__) || defined(__CUDA__)
#endif // __CLANG_GPU_LIBCLC_DECLARES_H__
