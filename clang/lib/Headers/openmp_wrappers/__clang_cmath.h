/*===---- __clang_cmath.h - cmath decls ------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_CMATH_H__
#define __CLANG_CMATH_H__

#if !defined(__OPENMP_SPIRV__) && !defined(__OPENMP_AMDGCN__) &&               \
    !defined(__OPENMP_NVPTX__)
#error "This file is for SPIRV/HIP/CUDA OpenMP device compilation only."
#endif
#define __DEVICE__ static constexpr __attribute__((always_inline, nothrow))

__DEVICE__ float sin(float __x) { return ::sinf(__x); }
__DEVICE__ float sinh(float __x) { return ::sinhf(__x); }
__DEVICE__ float cos(float __x) { return ::cosf(__x); }
__DEVICE__ float cosh(float __x) { return ::coshf(__x); }
__DEVICE__ double abs(double __x) { return ::fabs(__x); }
__DEVICE__ float abs(float __x) { return ::fabsf(__x); }
__DEVICE__ long long abs(long long __n) { return ::llabs(__n); }
__DEVICE__ long abs(long __n) { return ::labs(__n); }
__DEVICE__ float fma(float __x, float __y, float __z) {
  return ::fmaf(__x, __y, __z);
}
__DEVICE__ int fpclassify(float __x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, __x);
}
__DEVICE__ int fpclassify(double __x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, __x);
}
__DEVICE__ float frexp(float __arg, int *__exp) {
  return ::frexpf(__arg, __exp);
}
__DEVICE__ float acos(float __x) { return ::acosf(__x); }
__DEVICE__ float acosh(float __x) { return ::acoshf(__x); }
__DEVICE__ float asin(float __x) { return ::asinf(__x); }
__DEVICE__ float asinh(float __x) { return ::asinhf(__x); }
__DEVICE__ float atan(float __x) { return ::atanf(__x); }
__DEVICE__ float atanh(float __x) { return ::atanhf(__x); }
__DEVICE__ float atan2(float __x, float __y) { return ::atan2f(__x, __y); }
__DEVICE__ float ceil(float __x) { return ::ceilf(__x); }
__DEVICE__ float exp(float __x) { return ::expf(__x); }
__DEVICE__ float exp2(float __x) { return ::exp2f(__x); }
__DEVICE__ float expm1(float __x) { return ::expm1f(__x); }
__DEVICE__ float fabs(float __x) { return ::fabsf(__x); }
__DEVICE__ float floor(float __x) { return ::floorf(__x); }
__DEVICE__ float fmod(float __x, float __y) { return ::fmodf(__x, __y); }
__DEVICE__ float hypot(float __x, float __y) { return ::hypotf(__x, __y); }

// For OpenMP we work around some old system headers that have non-conforming
// `isinf(float)` and `isnan(float)` implementations that return an `int`. We do
// this by providing two versions of these functions, differing only in the
// return type. To avoid conflicting definitions we disable implicit base
// function generation. That means we will end up with two specializations, one
// per type, but only one has a base function defined by the system header.
#pragma omp begin declare variant match(                                       \
        implementation = {extension(disable_implicit_base)})

// FIXME: We lack an extension to customize the mangling of the variants, e.g.,
//        add a suffix. This means we would clash with the names of the variants
//        (note that we do not create implicit base functions here). To avoid
//        this clash we add a new trait to some of them that is always true
//        (this is LLVM after all ;)). It will only influence the mangled name
//        of the variants inside the inner region and avoid the clash.
#pragma omp begin declare variant match(implementation = {vendor(llvm)})

__DEVICE__ int isinf(float __x) { return ::__isinff(__x); }
__DEVICE__ int isinf(double __x) { return ::__isinf(__x); }
__DEVICE__ int isfinite(float __x) { return ::__finitef(__x); }
__DEVICE__ int isfinite(double __x) { return ::__finite(__x); }
__DEVICE__ int isnan(float __x) { return ::__isnanf(__x); }
__DEVICE__ int isnan(double __x) { return ::__isnan(__x); }

#pragma omp end declare variant

__DEVICE__ bool isinf(float __x) { return ::__isinff(__x); }
__DEVICE__ bool isinf(double __x) { return ::__isinf(__x); }
__DEVICE__ bool isfinite(float __x) { return ::__finitef(__x); }
__DEVICE__ bool isfinite(double __x) { return ::__finite(__x); }
__DEVICE__ bool isnan(float __x) { return ::__isnanf(__x); }
__DEVICE__ bool isnan(double __x) { return ::__isnan(__x); }

#pragma omp end declare variant

__DEVICE__ bool isgreater(float __x, float __y) {
  return __builtin_isgreater(__x, __y);
}
__DEVICE__ bool isgreater(double __x, double __y) {
  return __builtin_isgreater(__x, __y);
}
__DEVICE__ bool isgreaterequal(float __x, float __y) {
  return __builtin_isgreaterequal(__x, __y);
}
__DEVICE__ bool isgreaterequal(double __x, double __y) {
  return __builtin_isgreaterequal(__x, __y);
}
__DEVICE__ bool isless(float __x, float __y) {
  return __builtin_isless(__x, __y);
}
__DEVICE__ bool isless(double __x, double __y) {
  return __builtin_isless(__x, __y);
}
__DEVICE__ bool islessequal(float __x, float __y) {
  return __builtin_islessequal(__x, __y);
}
__DEVICE__ bool islessequal(double __x, double __y) {
  return __builtin_islessequal(__x, __y);
}
__DEVICE__ bool islessgreater(float __x, float __y) {
  return __builtin_islessgreater(__x, __y);
}
__DEVICE__ bool islessgreater(double __x, double __y) {
  return __builtin_islessgreater(__x, __y);
}
__DEVICE__ bool isnormal(float __x) { return __builtin_isnormal(__x); }
__DEVICE__ bool isnormal(double __x) { return __builtin_isnormal(__x); }
__DEVICE__ bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ float modf(float __x, float *__iptr) { return ::modff(__x, __iptr); }
__DEVICE__ float pow(float __base, int __iexp) {
  return ::powif(__base, __iexp);
}
__DEVICE__ double pow(double __base, int __iexp) {
  return ::powi(__base, __iexp);
}
__DEVICE__ float remquo(float __x, float __y, int *__quo) {
  return ::remquof(__x, __y, __quo);
}
__DEVICE__ float scalbln(float __x, long int __n) {
  return ::scalblnf(__x, __n);
}
__DEVICE__ bool signbit(float __x) { return ::__signbitf(__x); }
__DEVICE__ bool signbit(double __x) {
#if defined(__OPENMP_NVPTX__)
  return ::__signbitd(__x);
#else
  return ::__signbit(__x);
#endif
}
__DEVICE__ float ldexp(float __arg, int __exp) {
  return ::ldexpf(__arg, __exp);
}
__DEVICE__ float log(float __x) { return ::logf(__x); }
__DEVICE__ float log10(float __x) { return ::log10f(__x); }
__DEVICE__ float log1p(float __x) { return ::log1pf(__x); }
__DEVICE__ float log2(float __x) { return ::log2f(__x); }
__DEVICE__ float logb(float __x) { return ::logbf(__x); }

__DEVICE__ float pow(float __base, float __exp) {
  return ::powf(__base, __exp);
}
__DEVICE__ float sqrt(float __x) { return ::sqrtf(__x); }
__DEVICE__ float tan(float __x) { return ::tanf(__x); }
__DEVICE__ float tanh(float __x) { return ::tanhf(__x); }
__DEVICE__ float cbrt(float __x) { return ::cbrtf(__x); }
__DEVICE__ float copysign(float __a, float __b) {
  return ::copysignf(__a, __b);
}
__DEVICE__ float erf(float __x) { return ::erff(__x); }
__DEVICE__ float erfc(float __x) { return ::erfcf(__x); }
__DEVICE__ float fdim(float __a, float __b) { return ::fdimf(__a, __b); }
__DEVICE__ int ilogb(float __x) { return ::ilogbf(__x); }
__DEVICE__ float lgamma(float __x) { return ::lgammaf(__x); }
__DEVICE__ float tgamma(float __x) { return ::tgammaf(__x); }
__DEVICE__ long long llrint(float __x) { return ::llrintf(__x); }
__DEVICE__ long long llround(float __x) { return ::llroundf(__x); }
__DEVICE__ long lrint(float __x) { return ::lrintf(__x); }
__DEVICE__ long lround(float __x) { return ::lroundf(__x); }
__DEVICE__ float rint(float __x) { return ::rintf(__x); }
__DEVICE__ float round(float __x) { return ::roundf(__x); }
__DEVICE__ float trunc(float __x) { return ::truncf(__x); }
__DEVICE__ float nearbyint(float __x) { return ::nearbyintf(__x); }
__DEVICE__ float nextafter(float __a, float __b) {
  return ::nextafterf(__a, __b);
}
__DEVICE__ float remainder(float __a, float __b) {
  return ::remainderf(__a, __b);
}
__DEVICE__ float scalbn(float __a, int __b) { return ::scalbnf(__a, __b); }

#if defined(__OPENMP_AMDGCN__)
__DEVICE__ _Float16 fma(_Float16 __x, _Float16 __y, _Float16 __z) {
  return __builtin_fmaf16(__x, __y, __z);
}
__DEVICE__ _Float16 pow(_Float16 __base, int __iexp) {
  return __ocml_pown_f16(__base, __iexp);
}
#endif

#undef __DEVICE__
#endif // __CLANG_CMATH_H__