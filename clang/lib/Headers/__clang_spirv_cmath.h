 /*===---- __clang_spirv_cmath.h - SPIRV cmath decls -----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_SPIRV_CMATH_H__
#define __CLANG_SPIRV_CMATH_H__

#if !defined(__SPIRV__) && !defined(__OPENMP_SPIRV__)
#error "This file is for SPIRV OpenMP device compilation only."
#endif

#if defined(__cplusplus)
#include <limits>
#include <type_traits>
#include <utility>
#endif
#include <limits.h>
#include <stdint.h>

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

__DEVICE__ float fabs(float __x) { return ::fabsf(__x); }
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
__DEVICE__ float fmax(float __x, float __y) { return ::fmaxf(__x, __y); }
__DEVICE__ float fmin(float __x, float __y) { return ::fminf(__x, __y); }
__DEVICE__ float hypot(float __x, float __y) { return ::hypotf(__x, __y); }

#if defined(__OPENMP_SPIRV__)
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
#endif // defined(__OPENMP_SPIRV__)

__DEVICE__ bool isinf(float __x) { return ::__isinff(__x); }
__DEVICE__ bool isinf(double __x) { return ::__isinf(__x); }
__DEVICE__ bool isfinite(float __x) { return ::__finitef(__x); }
__DEVICE__ bool isfinite(double __x) { return ::__finite(__x); }
__DEVICE__ bool isnan(float __x) { return ::__isnanf(__x); }
__DEVICE__ bool isnan(double __x) { return ::__isnan(__x); }

#if defined(__OPENMP_SPIRV__)
#pragma omp end declare variant
#endif // defined(__OPENMP_SPIRV__)

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
__DEVICE__ bool isnormal(float __x) {
  return __builtin_isnormal(__x);
}
__DEVICE__ bool isnormal(double __x) {
  return __builtin_isnormal(__x);
}
__DEVICE__ bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ float modf(float __x, float *__iptr) {
  return ::modff(__x, __iptr);
}
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
__DEVICE__  bool signbit(double __x) { return ::__signbit(__x); }
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
__DEVICE__ float copysign(float __a, float __b) { return ::copysignf(__a, __b); }
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
__DEVICE__ float nextafter(float __a, float __b) { return ::nextafterf(__a, __b); }
__DEVICE__ float remainder(float __a, float __b) { return ::remainderf(__a, __b); }
__DEVICE__ float scalbn(float __a, int __b) { return ::scalbnf(__a, __b); }

#ifndef __OPENMP_SPIRV__
#pragma push_macro("__SPIRV_OVERLOAD1")
#pragma push_macro("__SPIRV_OVERLOAD2")

// __SPIRV_OVERLOAD1 is used to resolve function calls with integer argument to
// avoid compilation error due to ambiguity. e.g. floor(5) is resolved with
// floor(double).
#define __SPIRV_OVERLOAD1(__retty, __fn)                                       \
  template <typename __T>                                                      \
  __DEVICE__                                                                   \
       std::enable_if_t<std::numeric_limits<__T>::is_integer, __retty>         \
      __fn(__T __x) {                                                          \
    return ::__fn((double)__x);                                                \
  }

#define __SPIRV_OVERLOAD2(__retty, __fn)                                       \
  template <typename __T1, typename __T2>                                      \
  __DEVICE__                                                                   \
       std::enable_if_t<std::numeric_limits<__T1>::is_specialized &&           \
       std::numeric_limits<__T2>::is_specialized,                              \
                               __retty>                                        \
      __fn(__T1 __x, __T2 __y) {                                               \
    return __fn((double)__x, (double)__y);                                     \
  }

__SPIRV_OVERLOAD1(double, acos)
__SPIRV_OVERLOAD1(double, acosh)
__SPIRV_OVERLOAD1(double, asin)
__SPIRV_OVERLOAD1(double, asinh)
__SPIRV_OVERLOAD1(double, atan)
__SPIRV_OVERLOAD2(double, atan2)
__SPIRV_OVERLOAD1(double, atanh)
__SPIRV_OVERLOAD1(double, cbrt)
__SPIRV_OVERLOAD1(double, ceil)
__SPIRV_OVERLOAD2(double, copysign)
__SPIRV_OVERLOAD1(double, cos)
__SPIRV_OVERLOAD1(double, cosh)
__SPIRV_OVERLOAD1(double, erf)
__SPIRV_OVERLOAD1(double, erfc)
__SPIRV_OVERLOAD1(double, exp)
__SPIRV_OVERLOAD1(double, exp2)
__SPIRV_OVERLOAD1(double, expm1)
__SPIRV_OVERLOAD1(double, fabs)
__SPIRV_OVERLOAD2(double, fdim)
__SPIRV_OVERLOAD1(double, floor)
__SPIRV_OVERLOAD2(double, fmax)
__SPIRV_OVERLOAD2(double, fmin)
__SPIRV_OVERLOAD2(double, fmod)
__SPIRV_OVERLOAD1(int, fpclassify)
__SPIRV_OVERLOAD2(double, hypot)
__SPIRV_OVERLOAD1(int, ilogb)
__SPIRV_OVERLOAD1(bool, isfinite)
__SPIRV_OVERLOAD2(bool, isgreater)
__SPIRV_OVERLOAD2(bool, isgreaterequal)
__SPIRV_OVERLOAD1(bool, isinf)
__SPIRV_OVERLOAD2(bool, isless)
__SPIRV_OVERLOAD2(bool, islessequal)
__SPIRV_OVERLOAD2(bool, islessgreater)
__SPIRV_OVERLOAD1(bool, isnan)
__SPIRV_OVERLOAD1(bool, isnormal)
__SPIRV_OVERLOAD2(bool, isunordered)
__SPIRV_OVERLOAD1(double, lgamma)
__SPIRV_OVERLOAD1(double, log)
__SPIRV_OVERLOAD1(double, log10)
__SPIRV_OVERLOAD1(double, log1p)
__SPIRV_OVERLOAD1(double, log2)
__SPIRV_OVERLOAD1(double, logb)
__SPIRV_OVERLOAD1(long long, llrint)
__SPIRV_OVERLOAD1(long long, llround)
__SPIRV_OVERLOAD1(long, lrint)
__SPIRV_OVERLOAD1(long, lround)
__SPIRV_OVERLOAD1(double, nearbyint)
__SPIRV_OVERLOAD2(double, nextafter)
__SPIRV_OVERLOAD2(double, pow)
__SPIRV_OVERLOAD2(double, remainder)
__SPIRV_OVERLOAD1(double, rint)
__SPIRV_OVERLOAD1(double, round)
__SPIRV_OVERLOAD1(bool, signbit)
__SPIRV_OVERLOAD1(double, sin)
__SPIRV_OVERLOAD1(double, sinh)
__SPIRV_OVERLOAD1(double, sqrt)
__SPIRV_OVERLOAD1(double, tan)
__SPIRV_OVERLOAD1(double, tanh)
__SPIRV_OVERLOAD1(double, tgamma)
__SPIRV_OVERLOAD1(double, trunc)

// Overload these but don't add them to std, they are not part of cmath.
__SPIRV_OVERLOAD2(double, max)
__SPIRV_OVERLOAD2(double, min)

template <typename __T1, typename __T2, typename __T3>
__DEVICE__ std::enable_if_t<
    std::numeric_limits<__T1>::is_specialized && 
    std::numeric_limits<__T2>::is_specialized &&
    std::numeric_limits<__T3>::is_specialized,
    double>
fma(__T1 __x, __T2 __y, __T3 __z) {
  return ::fma((double)__x, (double)__y, (double)__z);
}


template <typename __T>
__DEVICE__ std::enable_if_t<std::numeric_limits<__T>::is_integer, double>
    frexp(__T __x, int *__exp) {
  return ::frexp((double)__x, __exp);
}

template <typename __T>
__DEVICE__ std::enable_if_t<std::numeric_limits<__T>::is_integer, double>
    ldexp(__T __x, int __exp) {
  return ::ldexp((double)__x, __exp);
}

template <typename __T>
__DEVICE__ std::enable_if_t<std::numeric_limits<__T>::is_integer, double>
    modf(__T __x, double *__exp) {
  return ::modf((double)__x, __exp);
}

template <typename __T1, typename __T2>
__DEVICE__ std::enable_if_t<std::numeric_limits<__T1>::is_specialized &&
                            std::numeric_limits<__T2>::is_specialized,
                            double>
    remquo(__T1 __x, __T2 __y, int *__quo) {
  return ::remquo((double)__x, (double)__y, __quo);
}

template <typename __T>
__DEVICE__ std::enable_if_t<std::numeric_limits<__T>::is_integer, double>
    scalbln(__T __x, long int __exp) {
  return ::scalbln((double)__x, __exp);
}

template <typename __T>
__DEVICE__ std::enable_if_t<std::numeric_limits<__T>::is_integer, double>
    scalbn(__T __x, int __exp) {
  return ::scalbn((double)__x, __exp);
}

#pragma pop_macro("__SPIRV_OVERLOAD1")
#pragma pop_macro("__SPIRV_OVERLOAD2")

// Define these overloads inside the namespace our standard library uses.

#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif // _GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif // _LIBCPP_BEGIN_NAMESPACE_STD

// Pull the new overloads we defined above into namespace std.
// using ::abs; - This may be considered for C++.
using ::acos;
using ::acosh;
using ::asin;
using ::asinh;
using ::atan;
using ::atan2;
using ::atanh;
using ::cbrt;
using ::ceil;
using ::copysign;
using ::cos;
using ::cosh;
using ::erf;
using ::erfc;
using ::exp;
using ::exp2;
using ::expm1;
using ::fabs;
using ::fdim;
using ::floor;
using ::fma;
using ::fmax;
using ::fmin;
using ::fmod;
using ::fpclassify;
using ::frexp;
using ::hypot;
using ::ilogb;
using ::isfinite;
using ::isgreater;
using ::isgreaterequal;
using ::isless;
using ::islessequal;
using ::islessgreater;
using ::isnormal;
using ::isunordered;
using ::ldexp;
using ::lgamma;
using ::llrint;
using ::llround;
using ::log;
using ::log10;
using ::log1p;
using ::log2;
using ::logb;
using ::lrint;
using ::lround;
using ::modf;
using ::nearbyint;
using ::nextafter;
using ::pow;
using ::remainder;
using ::remquo;
using ::rint;
using ::round;
using ::scalbln;
using ::scalbn;
using ::signbit;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;
using ::tanh;
using ::tgamma;
using ::trunc;

// Well this is fun: We need to pull these symbols in for libc++, but we can't
// pull them in with libstdc++, because its ::isinf and ::isnan are different
// than its std::isinf and std::isnan.
#ifndef __GLIBCXX__
using ::isinf;
using ::isnan;
#endif

// Finally, pull the "foobarf" functions that HIP defines into std.
using ::acosf;
using ::acoshf;
using ::asinf;
using ::asinhf;
using ::atan2f;
using ::atanf;
using ::atanhf;
using ::cbrtf;
using ::ceilf;
using ::copysignf;
using ::cosf;
using ::coshf;
using ::erfcf;
using ::erff;
using ::exp2f;
using ::expf;
using ::expm1f;
using ::fabsf;
using ::fdimf;
using ::floorf;
using ::fmaf;
using ::fmaxf;
using ::fminf;
using ::fmodf;
using ::frexpf;
using ::hypotf;
using ::ilogbf;
using ::ldexpf;
using ::lgammaf;
using ::llrintf;
using ::llroundf;hfgh fghdggf h
using ::log10f;
using ::log1pf;
using ::log2f;
using ::logbf;
using ::logf;
using ::lrintf;
using ::lroundf;
using ::modff;
using ::nearbyintf;
using ::nextafterf;
using ::powf;
using ::remainderf;
using ::remquof;
using ::rintf;
using ::roundf;
using ::scalblnf;
using ::scalbnf;
using ::sinf;
using ::sinhf;
using ::sqrtf;
using ::tanf;
using ::tanhf;
using ::tgammaf;
using ::truncf;

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_END_NAMESPACE_VERSION
#endif // _GLIBCXX_BEGIN_NAMESPACE_VERSION
} // namespace std
#endif // _LIBCPP_END_NAMESPACE_STD
#endif // ifndef __OPENMP_SPIRV__
#endif // __CLANG_SPIRV_CMATH_H__