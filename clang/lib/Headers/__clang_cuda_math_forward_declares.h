/*===- __clang_math_forward_declares.h - Prototypes of __device__ math fns --===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __CLANG__CUDA_MATH_FORWARD_DECLARES_H__
#define __CLANG__CUDA_MATH_FORWARD_DECLARES_H__
#if !defined(__CUDA__) && !__HIP__
#error "This file is for CUDA/HIP compilation only."
#endif

// PURPOSE: Forward-declare __device__ math functions before <cmath> is included.
// Prevents standard library constexpr functions from becoming implicit
// __host__ __device__, which would clash with our __device__ overloads.

// ---------------------------------------------------------------------------
// Return Type: CUDA headers return 'bool' on MSVC, but 'int' on POSIX.
// Mismatches here cause "functions differ only in return type" errors.
// ---------------------------------------------------------------------------
// CORRECTED: Force 'int' for all CUDA compilations to match CUDA SDK headers
// (math_functions.hpp), which define these as returning int regardless of host.
#if defined(__CUDA__)
#define __CUDA_CLASSIFIER_RET_TYPE int
#elif defined(__OPENMP_NVPTX__)
#define __CUDA_CLASSIFIER_RET_TYPE int
#else
#define __CUDA_CLASSIFIER_RET_TYPE int
#endif

#pragma push_macro("__DEVICE__")
#define __DEVICE__                                                             \
  static __inline__ __attribute__((always_inline)) __attribute__((device))

__DEVICE__ long abs(long);
__DEVICE__ long long abs(long long);
__DEVICE__ double abs(double);
__DEVICE__ float abs(float);
__DEVICE__ int abs(int);
__DEVICE__ double acos(double);
__DEVICE__ float acos(float);
__DEVICE__ double acosh(double);
__DEVICE__ float acosh(float);
__DEVICE__ double asin(double);
__DEVICE__ float asin(float);
__DEVICE__ double asinh(double);
__DEVICE__ float asinh(float);
__DEVICE__ double atan2(double, double);
__DEVICE__ float atan2(float, float);
__DEVICE__ double atan(double);
__DEVICE__ float atan(float);
__DEVICE__ double atanh(double);
__DEVICE__ float atanh(float);
__DEVICE__ double cbrt(double);
__DEVICE__ float cbrt(float);
__DEVICE__ double ceil(double);
__DEVICE__ float ceil(float);
__DEVICE__ double copysign(double, double);
__DEVICE__ float copysign(float, float);
__DEVICE__ double cos(double);
__DEVICE__ float cos(float);
__DEVICE__ double cosh(double);
__DEVICE__ float cosh(float);
__DEVICE__ double erfc(double);
__DEVICE__ float erfc(float);
__DEVICE__ double erf(double);
__DEVICE__ float erf(float);
__DEVICE__ double exp2(double);
__DEVICE__ float exp2(float);
__DEVICE__ double exp(double);
__DEVICE__ float exp(float);
__DEVICE__ double expm1(double);
__DEVICE__ float expm1(float);
__DEVICE__ double fabs(double);
__DEVICE__ float fabs(float);
__DEVICE__ double fdim(double, double);
__DEVICE__ float fdim(float, float);
__DEVICE__ double floor(double);
__DEVICE__ float floor(float);
__DEVICE__ double fma(double, double, double);
__DEVICE__ float fma(float, float, float);
#ifdef _MSC_VER
// long double fma variant is not actually supported by CUDA (PTX).
// However, MS-STL requires that this is forward declared anyways.
__DEVICE__ long double fma(long double, long double, long double);
#endif
__DEVICE__ double fmax(double, double);
__DEVICE__ float fmax(float, float);
__DEVICE__ double fmin(double, double);
__DEVICE__ float fmin(float, float);
__DEVICE__ double fmod(double, double);
__DEVICE__ float fmod(float, float);
__DEVICE__ int fpclassify(double);
__DEVICE__ int fpclassify(float);
__DEVICE__ double frexp(double, int *);
__DEVICE__ float frexp(float, int *);
__DEVICE__ double hypot(double, double);
__DEVICE__ float hypot(float, float);
__DEVICE__ int ilogb(double);
__DEVICE__ int ilogb(float);

// ---------------------------------------------------------------------------
// Classification Functions
// ---------------------------------------------------------------------------
// Note: We declare long double versions here if not MSVC to match
// __clang_cuda_cmath.h logic, but they require implementations in
// __clang_cuda_device_functions.h to avoid link errors.
#if !defined(_MSC_VER)
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isfinite(long double);
#endif
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isfinite(double);
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isfinite(float);
__DEVICE__ bool isgreater(double, double);
__DEVICE__ bool isgreaterequal(double, double);
__DEVICE__ bool isgreaterequal(float, float);
__DEVICE__ bool isgreater(float, float);
#if !defined(_MSC_VER)
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isinf(long double);
#endif
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isinf(double);
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isinf(float);
__DEVICE__ bool isless(double, double);
__DEVICE__ bool islessequal(double, double);
__DEVICE__ bool islessequal(float, float);
__DEVICE__ bool isless(float, float);
__DEVICE__ bool islessgreater(double, double);
__DEVICE__ bool islessgreater(float, float);
#if !defined(_MSC_VER)
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isnan(long double);
#endif
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isnan(double);
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE isnan(float);
__DEVICE__ bool isnormal(double);
__DEVICE__ bool isnormal(float);
__DEVICE__ bool isunordered(double, double);
__DEVICE__ bool isunordered(float, float);
__DEVICE__ long labs(long);
__DEVICE__ double ldexp(double, int);
__DEVICE__ float ldexp(float, int);
__DEVICE__ double lgamma(double);
__DEVICE__ float lgamma(float);
__DEVICE__ long long llabs(long long);
__DEVICE__ long long llrint(double);
__DEVICE__ long long llrint(float);
__DEVICE__ double log10(double);
__DEVICE__ float log10(float);
__DEVICE__ double log1p(double);
__DEVICE__ float log1p(float);
__DEVICE__ double log2(double);
__DEVICE__ float log2(float);
__DEVICE__ double logb(double);
__DEVICE__ float logb(float);
__DEVICE__ double log(double);
__DEVICE__ float log(float);
__DEVICE__ long lrint(double);
__DEVICE__ long lrint(float);
__DEVICE__ long lround(double);
__DEVICE__ long lround(float);
__DEVICE__ long long llround(float); // No llround(double).
__DEVICE__ double modf(double, double *);
__DEVICE__ float modf(float, float *);
__DEVICE__ double nan(const char *);
__DEVICE__ float nanf(const char *);
__DEVICE__ double nearbyint(double);
__DEVICE__ float nearbyint(float);
__DEVICE__ double nextafter(double, double);
__DEVICE__ float nextafter(float, float);
__DEVICE__ double pow(double, double);
__DEVICE__ double pow(double, int);
__DEVICE__ float pow(float, float);
__DEVICE__ float pow(float, int);
__DEVICE__ double remainder(double, double);
__DEVICE__ float remainder(float, float);
__DEVICE__ double remquo(double, double, int *);
__DEVICE__ float remquo(float, float, int *);
__DEVICE__ double rint(double);
__DEVICE__ float rint(float);
__DEVICE__ double round(double);
__DEVICE__ float round(float);
__DEVICE__ double scalbln(double, long);
__DEVICE__ float scalbln(float, long);
__DEVICE__ double scalbn(double, int);
__DEVICE__ float scalbn(float, int);
#if !defined(_MSC_VER)
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE signbit(long double);
#endif
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE signbit(double);
__DEVICE__ __CUDA_CLASSIFIER_RET_TYPE signbit(float);
__DEVICE__ double sin(double);
__DEVICE__ float sin(float);
__DEVICE__ double sinh(double);
__DEVICE__ float sinh(float);
__DEVICE__ double sqrt(double);
__DEVICE__ float sqrt(float);
__DEVICE__ double tan(double);
__DEVICE__ float tan(float);
__DEVICE__ double tanh(double);
__DEVICE__ float tanh(float);
__DEVICE__ double tgamma(double);
__DEVICE__ float tgamma(float);
__DEVICE__ double trunc(double);
__DEVICE__ float trunc(float);


// We need to define these overloads in exactly the namespace our standard
// library uses (including the right inline namespace), otherwise they won't be
// picked up by other functions in the standard library (e.g. functions in
// <complex>).  Thus the ugliness below.
#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif
#endif

using ::abs;
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
using ::isinf;
using ::isless;
using ::islessequal;
using ::islessgreater;
using ::isnan;
using ::isnormal;
using ::isunordered;
using ::labs;
using ::ldexp;
using ::lgamma;
using ::llabs;
using ::llrint;
using ::log;
using ::log10;
using ::log1p;
using ::log2;
using ::logb;
using ::lrint;
using ::lround;
using ::llround;
using ::modf;
using ::nan;
using ::nanf;
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

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_END_NAMESPACE_VERSION
#endif
} // namespace std
#endif

#pragma pop_macro("__DEVICE__")

#endif
