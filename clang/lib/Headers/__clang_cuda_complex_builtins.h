/*===-- __clang_cuda_complex_builtins - CUDA impls of runtime complex fns ---===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_CUDA_COMPLEX_BUILTINS
#define __CLANG_CUDA_COMPLEX_BUILTINS

// This header defines __muldc3, __mulsc3, __divdc3, and __divsc3.  These are
// libgcc functions that clang assumes are available when compiling c99 complex
// operations.  (These implementations come from libc++, and have been modified
// to work with CUDA and OpenMP target offloading [in C and C++ mode].)

#pragma push_macro("__DEVICE__")
#if defined(__OPENMP_NVPTX__) || defined(__OPENMP_AMDGCN__) ||                 \
    defined(__OPENMP_SPIRV__)
#pragma omp declare target
#define __DEVICE__ __attribute__((noinline, nothrow, cold, weak))
#else
#define __DEVICE__ __device__ inline
#endif

#if defined(__NVPTX__)
// FIXME: NVPTX should use generic builtins.
#define _SCALBNd __nv_scalbn
#define _SCALBNf __nv_scalbnf
#define _LOGBd __nv_logb
#define _LOGBf __nv_logbf
#elif defined(__OPENMP_SPIRV__)
#define _SCALBNd __spirv_ocl_ldexp
#define _SCALBNf __spirv_ocl_ldexp
#define _LOGBd __spirv_ocl_logb
#define _LOGBf __spirv_ocl_logb
#else
#define _SCALBNd __builtin_scalbn
#define _SCALBNf __builtin_scalbnf
#define _LOGBd __builtin_logb
#define _LOGBf __builtin_logbf
#endif

#if defined(__cplusplus)
extern "C" {
#endif

__DEVICE__ double _Complex __muldc3(double __a, double __b, double __c,
                                    double __d) {
  double __ac = __a * __c;
  double __bd = __b * __d;
  double __ad = __a * __d;
  double __bc = __b * __c;
  double _Complex z;
  __real__(z) = __ac - __bd;
  __imag__(z) = __ad + __bc;
  if (__builtin_isnan(__real__(z)) && __builtin_isnan(__imag__(z))) {
    int __recalc = 0;
    if (__builtin_isinf(__a) || __builtin_isinf(__b)) {
      __a = __builtin_copysign(__builtin_isinf(__a) ? 1 : 0, __a);
      __b = __builtin_copysign(__builtin_isinf(__b) ? 1 : 0, __b);
      if (__builtin_isnan(__c))
        __c = __builtin_copysign(0, __c);
      if (__builtin_isnan(__d))
        __d = __builtin_copysign(0, __d);
      __recalc = 1;
    }
    if (__builtin_isinf(__c) || __builtin_isinf(__d)) {
      __c = __builtin_copysign(__builtin_isinf(__c) ? 1 : 0, __c);
      __d = __builtin_copysign(__builtin_isinf(__d) ? 1 : 0, __d);
      if (__builtin_isnan(__a))
        __a = __builtin_copysign(0, __a);
      if (__builtin_isnan(__b))
        __b = __builtin_copysign(0, __b);
      __recalc = 1;
    }
    if (!__recalc && (__builtin_isinf(__ac) || __builtin_isinf(__bd) ||
                      __builtin_isinf(__ad) || __builtin_isinf(__bc))) {
      if (__builtin_isnan(__a))
        __a = __builtin_copysign(0, __a);
      if (__builtin_isnan(__b))
        __b = __builtin_copysign(0, __b);
      if (__builtin_isnan(__c))
        __c = __builtin_copysign(0, __c);
      if (__builtin_isnan(__d))
        __d = __builtin_copysign(0, __d);
      __recalc = 1;
    }
    if (__recalc) {
      // Can't use std::numeric_limits<double>::infinity() -- that doesn't have
      // a device overload (and isn't constexpr before C++11, naturally).
      __real__(z) = __builtin_huge_val() * (__a * __c - __b * __d);
      __imag__(z) = __builtin_huge_val() * (__a * __d + __b * __c);
    }
  }
  return z;
}

__DEVICE__ float _Complex __mulsc3(float __a, float __b, float __c, float __d) {
  float __ac = __a * __c;
  float __bd = __b * __d;
  float __ad = __a * __d;
  float __bc = __b * __c;
  float _Complex z;
  __real__(z) = __ac - __bd;
  __imag__(z) = __ad + __bc;
  if (__builtin_isnan(__real__(z)) && __builtin_isnan(__imag__(z))) {
    int __recalc = 0;
    if (__builtin_isinf(__a) || __builtin_isinf(__b)) {
      __a = __builtin_copysignf(__builtin_isinf(__a) ? 1 : 0, __a);
      __b = __builtin_copysignf(__builtin_isinf(__b) ? 1 : 0, __b);
      if (__builtin_isnan(__c))
        __c = __builtin_copysignf(0, __c);
      if (__builtin_isnan(__d))
        __d = __builtin_copysignf(0, __d);
      __recalc = 1;
    }
    if (__builtin_isinf(__c) || __builtin_isinf(__d)) {
      __c = __builtin_copysignf(__builtin_isinf(__c) ? 1 : 0, __c);
      __d = __builtin_copysignf(__builtin_isinf(__d) ? 1 : 0, __d);
      if (__builtin_isnan(__a))
        __a = __builtin_copysignf(0, __a);
      if (__builtin_isnan(__b))
        __b = __builtin_copysignf(0, __b);
      __recalc = 1;
    }
    if (!__recalc && (__builtin_isinf(__ac) || __builtin_isinf(__bd) ||
                      __builtin_isinf(__ad) || __builtin_isinf(__bc))) {
      if (__builtin_isnan(__a))
        __a = __builtin_copysignf(0, __a);
      if (__builtin_isnan(__b))
        __b = __builtin_copysignf(0, __b);
      if (__builtin_isnan(__c))
        __c = __builtin_copysignf(0, __c);
      if (__builtin_isnan(__d))
        __d = __builtin_copysignf(0, __d);
      __recalc = 1;
    }
    if (__recalc) {
      __real__(z) = __builtin_huge_valf() * (__a * __c - __b * __d);
      __imag__(z) = __builtin_huge_valf() * (__a * __d + __b * __c);
    }
  }
  return z;
}

__DEVICE__ double _Complex __divdc3(double __a, double __b, double __c,
                                    double __d) {
  int __ilogbw = 0;
  // Can't use std::max, because that's defined in <algorithm>, and we don't
  // want to pull that in for every compile.  The CUDA headers define
  // ::max(float, float) and ::max(double, double), which is sufficient for us.
  double __logbw =
      _LOGBd(__builtin_fmax(__builtin_fabs(__c), __builtin_fabs(__d)));
  if (__builtin_isfinite(__logbw)) {
    __ilogbw = (int)__logbw;
    __c = _SCALBNd(__c, -__ilogbw);
    __d = _SCALBNd(__d, -__ilogbw);
  }
  double __denom = __c * __c + __d * __d;
  double _Complex z;
  __real__(z) = _SCALBNd((__a * __c + __b * __d) / __denom, -__ilogbw);
  __imag__(z) = _SCALBNd((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (__builtin_isnan(__real__(z)) && __builtin_isnan(__imag__(z))) {
    if ((__denom == 0.0) && (!__builtin_isnan(__a) || !__builtin_isnan(__b))) {
      __real__(z) = __builtin_copysign(__builtin_huge_val(), __c) * __a;
      __imag__(z) = __builtin_copysign(__builtin_huge_val(), __c) * __b;
    } else if ((__builtin_isinf(__a) || __builtin_isinf(__b)) &&
               __builtin_isfinite(__c) && __builtin_isfinite(__d)) {
      __a = __builtin_copysign(__builtin_isinf(__a) ? 1.0 : 0.0, __a);
      __b = __builtin_copysign(__builtin_isinf(__b) ? 1.0 : 0.0, __b);
      __real__(z) = __builtin_huge_val() * (__a * __c + __b * __d);
      __imag__(z) = __builtin_huge_val() * (__b * __c - __a * __d);
    } else if (__builtin_isinf(__logbw) && __logbw > 0.0 &&
               __builtin_isfinite(__a) && __builtin_isfinite(__b)) {
      __c = __builtin_copysign(__builtin_isinf(__c) ? 1.0 : 0.0, __c);
      __d = __builtin_copysign(__builtin_isinf(__d) ? 1.0 : 0.0, __d);
      __real__(z) = 0.0 * (__a * __c + __b * __d);
      __imag__(z) = 0.0 * (__b * __c - __a * __d);
    }
  }
  return z;
}

__DEVICE__ float _Complex __divsc3(float __a, float __b, float __c, float __d) {
  int __ilogbw = 0;
  float __logbw =
      _LOGBf(__builtin_fmaxf(__builtin_fabsf(__c), __builtin_fabsf(__d)));
  if (__builtin_isfinite(__logbw)) {
    __ilogbw = (int)__logbw;
    __c = _SCALBNf(__c, -__ilogbw);
    __d = _SCALBNf(__d, -__ilogbw);
  }
  float __denom = __c * __c + __d * __d;
  float _Complex z;
  __real__(z) = _SCALBNf((__a * __c + __b * __d) / __denom, -__ilogbw);
  __imag__(z) = _SCALBNf((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (__builtin_isnan(__real__(z)) && __builtin_isnan(__imag__(z))) {
    if ((__denom == 0) && (!__builtin_isnan(__a) || !__builtin_isnan(__b))) {
      __real__(z) = __builtin_copysignf(__builtin_huge_valf(), __c) * __a;
      __imag__(z) = __builtin_copysignf(__builtin_huge_valf(), __c) * __b;
    } else if ((__builtin_isinf(__a) || __builtin_isinf(__b)) &&
               __builtin_isfinite(__c) && __builtin_isfinite(__d)) {
      __a = __builtin_copysignf(__builtin_isinf(__a) ? 1 : 0, __a);
      __b = __builtin_copysignf(__builtin_isinf(__b) ? 1 : 0, __b);
      __real__(z) = __builtin_huge_valf() * (__a * __c + __b * __d);
      __imag__(z) = __builtin_huge_valf() * (__b * __c - __a * __d);
    } else if (__builtin_isinf(__logbw) && __logbw > 0 &&
               __builtin_isfinite(__a) && __builtin_isfinite(__b)) {
      __c = __builtin_copysignf(__builtin_isinf(__c) ? 1 : 0, __c);
      __d = __builtin_copysignf(__builtin_isinf(__d) ? 1 : 0, __d);
      __real__(z) = 0 * (__a * __c + __b * __d);
      __imag__(z) = 0 * (__b * __c - __a * __d);
    }
  }
  return z;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#undef _SCALBNd
#undef _SCALBNf
#undef _LOGBd
#undef _LOGBf

#if defined(__OPENMP_NVPTX__) || defined(__OPENMP_AMDGCN__) ||                 \
    defined(__OPENMP_SPIRV__)
#pragma omp end declare target
#endif

#pragma pop_macro("__DEVICE__")

#endif // __CLANG_CUDA_COMPLEX_BUILTINS
