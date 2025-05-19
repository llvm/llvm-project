/*===-- flang/Common/float128.h ----------------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/

/* This header is usable in both C and C++ code.
 * Isolates build compiler checks to determine the presence of an IEEE-754
 * quad-precision type named __float128 type that isn't __ibm128
 * (double/double). We don't care whether the type has underlying hardware
 * support or is emulated.
 *
 * 128-bit arithmetic may be available via "long double"; this can
 * be determined by LDBL_MANT_DIG == 113.  A machine may have both 128-bit
 * long double and __float128; prefer long double by testing for it first.
 */

#ifndef FORTRAN_COMMON_FLOAT128_H_
#define FORTRAN_COMMON_FLOAT128_H_

#include "api-attrs.h"
#include <float.h>

#ifdef __cplusplus
/*
 * libc++ does not fully support __float128 right now, e.g.
 * std::complex<__float128> multiplication ends up calling
 * copysign() that is not defined for __float128.
 * In order to check for libc++'s _LIBCPP_VERSION macro
 * we need to include at least one libc++ header file.
 */
#include <cstddef>
#endif

#undef HAS_FLOAT128
#if (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)) && \
    !defined(_LIBCPP_VERSION) && !defined(__CUDA_ARCH__)
/*
 * It may still be worth checking for compiler versions,
 * since earlier versions may define the macros above, but
 * still do not support __float128 fully.
 */
#if __x86_64__
#if __GNUC__ >= 7 || __clang_major__ >= 7
#define HAS_FLOAT128 1
#endif
#elif defined __PPC__ && __GNUC__ >= 8
#define HAS_FLOAT128 1
#endif
#endif /* (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)) && \
          !defined(_LIBCPP_VERSION)  && !defined(__CUDA_ARCH__) */

#if LDBL_MANT_DIG == 113
#define HAS_LDBL128 1
#endif

#if defined(RT_DEVICE_COMPILATION) && defined(__CUDACC__)
/*
 * Most offload targets do not support 128-bit 'long double'.
 * Disable HAS_LDBL128 for __CUDACC__ for the time being.
 */
#undef HAS_LDBL128
#endif

/* Define pure C CFloat128Type and CFloat128ComplexType. */
#if HAS_LDBL128
typedef long double CFloat128Type;
#ifndef __cplusplus
typedef long double _Complex CFloat128ComplexType;
#endif
#elif HAS_FLOAT128
typedef __float128 CFloat128Type;

#ifndef __cplusplus
/*
 * Use mode() attribute supported by GCC and Clang.
 * Adjust it for other compilers as needed.
 */
#if !defined(_ARCH_PPC) || defined(__LONG_DOUBLE_IEEE128__)
typedef _Complex float __attribute__((mode(TC))) CFloat128ComplexType;
#else
typedef _Complex float __attribute__((mode(KC))) CFloat128ComplexType;
#endif
#endif // __cplusplus
#endif
#endif /* FORTRAN_COMMON_FLOAT128_H_ */
