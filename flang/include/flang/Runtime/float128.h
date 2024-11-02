/*===-- flang/Runtime/float128.h ----------------------------------*- C -*-===
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

#ifndef FORTRAN_RUNTIME_FLOAT128_H_
#define FORTRAN_RUNTIME_FLOAT128_H_

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
    !defined(_LIBCPP_VERSION)
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
          !defined(_LIBCPP_VERSION) */

#endif /* FORTRAN_RUNTIME_FLOAT128_H_ */
