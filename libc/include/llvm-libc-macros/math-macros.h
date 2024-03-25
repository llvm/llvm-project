//===-- Definition of macros from math.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_MATH_MACROS_H
#define LLVM_LIBC_MACROS_MATH_MACROS_H

#include "limits-macros.h"

#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4

#define FP_INT_UPWARD 0
#define FP_INT_DOWNWARD 1
#define FP_INT_TOWARDZERO 2
#define FP_INT_TONEARESTFROMZERO 3
#define FP_INT_TONEAREST 4

#define MATH_ERRNO 1
#define MATH_ERREXCEPT 2

#define HUGE_VAL __builtin_huge_val()
#define INFINITY __builtin_inf()
#define NAN __builtin_nanf("")

#define FP_ILOGB0 (-INT_MAX - 1)
#define FP_ILOGBNAN INT_MAX

#define FP_LLOGB0 (-LONG_MAX - 1)
#define FP_LLOGBNAN LONG_MAX

#ifdef __FAST_MATH__
#define math_errhandling 0
#elif defined(__NO_MATH_ERRNO__)
#define math_errhandling (MATH_ERREXCEPT)
#elif defined(__NVPTX__) || defined(__AMDGPU__)
#define math_errhandling (MATH_ERRNO)
#else
#define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
#endif

// These must be type-generic functions.  The C standard specifies them as
// being macros rather than functions, in fact.  However, in C++ it's important
// that there be function declarations that don't interfere with other uses of
// the identifier, even in places with parentheses where a function-like macro
// will be expanded (such as a function declaration in a C++ namespace).

#ifdef __cplusplus

template <typename T> inline constexpr bool isfinite(T x) {
  return __builtin_isfinite(x);
}

template <typename T> inline constexpr bool isinf(T x) {
  return __builtin_isinf(x);
}

template <typename T> inline constexpr bool isnan(T x) {
  return __builtin_isnan(x);
}

#else

#define isfinite(x) __builtin_isfinite(x)
#define isinf(x) __builtin_isinf(x)
#define isnan(x) __builtin_isnan(x)

#endif

#endif // LLVM_LIBC_MACROS_MATH_MACROS_H
