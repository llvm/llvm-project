//===-- Definition of function macros from math.h -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_MATH_FUNCTION_MACROS_H
#define LLVM_LIBC_MACROS_MATH_FUNCTION_MACROS_H

#include "math-macros.h"

#define isfinite(x) __builtin_isfinite(x)
#define isinf(x) __builtin_isinf(x)
#define isnan(x) __builtin_isnan(x)
#define signbit(x) __builtin_signbit(x)
#define iszero(x) (x == 0)
#define fpclassify(x)                                                          \
  __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x)
#define isnormal(x) __builtin_isnormal(x)

#endif // LLVM_LIBC_MACROS_MATH_FUNCTION_MACROS_H
