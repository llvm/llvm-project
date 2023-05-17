//===-- Definition of macros from math.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_MATH_MACROS_H
#define __LLVM_LIBC_MACROS_MATH_MACROS_H

#define MATH_ERRNO 1
#define MATH_ERREXCEPT 2

#define HUGE_VAL __builtin_huge_val()
#define INFINITY __builtin_inf()
#define NAN __builtin_nanf("")

#define FP_ILOGB0 (-__INT_MAX__ - 1)
#define FP_ILOGBNAN __INT_MAX__

#define isfinite(x) __builtin_isfinite(x)
#define isinf(x) __builtin_isinf(x)
#define isnan(x) __builtin_isnan(x)

#ifdef __FAST_MATH__
#define math_errhandling 0
#elif defined __NO_MATH_ERRNO__
#define math_errhandling (MATH_ERREXCEPT)
#else
#define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
#endif

#endif // __LLVM_LIBC_MACROS_MATH_MACROS_H
