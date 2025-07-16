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
#define HUGE_VALF __builtin_huge_valf()
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")

#define FP_ILOGB0 (-INT_MAX - 1)
#define FP_LLOGB0 (-LONG_MAX - 1)

#ifdef __FP_LOGBNAN_MIN
#define FP_ILOGBNAN (-INT_MAX - 1)
#define FP_LLOGBNAN (-LONG_MAX - 1)
#else
#define FP_ILOGBNAN INT_MAX
#define FP_LLOGBNAN LONG_MAX
#endif

#if defined(__NVPTX__) || defined(__AMDGPU__) || defined(__FAST_MATH__)
#define math_errhandling 0
#elif defined(__NO_MATH_ERRNO__)
#define math_errhandling (MATH_ERREXCEPT)
#else
#define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
#endif

// POSIX math constants
// https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/math.h.html
#define M_E (__extension__ 0x1.5bf0a8b145769p1)
#define M_EGAMMA (__extension__ 0x1.2788cfc6fb619p-1)
#define M_LOG2E (__extension__ 0x1.71547652b82fep0)
#define M_LOG10E (__extension__ 0x1.bcb7b1526e50ep-2)
#define M_LN2 (__extension__ 0x1.62e42fefa39efp-1)
#define M_LN10 (__extension__ 0x1.26bb1bbb55516p1)
#define M_PHI (__extension__ 0x1.9e3779b97f4a8p0)
#define M_PI (__extension__ 0x1.921fb54442d18p1)
#define M_PI_2 (__extension__ 0x1.921fb54442d18p0)
#define M_PI_4 (__extension__ 0x1.921fb54442d18p-1)
#define M_1_PI (__extension__ 0x1.45f306dc9c883p-2)
#define M_1_SQRTPI (__extension__ 0x1.20dd750429b6dp-1)
#define M_2_PI (__extension__ 0x1.45f306dc9c883p-1)
#define M_2_SQRTPI (__extension__ 0x1.20dd750429b6dp0)
#define M_SQRT2 (__extension__ 0x1.6a09e667f3bcdp0)
#define M_SQRT3 (__extension__ 0x1.bb67ae8584caap0)
#define M_SQRT1_2 (__extension__ 0x1.6a09e667f3bcdp-1)
#define M_SQRT1_3 (__extension__ 0x1.279a74590331cp-1)

#endif // LLVM_LIBC_MACROS_MATH_MACROS_H
