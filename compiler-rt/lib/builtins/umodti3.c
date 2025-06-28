//===-- umodti3.c - Implement __umodti3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __umodti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT

/// Returns: a % b
COMPILER_RT_ABI tu_int __umodti3(tu_int a, tu_int b) {
  utwords aWords = {.all = a};
  utwords bWords = {.all = b};

  // Optimization based on modular arithmetic:
  //   (A_hi * 2^64 + A_lo) % B   =   ((A_hi % B) * 2^64 + A_lo) % B
  //
  // The first division on the right hand side is a 64-bit division.
  // The second division has a 64-bit quotient, so divq is safe to use.
  // Even if we don't have divq, it's faster to reduce the dividend bits prior
  // to a 128-bit software division.
  aWords.s.high = aWords.s.high % bWords.s.low;

#if defined(__x86_64__)
  du_int quotient;
  du_int remainder;
  __asm__("divq %[v]"
          : "=a"(quotient), "=d"(remainder)
          : [v] "r"(b), "a"(aWords.s.low), "d"(aWords.s.high));
  (void)quotient;
  return remainder;
#else
  tu_int remainder;
  __udivmodti4(aWords.all, b, &remainder);
  return (du_int)remainder;
#endif
}

#endif // CRT_HAS_128BIT
