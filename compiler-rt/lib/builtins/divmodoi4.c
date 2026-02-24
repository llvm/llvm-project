//===-- divmodoi4.c - Implement __divmodoi4 -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __divmodoi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a / b, *rem = a % b

COMPILER_RT_ABI oi_int __divmodoi4(oi_int a, oi_int b, oi_int *rem) {
  const int bits_in_oword_m1 = (int)(sizeof(oi_int) * CHAR_BIT) - 1;
  oi_int s_a = a >> bits_in_oword_m1; // s_a = a < 0 ? -1 : 0
  oi_int s_b = b >> bits_in_oword_m1; // s_b = b < 0 ? -1 : 0
  a = (ou_int)(a ^ s_a) - s_a;        // negate if s_a == -1
  b = (ou_int)(b ^ s_b) - s_b;        // negate if s_b == -1
  s_b ^= s_a;                         // sign of quotient
  ou_int r;
  oi_int q = (__udivmodoi4(a, b, &r) ^ s_b) - s_b; // negate if s_b == -1
  *rem = (r ^ s_a) - s_a;                          // negate if s_a == -1
  return q;
}

#endif // CRT_HAS_256BIT
