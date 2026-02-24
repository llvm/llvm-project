//===-- parityoi2.c - Implement __parityoi2 -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __parityoi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: 1 if number of bits is odd else returns 0

COMPILER_RT_ABI int __parityoi2(oi_int a) {
  owords x;
  x.all = a;
  // XOR the two 128-bit halves, then delegate to parityti2's approach.
  tu_int x2 = x.s.high ^ x.s.low;
  // XOR the two 64-bit halves of the 128-bit result.
  dwords x3;
  utwords t;
  t.all = x2;
  x3.all = t.s.high ^ t.s.low;
  su_int x4 = x3.s.high ^ x3.s.low;
  x4 ^= x4 >> 16;
  x4 ^= x4 >> 8;
  x4 ^= x4 >> 4;
  return (0x6996 >> (x4 & 0xF)) & 1;
}

#endif // CRT_HAS_256BIT
