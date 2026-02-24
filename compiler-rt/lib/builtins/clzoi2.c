//===-- clzoi2.c - Implement __clzoi2 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __clzoi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: the number of leading 0-bits

// Precondition: a != 0

COMPILER_RT_ABI int __clzoi2(oi_int a) {
  owords x;
  x.all = a;
  const ti_int f = -(x.s.high == 0);
  return __clzti2((x.s.high & ~f) | (x.s.low & f)) +
         ((si_int)f & ((si_int)(sizeof(ti_int) * CHAR_BIT)));
}

#endif // CRT_HAS_256BIT
