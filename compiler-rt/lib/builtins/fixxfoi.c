//===-- fixxfoi.c - Implement __fixxfoi -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __fixxfoi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: convert a to a signed 256-bit integer, rounding toward zero.

// Assumption: long double is an intel 80 bit floating point type padded with 6
// bytes oi_int is a 256 bit integral type value in long double is representable
// in oi_int

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee
// eeee | 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm
// mmmm mmmm mmmm

COMPILER_RT_ABI oi_int __fixxfoi(xf_float a) {
  const oi_int oi_max = (oi_int)((~(ou_int)0) / 2);
  const oi_int oi_min = -oi_max - 1;
  xf_bits fb;
  fb.f = a;
  int e = (fb.u.high.s.low & 0x00007FFF) - 16383;
  if (e < 0)
    return 0;
  oi_int s = -(si_int)((fb.u.high.s.low & 0x00008000) >> 15);
  oi_int r = fb.u.low.all;
  if ((unsigned)e >= sizeof(oi_int) * CHAR_BIT)
    return a > 0 ? oi_max : oi_min;
  if (e > 63)
    r <<= (e - 63);
  else
    r >>= (63 - e);
  return (r ^ s) - s;
}

#endif // CRT_HAS_256BIT
