//===-- multi5.c - Implement __multi5 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __multi5 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a * b

static oi_int __multti3(tu_int a, tu_int b) {
  owords r;
  const int bits_in_tword_2 = (int)(sizeof(ti_int) * CHAR_BIT) / 2;
  const tu_int lower_mask = (tu_int)~0 >> bits_in_tword_2;
  r.s.low = (a & lower_mask) * (b & lower_mask);
  tu_int t = (tu_int)r.s.low >> bits_in_tword_2;
  r.s.low &= lower_mask;
  t += (a >> bits_in_tword_2) * (b & lower_mask);
  r.s.low += (t & lower_mask) << bits_in_tword_2;
  r.s.high = t >> bits_in_tword_2;
  t = (tu_int)r.s.low >> bits_in_tword_2;
  r.s.low &= lower_mask;
  t += (b >> bits_in_tword_2) * (a & lower_mask);
  r.s.low += (t & lower_mask) << bits_in_tword_2;
  r.s.high += t >> bits_in_tword_2;
  r.s.high += (a >> bits_in_tword_2) * (b >> bits_in_tword_2);
  return r.all;
}

// Returns: a * b

COMPILER_RT_ABI oi_int __multi5(oi_int a, oi_int b) {
  owords x;
  x.all = a;
  owords y;
  y.all = b;
  owords r;
  r.all = __multti3(x.s.low, y.s.low);
  r.s.high += x.s.high * y.s.low + x.s.low * y.s.high;
  return r.all;
}

#endif // CRT_HAS_256BIT
