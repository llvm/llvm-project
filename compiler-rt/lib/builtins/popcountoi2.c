//===-- popcountoi2.c - Implement __popcountoi2 ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __popcountoi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI int __popcountti2(ti_int a);

// Returns: count of 1 bits

COMPILER_RT_ABI int __popcountoi2(oi_int a) {
  uowords x;
  x.all = (ou_int)a;
  return __popcountti2(x.s.low) + __popcountti2(x.s.high);
}

#endif // CRT_HAS_256BIT
