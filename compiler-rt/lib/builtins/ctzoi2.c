//===-- ctzoi2.c - Implement __ctzoi2 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __ctzoi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: the number of trailing 0-bits

// Precondition: a != 0

COMPILER_RT_ABI int __ctzoi2(oi_int a) {
  owords x;
  x.all = a;
  if (x.s.low != 0)
    return __ctzti2(x.s.low);
  return __ctzti2(x.s.high) + (int)(sizeof(ti_int) * CHAR_BIT);
}

#endif // CRT_HAS_256BIT
