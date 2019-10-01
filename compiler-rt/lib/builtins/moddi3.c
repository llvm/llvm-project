//===-- moddi3.c - Implement __moddi3 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __moddi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

// Returns: a % b

COMPILER_RT_ABI di_int __moddi3(di_int a, di_int b) {
  const int bits_in_dword_m1 = (int)(sizeof(di_int) * CHAR_BIT) - 1;
  di_int s = b >> bits_in_dword_m1; // s = b < 0 ? -1 : 0
  b = (b ^ s) - s;                  // negate if s == -1
  s = a >> bits_in_dword_m1;        // s = a < 0 ? -1 : 0
  a = (a ^ s) - s;                  // negate if s == -1
  du_int r;
  __udivmoddi4(a, b, &r);
  return ((di_int)r ^ s) - s; // negate if s == -1
}
