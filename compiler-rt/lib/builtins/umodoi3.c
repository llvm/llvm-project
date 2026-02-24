//===-- umodoi3.c - Implement __umodoi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __umodoi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a % b

COMPILER_RT_ABI ou_int __umodoi3(ou_int a, ou_int b) {
  ou_int r;
  __udivmodoi4(a, b, &r);
  return r;
}

#endif // CRT_HAS_256BIT
