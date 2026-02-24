//===-- udivoi3.c - Implement __udivoi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __udivoi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a / b

COMPILER_RT_ABI ou_int __udivoi3(ou_int a, ou_int b) {
  return __udivmodoi4(a, b, 0);
}

#endif // CRT_HAS_256BIT
