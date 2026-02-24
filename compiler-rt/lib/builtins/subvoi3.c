//===-- subvoi3.c - Implement __subvoi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __subvoi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a - b

// Effects: aborts if a - b overflows

COMPILER_RT_ABI oi_int __subvoi3(oi_int a, oi_int b) {
  oi_int s = (ou_int)a - (ou_int)b;
  if (b >= 0) {
    if (s > a)
      compilerrt_abort();
  } else {
    if (s <= a)
      compilerrt_abort();
  }
  return s;
}

#endif // CRT_HAS_256BIT
