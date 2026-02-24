//===-- muloi5.c - Implement __muloi5 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __muloi5 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a * b

// Effects: sets *overflow to 1  if a * b overflows

#define fixint_t oi_int
#define fixuint_t ou_int
#include "int_mulo_impl.inc"

COMPILER_RT_ABI oi_int __muloi5(oi_int a, oi_int b, int *overflow) {
  return __muloXi4(a, b, overflow);
}

#endif // CRT_HAS_256BIT
