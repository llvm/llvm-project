//===-- negoi2.c - Implement __negoi2 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __negoi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: -a

COMPILER_RT_ABI oi_int __negoi2(oi_int a) {
  // Note: this routine is here for API compatibility; any sane compiler
  // should expand it inline.
  return -(ou_int)a;
}

#endif // CRT_HAS_256BIT
