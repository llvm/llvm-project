//===-- negvoi2.c - Implement __negvoi2 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __negvoi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: -a

// Effects: aborts if -a overflows

COMPILER_RT_ABI oi_int __negvoi2(oi_int a) {
  const oi_int MIN = (ou_int)1 << ((int)(sizeof(oi_int) * CHAR_BIT) - 1);
  if (a == MIN)
    compilerrt_abort();
  return -a;
}

#endif // CRT_HAS_256BIT
