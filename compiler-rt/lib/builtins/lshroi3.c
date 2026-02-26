//===-- lshroi3.c - Implement __lshroi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __lshroi3 for the compiler_rt library.
//
// NOTE: This builtin is not called by the compiler (shift libcalls are not
// registered for i256 to avoid sanitizer link failures -- ASan embeds UBSan
// but does not link against compiler-rt builtins). It exists for direct use
// by libraries and applications that need 256-bit shift operations.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: logical a >> b

// Precondition:  0 <= b < bits_in_oword

COMPILER_RT_ABI oi_int __lshroi3(oi_int a, int b) {
  const int bits_in_tword = (int)(sizeof(ti_int) * CHAR_BIT);
  uowords input;
  uowords result;
  input.all = a;
  if (b & bits_in_tword) /* bits_in_tword <= b < bits_in_oword */ {
    result.s.high = 0;
    result.s.low = input.s.high >> (b - bits_in_tword);
  } else /* 0 <= b < bits_in_tword */ {
    if (b == 0)
      return a;
    result.s.high = input.s.high >> b;
    result.s.low = (input.s.high << (bits_in_tword - b)) | (input.s.low >> b);
  }
  return result.all;
}

#endif // CRT_HAS_256BIT
