//===-- ashroi3.c - Implement __ashroi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __ashroi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: arithmetic a >> b

// Precondition:  0 <= b < bits_in_oword

COMPILER_RT_ABI oi_int __ashroi3(oi_int a, int b) {
  const int bits_in_tword = (int)(sizeof(ti_int) * CHAR_BIT);
  owords input;
  owords result;
  input.all = a;
  if (b & bits_in_tword) /* bits_in_tword <= b < bits_in_oword */ {
    // result.s.high = input.s.high < 0 ? -1 : 0
    result.s.high = input.s.high >> (bits_in_tword - 1);
    result.s.low = input.s.high >> (b - bits_in_tword);
  } else /* 0 <= b < bits_in_tword */ {
    if (b == 0)
      return a;
    result.s.high = input.s.high >> b;
    result.s.low =
        ((tu_int)input.s.high << (bits_in_tword - b)) | (input.s.low >> b);
  }
  return result.all;
}

#endif // CRT_HAS_256BIT
