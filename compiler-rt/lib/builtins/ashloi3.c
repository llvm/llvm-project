//===-- ashloi3.c - Implement __ashloi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __ashloi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns: a << b

// Precondition:  0 <= b < bits_in_oword

COMPILER_RT_ABI oi_int __ashloi3(oi_int a, int b) {
  const int bits_in_tword = (int)(sizeof(ti_int) * CHAR_BIT);
  owords input;
  owords result;
  input.all = a;
  if (b & bits_in_tword) /* bits_in_tword <= b < bits_in_oword */ {
    result.s.low = 0;
    result.s.high = input.s.low << (b - bits_in_tword);
  } else /* 0 <= b < bits_in_tword */ {
    if (b == 0)
      return a;
    result.s.low = input.s.low << b;
    result.s.high =
        ((tu_int)input.s.high << b) | (input.s.low >> (bits_in_tword - b));
  }
  return result.all;
}

#endif // CRT_HAS_256BIT
