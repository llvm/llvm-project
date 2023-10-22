//===-- lib/extendxftf2.c - long double -> quad conversion --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Assumption: long double is a IEEE 80 bit floating point type padded to 128
// bits.

// TODO: use fp_lib.h once QUAD_PRECISION is available on x86_64.
#if __LDBL_MANT_DIG__ == 64 && defined(__x86_64__) &&                          \
    (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))
#define SRC_80
#define DST_QUAD
#include "fp_extend_impl.inc"

COMPILER_RT_ABI __float128 __extendxftf2(long double a) {
  return __extendXfYf2__(a);
}

#endif
