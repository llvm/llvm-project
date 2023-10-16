//===-- lib/trunctfsf2.c - long double -> quad conversion ---------*- C -*-===//
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

#define SRC_QUAD
#define DST_80
#include "fp_trunc_impl.inc"

COMPILER_RT_ABI long double __trunctfxf2(__float128 a) {
  return __truncXfYf2__(a);
}

#endif
