//===-- lib/truncxfbf2.c - long double -> bfloat conversion -------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(CRT_HAS_TF_MODE) && __LDBL_MANT_DIG__ == 64 && defined(__x86_64__)
#define SRC_80
#define DST_BFLOAT
#include "fp_trunc_impl.inc"

COMPILER_RT_ABI dst_t __truncxfbf2(long double a) { return __truncXfYf2__(a); }

#endif
