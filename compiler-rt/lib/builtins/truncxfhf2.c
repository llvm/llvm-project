//===-- lib/truncxfhf2.c - long double -> half conversion ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

// The 80-bit source needs 128-bit integers. Without them, convert through
// float, which rounds twice.
#if defined(CRT_HAS_TF_MODE) && __LDBL_MANT_DIG__ == 64 && defined(__x86_64__)
#define SRC_80
#else
#define SRC_SINGLE
#endif
#define DST_HALF
#include "fp_trunc_impl.inc"

COMPILER_RT_ABI dst_t __truncxfhf2(xf_float a) {
#if defined(SRC_80)
  return __truncXfYf2__(a);
#else
  return __truncXfYf2__((float)a);
#endif
}
