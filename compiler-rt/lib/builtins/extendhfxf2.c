//===-- lib/extendhfxf2.c - half -> x86 FP80 conversion -----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define FP80_PRECISION
#include "fp_lib.h"

#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_80BIT)

#define SRC_HALF
#define DST_FP80
#include "fp_extend_impl.inc"

// Use a forwarding definition and noinline to implement a poor man's alias,
// as there isn't a good cross-platform way of defining one.
COMPILER_RT_ABI NOINLINE long double __extendhfxf2(src_t a) {
  return __extendXfYf2__(a);
}

#endif
