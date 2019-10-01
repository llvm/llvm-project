//===-- lib/extendsftf2.c - single -> quad conversion -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT)
#define SRC_SINGLE
#define DST_QUAD
#include "fp_extend_impl.inc"

COMPILER_RT_ABI long double __extendsftf2(float a) {
  return __extendXfYf2__(a);
}

#endif
