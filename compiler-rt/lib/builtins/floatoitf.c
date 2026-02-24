//===-- floatoitf.c - int256 -> quad-precision conversion ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements oi_int to quad-precision conversion for the
// compiler-rt library in the IEEE-754 default round-to-nearest, ties-to-even
// mode.
//
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"
#include "int_lib.h"

#if defined(CRT_HAS_TF_MODE) && defined(CRT_HAS_256BIT)
#define SRC_I256
#define DST_QUAD
#include "int_to_fp_impl.inc"

COMPILER_RT_ABI fp_t __floatoitf(oi_int a) { return __floatXiYf__(a); }

#endif
