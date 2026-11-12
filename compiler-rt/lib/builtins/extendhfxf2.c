//===-- lib/extendhfxf2.c - half -> long double conversion --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "int_lib.h"
#define SRC_HALF
#define DST_DOUBLE
#include "fp_extend_impl.inc"

// Long double are expected to be as precise as double.
COMPILER_RT_ABI xf_float __extendhfxf2(src_t a) {
  return (xf_float)__extendXfYf2__(a);
}
