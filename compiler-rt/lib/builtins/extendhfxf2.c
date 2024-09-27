//===-- lib/extendhfxf2.c - half -> long double conversion -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SRC_HALF
#define DST_DOUBLE
#include "fp_extend_impl.inc"

// Use a forwarding definition and noinline to implement a poor man's alias,
// as there isn't a good cross-platform way of defining one.
// Long double are expected to be as precise as double.
COMPILER_RT_ABI NOINLINE long double __extendhfxf2(src_t a) {
  return (long double)__extendXfYf2__(a);
}
