//===-- lib/extendhfdf2.c - half -> single conversion -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SRC_HALF
#define DST_DOUBLE
#include "fp_extend_impl.inc"

COMPILER_RT_ABI NOINLINE dst_t __extendhfdf2(src_t a) {
  return __extendXfYf2__(a);
}
