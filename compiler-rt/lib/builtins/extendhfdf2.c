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

// Use a forwarding definition and noinline to implement a poor man's alias,
// as there isn't a good cross-platform way of defining one.
COMPILER_RT_ABI NOINLINE float __extendhfdf2(src_t a) {
  return __extendXfYf2__(a);
}

COMPILER_RT_ABI float __gnu_h2d_ieee(src_t a) { return __extendhfdf2(a); }

#if defined(__ARM_EABI__)
#if defined(COMPILER_RT_ARMHF_TARGET)
AEABI_RTABI float __aeabi_h2d(src_t a) { return __extendhfdf2(a); }
#else
COMPILER_RT_ALIAS(__extendhfdf2, __aeabi_h2d)
#endif
#endif
