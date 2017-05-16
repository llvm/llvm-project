//===-- lib/truncdfsf2.c - double -> single conversion ------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SRC_DOUBLE
#define DST_SINGLE
#include "fp_trunc_impl.inc"

COMPILER_RT_ABI float __truncdfsf2(double a) {
    return __truncXfYf2__(a);
}

#if defined(__ARM_EABI__)
AEABI_RTABI float __aeabi_d2f(double a) {
  return __truncdfsf2(a);
}
#endif

