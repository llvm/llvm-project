//===-- lib/extendhftf2.c - half -> quad conversion ---------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT) &&                     \
    defined(COMPILER_RT_HAS_FLOAT16)
#define SRC_HALF
#define DST_QUAD
#include "fp_extend_impl.inc"

COMPILER_RT_ABI long double __extendhftf2(_Float16 a) {
  return __extendXfYf2__(a);
}

#endif
