//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>

#define DEGREES_SINGLE_DEF(TYPE, LITERAL)                                      \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_degrees(TYPE radians) {                    \
    return (TYPE)LITERAL * radians;                                            \
  }

#define DEGREES_DEF(TYPE, LITERAL)                                             \
  DEGREES_SINGLE_DEF(TYPE, LITERAL)                                            \
  DEGREES_SINGLE_DEF(TYPE##2, LITERAL)                                         \
  DEGREES_SINGLE_DEF(TYPE##3, LITERAL)                                         \
  DEGREES_SINGLE_DEF(TYPE##4, LITERAL)                                         \
  DEGREES_SINGLE_DEF(TYPE##8, LITERAL)                                         \
  DEGREES_SINGLE_DEF(TYPE##16, LITERAL)

// 180/pi = ~57.29577951308232087685 or 0x1.ca5dc1a63c1f8p+5 or 0x1.ca5dc2p+5F
DEGREES_DEF(float, 0x1.ca5dc2p+5F)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// 180/pi = ~57.29577951308232087685 or 0x1.ca5dc1a63c1f8p+5 or 0x1.ca5dc2p+5F
DEGREES_DEF(double, 0x1.ca5dc1a63c1f8p+5)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 180/pi = ~57.29577951308232087685 or 0x1.ca5dc1a63c1f8p+5 or 0x1.ca5dc2p+5F
DEGREES_DEF(half, (half)0x1.ca5dc1a63c1f8p+5)

#endif
