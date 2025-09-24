//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>

#define __CLC_RADIANS_SINGLE_DEF(TYPE, LITERAL)                                \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_radians(TYPE radians) {                    \
    return (TYPE)LITERAL * radians;                                            \
  }

#define __CLC_RADIANS_DEF(TYPE, LITERAL)                                       \
  __CLC_RADIANS_SINGLE_DEF(TYPE, LITERAL)                                      \
  __CLC_RADIANS_SINGLE_DEF(TYPE##2, LITERAL)                                   \
  __CLC_RADIANS_SINGLE_DEF(TYPE##3, LITERAL)                                   \
  __CLC_RADIANS_SINGLE_DEF(TYPE##4, LITERAL)                                   \
  __CLC_RADIANS_SINGLE_DEF(TYPE##8, LITERAL)                                   \
  __CLC_RADIANS_SINGLE_DEF(TYPE##16, LITERAL)

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
__CLC_RADIANS_DEF(float, 0x1.1df46ap-6F)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
__CLC_RADIANS_DEF(double, 0x1.1df46a2529d39p-6)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
__CLC_RADIANS_DEF(half, (half)0x1.1df46a2529d39p-6)

#endif
