//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <clc/internal/clc.h>
#include <clc/shared/clc_clamp.h>

#define SMOOTHSTEP_SINGLE_DEF(edge_type, x_type, lit_suff)                     \
  _CLC_OVERLOAD _CLC_DEF x_type __clc_smoothstep(edge_type edge0,              \
                                                 edge_type edge1, x_type x) {  \
    x_type t = __clc_clamp((x - edge0) / (edge1 - edge0), 0.0##lit_suff,       \
                           1.0##lit_suff);                                     \
    return t * t * (3.0##lit_suff - 2.0##lit_suff * t);                        \
  }

#define SMOOTHSTEP_DEF(type, lit_suffix)                                       \
  SMOOTHSTEP_SINGLE_DEF(type, type, lit_suffix)                                \
  SMOOTHSTEP_SINGLE_DEF(type##2, type##2, lit_suffix)                          \
  SMOOTHSTEP_SINGLE_DEF(type##3, type##3, lit_suffix)                          \
  SMOOTHSTEP_SINGLE_DEF(type##4, type##4, lit_suffix)                          \
  SMOOTHSTEP_SINGLE_DEF(type##8, type##8, lit_suffix)                          \
  SMOOTHSTEP_SINGLE_DEF(type##16, type##16, lit_suffix)

SMOOTHSTEP_DEF(float, F)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
SMOOTHSTEP_DEF(double, );
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
SMOOTHSTEP_DEF(half, H);
#endif
