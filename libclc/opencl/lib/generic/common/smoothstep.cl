//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/common/clc_smoothstep.h>
#include <clc/opencl/common/smoothstep.h>

#define SMOOTHSTEP_SINGLE_DEF(X_TYPE)                                          \
  _CLC_OVERLOAD _CLC_DEF X_TYPE smoothstep(X_TYPE edge0, X_TYPE edge1,         \
                                           X_TYPE x) {                         \
    return __clc_smoothstep(edge0, edge1, x);                                  \
  }

#define SMOOTHSTEP_S_S_V_DEFS(X_TYPE)                                          \
  _CLC_OVERLOAD _CLC_DEF X_TYPE##2 smoothstep(X_TYPE x, X_TYPE y,              \
                                              X_TYPE##2 z) {                   \
    return __clc_smoothstep((X_TYPE##2)x, (X_TYPE##2)y, z);                    \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF X_TYPE##3 smoothstep(X_TYPE x, X_TYPE y,              \
                                              X_TYPE##3 z) {                   \
    return __clc_smoothstep((X_TYPE##3)x, (X_TYPE##3)y, z);                    \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF X_TYPE##4 smoothstep(X_TYPE x, X_TYPE y,              \
                                              X_TYPE##4 z) {                   \
    return __clc_smoothstep((X_TYPE##4)x, (X_TYPE##4)y, z);                    \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF X_TYPE##8 smoothstep(X_TYPE x, X_TYPE y,              \
                                              X_TYPE##8 z) {                   \
    return __clc_smoothstep((X_TYPE##8)x, (X_TYPE##8)y, z);                    \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF X_TYPE##16 smoothstep(X_TYPE x, X_TYPE y,             \
                                               X_TYPE##16 z) {                 \
    return __clc_smoothstep((X_TYPE##16)x, (X_TYPE##16)y, z);                  \
  }

#define SMOOTHSTEP_DEF(type)                                                   \
  SMOOTHSTEP_SINGLE_DEF(type)                                                  \
  SMOOTHSTEP_SINGLE_DEF(type##2)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##3)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##4)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##8)                                               \
  SMOOTHSTEP_SINGLE_DEF(type##16)                                              \
  SMOOTHSTEP_S_S_V_DEFS(type)

SMOOTHSTEP_DEF(float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

SMOOTHSTEP_DEF(double);

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

SMOOTHSTEP_DEF(half);

#endif
