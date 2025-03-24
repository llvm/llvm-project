//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float step(float edge, float x) {
  return x < edge ? 0.0f : 1.0f;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, step, float, float);

_CLC_V_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, step, float, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define STEP_DEF(edge_type, x_type) \
  _CLC_OVERLOAD _CLC_DEF x_type step(edge_type edge, x_type x) { \
    return x < edge ? 0.0 : 1.0; \
 }

STEP_DEF(double, double);

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, step, double, double);
_CLC_V_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, step, double, double);

#if !defined(CLC_SPIRV)
STEP_DEF(float, double);
STEP_DEF(double, float);

_CLC_V_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, step, float, double);
_CLC_V_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, step, double, float);
#endif

#endif
