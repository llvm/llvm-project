/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <clc/clc.h>

#include "../clcmacro.h"

_CLC_OVERLOAD _CLC_DEF float smoothstep(float edge0, float edge1, float x) {
  float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, smoothstep, float, float, float);

_CLC_V_S_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, smoothstep, float, float, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define SMOOTH_STEP_DEF(edge_type, x_type, impl) \
  _CLC_OVERLOAD _CLC_DEF x_type smoothstep(edge_type edge0, edge_type edge1, x_type x) { \
    double t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0); \
    return t * t * (3.0 - 2.0 * t); \
 }

SMOOTH_STEP_DEF(double, double, SMOOTH_STEP_IMPL_D);

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, smoothstep, double, double, double);

#if !defined(CLC_SPIRV) && !defined(CLC_SPIRV64)
SMOOTH_STEP_DEF(float, double, SMOOTH_STEP_IMPL_D);
SMOOTH_STEP_DEF(double, float, SMOOTH_STEP_IMPL_D);

_CLC_V_S_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, smoothstep, float, float, double);
_CLC_V_S_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, smoothstep, double, double, float);
#endif

#endif
