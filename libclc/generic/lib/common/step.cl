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

STEP_DEF(float, double);
STEP_DEF(double, float);

_CLC_V_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, step, float, double);
_CLC_V_S_V_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, step, double, float);

#endif
