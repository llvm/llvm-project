/*
 * Copyright (c) 2015 Advanced Micro Devices, Inc.
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

#include "math/clc_sqrt.h"
#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_DEFINE_UNARY_BUILTIN(float, sqrt, __clc_sqrt, float)

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEFINE_UNARY_BUILTIN(half, sqrt, __clc_sqrt, half)

#endif

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifdef __AMDGCN__
  #define __clc_builtin_rsq __builtin_amdgcn_rsq
#else
  #define __clc_builtin_rsq __builtin_r600_recipsqrt_ieee
#endif

_CLC_OVERLOAD _CLC_DEF double sqrt(double x) {

  uint vcc = x < 0x1p-767;
  uint exp0 = vcc ? 0x100 : 0;
  unsigned exp1 = vcc ? 0xffffff80 : 0;

  double v01 = ldexp(x, exp0);
  double v23 = __clc_builtin_rsq(v01);
  double v45 = v01 * v23;
  v23 = v23 * 0.5;

  double v67 = fma(-v23, v45, 0.5);
  v45 = fma(v45, v67, v45);
  double v89 = fma(-v45, v45, v01);
  v23 = fma(v23, v67, v23);
  v45 = fma(v89, v23, v45);
  v67 = fma(-v45, v45, v01);
  v23 = fma(v67, v23, v45);

  v23 = ldexp(v23, exp1);
  return ((x == __builtin_inf()) || (x == 0.0)) ? v01 : v23;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, sqrt, double);

#endif
