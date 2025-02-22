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
#include <clc/clcmacro.h>
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
