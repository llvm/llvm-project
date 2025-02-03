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

#define RADIANS_SINGLE_DEF(TYPE, LITERAL)                                      \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_radians(TYPE radians) {                    \
    return (TYPE)LITERAL * radians;                                            \
  }

#define RADIANS_DEF(TYPE, LITERAL)                                             \
  RADIANS_SINGLE_DEF(TYPE, LITERAL)                                            \
  RADIANS_SINGLE_DEF(TYPE##2, LITERAL)                                         \
  RADIANS_SINGLE_DEF(TYPE##3, LITERAL)                                         \
  RADIANS_SINGLE_DEF(TYPE##4, LITERAL)                                         \
  RADIANS_SINGLE_DEF(TYPE##8, LITERAL)                                         \
  RADIANS_SINGLE_DEF(TYPE##16, LITERAL)

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
RADIANS_DEF(float, 0x1.1df46ap-6F)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
RADIANS_DEF(double, 0x1.1df46a2529d39p-6)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
RADIANS_DEF(half, (half)0x1.1df46a2529d39p-6)

#endif
