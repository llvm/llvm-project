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
