/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
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

_CLC_OVERLOAD _CLC_DEF float length(float p) {
  return fabs(p);
}

#define V_FLENGTH(p)                     \
  float l2 = dot(p, p);                  \
                                         \
  if (l2 < FLT_MIN) {                    \
    p *= 0x1.0p+86F;                     \
    return sqrt(dot(p, p)) * 0x1.0p-86F; \
  } else if (l2 == INFINITY) {           \
    p *= 0x1.0p-65F;                     \
    return sqrt(dot(p, p)) * 0x1.0p+65F; \
  }                                      \
                                         \
  return sqrt(l2);

_CLC_OVERLOAD _CLC_DEF float length(float2 p) {
  V_FLENGTH(p);
}

_CLC_OVERLOAD _CLC_DEF float length(float3 p) {
  V_FLENGTH(p);
}

_CLC_OVERLOAD _CLC_DEF float length(float4 p) {
  V_FLENGTH(p);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double length(double p){
  return fabs(p);
}

#define V_DLENGTH(p)                       \
  double l2 = dot(p, p);                   \
                                           \
  if (l2 < DBL_MIN) {                      \
      p *= 0x1.0p+563;                     \
      return sqrt(dot(p, p)) * 0x1.0p-563; \
  } else if (l2 == INFINITY) {             \
      p *= 0x1.0p-513;                     \
      return sqrt(dot(p, p)) * 0x1.0p+513; \
  }                                        \
                                           \
  return sqrt(l2);

_CLC_OVERLOAD _CLC_DEF double length(double2 p) {
  V_DLENGTH(p);
}

_CLC_OVERLOAD _CLC_DEF double length(double3 p) {
  V_DLENGTH(p);
}

_CLC_OVERLOAD _CLC_DEF double length(double4 p) {
  V_DLENGTH(p);
}

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half length(half p){
  return fabs(p);
}

// Only available in CLC1.2
#ifndef HALF_MIN
#define HALF_MIN   0x1.0p-14h
#endif

#define V_HLENGTH(p)                       \
  half l2 = dot(p, p);                     \
                                           \
  if (l2 < HALF_MIN) {                     \
      p *= 0x1.0p+12h;                     \
      return sqrt(dot(p, p)) * 0x1.0p-12h; \
  } else if (l2 == INFINITY) {             \
      p *= 0x1.0p-7h;                      \
      return sqrt(dot(p, p)) * 0x1.0p+7h;  \
  }                                        \
                                           \
  return sqrt(l2);

_CLC_OVERLOAD _CLC_DEF half length(half2 p) {
  V_HLENGTH(p);
}

_CLC_OVERLOAD _CLC_DEF half length(half3 p) {
  V_HLENGTH(p);
}

_CLC_OVERLOAD _CLC_DEF half length(half4 p) {
  V_HLENGTH(p);
}

#endif
