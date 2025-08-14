//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_rsqrt.h>

float __nv_rsqrtf(float);
double __nv_rsqrt(double);

_CLC_OVERLOAD _CLC_DEF float __clc_rsqrt(float x) { return __nv_rsqrtf(x); }

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_rsqrt(double x) { return __nv_rsqrt(x); }

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_rsqrt(half x) {
  return (half)__clc_rsqrt((float)x);
}

#endif

#define FUNCTION __clc_rsqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
