//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_rsqrt.h>

float __ocml_rsqrt_f32(float);

_CLC_OVERLOAD _CLC_DEF float __clc_rsqrt(float x) { return __ocml_rsqrt_f32(x); }

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_rsqrt_f64(double);

_CLC_OVERLOAD _CLC_DEF double __clc_rsqrt(double x) { return __ocml_rsqrt_f64(x); }

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_rsqrt_f16(half);

_CLC_OVERLOAD _CLC_DEF half __clc_rsqrt(half x) {
  return __ocml_rsqrt_f16(x);
}

#endif

#define FUNCTION __clc_rsqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
