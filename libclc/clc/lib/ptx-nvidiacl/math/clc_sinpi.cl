//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_sinpi.h>

float __nv_sinpif(float);
double __nv_sinpi(double);

_CLC_OVERLOAD _CLC_DEF float __clc_sinpi(float x) { return __nv_sinpif(x); }

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_sinpi(double x) { return __nv_sinpi(x); }

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_sinpi(half x) {
  return (half)__clc_sinpi((float)x);
}

#endif

#define __CLC_FUNCTION __clc_sinpi
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
