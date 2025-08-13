//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_fmax.h>

float __ocml_fmax_f32(float, float);
_CLC_OVERLOAD _CLC_DEF float __clc_fmax(float x, float y) {
  return __ocml_fmax_f32(x, y);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_fmax_f64(double, double);
_CLC_OVERLOAD _CLC_DEF double __clc_fmax(double x, double y) {
  return __ocml_fmax_f64(x, y);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_fmax_f16(half, half);
_CLC_OVERLOAD _CLC_DEF half __clc_fmax(half x, half y) {
  return __ocml_fmax_f16(x, y);
}
#endif

#define FUNCTION __clc_fmax
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
