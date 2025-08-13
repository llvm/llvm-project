//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_exp10.h>

float __ocml_exp10_f32(float);
_CLC_OVERLOAD _CLC_DEF float __clc_exp10(float x) {
  return __ocml_exp10_f32(x);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_exp10_f64(double);
_CLC_OVERLOAD _CLC_DEF double __clc_exp10(double x) {
  return __ocml_exp10_f64(x);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_exp10_f16(half);
_CLC_OVERLOAD _CLC_DEF half __clc_exp10(half x) { return __ocml_exp10_f16(x); }
#endif

#define FUNCTION __clc_exp10
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
