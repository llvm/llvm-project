//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_sqrt.h"

_CLC_OVERLOAD _CLC_DEF float __clc_sqrt(float x) { return __builtin_sqrtf(x); }

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_sqrt(double x) { return __builtin_sqrt(x); }

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_sqrt(half x) {
  return (half)__clc_sqrt((float)x);
}

#endif

#define __CLC_FUNCTION __clc_sqrt
#define __CLC_BODY "clc/shared/unary_def_scalarize.inc"
#include "clc/math/gentype.inc"
