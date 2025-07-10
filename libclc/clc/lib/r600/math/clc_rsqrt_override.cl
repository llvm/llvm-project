//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_DEF float __clc_rsqrt(float x) {
  return __builtin_r600_recipsqrt_ieeef(x);
}

#define __FLOAT_ONLY
#define FUNCTION __clc_rsqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_rsqrt(double x) {
  return __builtin_r600_recipsqrt_ieee(x);
}

#define __DOUBLE_ONLY
#define FUNCTION __clc_rsqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

#endif
