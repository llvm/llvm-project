//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/math/math.h>

_CLC_DEF _CLC_OVERLOAD float __clc_fmax(float x, float y) {
  // Flush denormals if not enabled. Otherwise fmax instruction flushes the
  // values for comparison, but outputs original denormal
  x = __clc_flush_denormal_if_not_supported(x);
  y = __clc_flush_denormal_if_not_supported(y);
  return __builtin_fmaxf(x, y);
}

#define __FLOAT_ONLY
#define FUNCTION __clc_fmax
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __clc_fmax(double x, double y) {
  return __builtin_fmax(x, y);
}

#define __DOUBLE_ONLY
#define FUNCTION __clc_fmax
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

#endif
