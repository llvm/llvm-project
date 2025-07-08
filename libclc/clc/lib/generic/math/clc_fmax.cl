//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/relational/clc_isnan.h>

#define __FLOAT_ONLY
#define __CLC_MIN_VECSIZE 1
#define FUNCTION __clc_fmax
#define __IMPL_FUNCTION __builtin_fmaxf
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_MIN_VECSIZE
#undef FUNCTION
#undef __IMPL_FUNCTION

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __DOUBLE_ONLY
#define __CLC_MIN_VECSIZE 1
#define FUNCTION __clc_fmax
#define __IMPL_FUNCTION __builtin_fmax
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_MIN_VECSIZE
#undef FUNCTION
#undef __IMPL_FUNCTION

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __clc_fmax(half x, half y) {
  if (__clc_isnan(x))
    return y;
  if (__clc_isnan(y))
    return x;
  return (x < y) ? y : x;
}

#define __HALF_ONLY
#define __CLC_SUPPORTED_VECSIZE_OR_1 2
#define FUNCTION __clc_fmax
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

#endif
