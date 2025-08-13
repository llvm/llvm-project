//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_sqrt.h>

float __ocml_sqrt_f32(float);
_CLC_OVERLOAD _CLC_DEF float __clc_sqrt(float x) { return __ocml_sqrt_f32(x); }

#define __FLOAT_ONLY
#define FUNCTION __clc_sqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_sqrt_f16(half);
_CLC_OVERLOAD _CLC_DEF half __clc_sqrt(half x) { return __ocml_sqrt_f16(x); }
#endif

#define __HALF_ONLY
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
