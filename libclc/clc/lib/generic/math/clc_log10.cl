//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/math/tables.h>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif // cl_khr_fp16

#define COMPILING_LOG10
#include "clc_log_base.h"
#undef COMPILING_LOG10

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __clc_log10, float);

#ifdef cl_khr_fp64
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __clc_log10, double);
#endif // cl_khr_fp64

#ifdef cl_khr_fp16
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __clc_log10, half);
#endif // cl_khr_fp16
