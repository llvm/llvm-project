//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/float/definitions.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_fabs.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_frexp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_log2.h"
#include "clc/math/clc_mad.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_isnan.h"

/*
 *log(x) = log2(x) * (1/log2(e))
 */

_CLC_OVERLOAD _CLC_DEF float __clc_log(float x) {
  return __clc_log2(x) * (1.0f / M_LOG2E_F);
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_log(double x) {
  return __clc_log2(x) * (1.0 / M_LOG2E);
}

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_log(half x) {
  return (half)__clc_log2((float)x) * (1.0h / M_LOG2E_H);
}

#endif // cl_khr_fp16

#define __CLC_FUNCTION __clc_log
#define __CLC_BODY "clc/shared/unary_def_scalarize.inc"
#include "clc/math/gentype.inc"
