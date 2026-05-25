//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/float/definitions.h"
#include "clc/math/clc_subnormal_config.h"

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF bool __clc_denormals_are_zero_fp16() {
  return __builtin_isfpclass(__builtin_canonicalizef16(HALF_TRUE_MIN),
                             __FPCLASS_POSZERO);
}

#endif

_CLC_DEF bool __clc_denormals_are_zero_fp32() {
  return __builtin_isfpclass(__builtin_canonicalizef(FLT_TRUE_MIN),
                             __FPCLASS_POSZERO);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF bool __clc_denormals_are_zero_fp64() {
  return __builtin_isfpclass(__builtin_canonicalize(DBL_TRUE_MIN),
                             __FPCLASS_POSZERO);
}

#endif
