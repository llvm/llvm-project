//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __CLC_MATH_CLC_SUBNORMAL_CONFIG_H__
#define __CLC_MATH_CLC_SUBNORMAL_CONFIG_H__

#include "clc/clcfunc.h"

#ifdef cl_khr_fp16
_CLC_DECL bool __clc_denormals_are_zero_fp16();
#endif

_CLC_DECL bool __clc_denormals_are_zero_fp32();
_CLC_DECL bool __clc_denormals_are_zero_fp64();

#endif // __CLC_MATH_CLC_SUBNORMAL_CONFIG_H__
