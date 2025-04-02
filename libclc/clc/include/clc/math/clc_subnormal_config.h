//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __CLC_MATH_CLC_SUBNORMAL_CONFIG_H__
#define __CLC_MATH_CLC_SUBNORMAL_CONFIG_H__

#include <clc/clcfunc.h>

_CLC_DECL bool __clc_subnormals_disabled();
_CLC_DECL bool __clc_fp16_subnormals_supported();
_CLC_DECL bool __clc_fp32_subnormals_supported();
_CLC_DECL bool __clc_fp64_subnormals_supported();

#endif // __CLC_MATH_CLC_SUBNORMAL_CONFIG_H__
