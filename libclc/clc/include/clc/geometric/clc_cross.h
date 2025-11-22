//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_GEOMETRIC_CLC_CROSS_H__
#define __CLC_GEOMETRIC_CLC_CROSS_H__

#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_CONST _CLC_DECL float3 __clc_cross(float3 p0, float3 p1);
_CLC_OVERLOAD _CLC_CONST _CLC_DECL float4 __clc_cross(float4 p0, float4 p1);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_CONST _CLC_DECL double3 __clc_cross(double3 p0, double3 p1);
_CLC_OVERLOAD _CLC_CONST _CLC_DECL double4 __clc_cross(double4 p0, double4 p1);

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_CONST _CLC_DECL half3 __clc_cross(half3 p0, half3 p1);
_CLC_OVERLOAD _CLC_CONST _CLC_DECL half4 __clc_cross(half4 p0, half4 p1);

#endif

#endif // __CLC_GEOMETRIC_CLC_CROSS_H__
