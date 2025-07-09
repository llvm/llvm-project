//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_cross.h>
#include <clc/opencl/geometric/cross.h>

_CLC_OVERLOAD _CLC_DEF float3 cross(float3 p0, float3 p1) {
  return __clc_cross(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF float4 cross(float4 p0, float4 p1) {
  return __clc_cross(p0, p1);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double3 cross(double3 p0, double3 p1) {
  return __clc_cross(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF double4 cross(double4 p0, double4 p1) {
  return __clc_cross(p0, p1);
}

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half3 cross(half3 p0, half3 p1) {
  return __clc_cross(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF half4 cross(half4 p0, half4 p1) {
  return __clc_cross(p0, p1);
}

#endif
