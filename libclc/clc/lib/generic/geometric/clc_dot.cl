#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_DEF float __clc_dot(float p0, float p1) { return p0 * p1; }

_CLC_OVERLOAD _CLC_DEF float __clc_dot(float2 p0, float2 p1) {
  return p0.x * p1.x + p0.y * p1.y;
}

_CLC_OVERLOAD _CLC_DEF float __clc_dot(float3 p0, float3 p1) {
  return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z;
}

_CLC_OVERLOAD _CLC_DEF float __clc_dot(float4 p0, float4 p1) {
  return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z + p0.w * p1.w;
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_dot(double p0, double p1) {
  return p0 * p1;
}

_CLC_OVERLOAD _CLC_DEF double __clc_dot(double2 p0, double2 p1) {
  return p0.x * p1.x + p0.y * p1.y;
}

_CLC_OVERLOAD _CLC_DEF double __clc_dot(double3 p0, double3 p1) {
  return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z;
}

_CLC_OVERLOAD _CLC_DEF double __clc_dot(double4 p0, double4 p1) {
  return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z + p0.w * p1.w;
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_dot(half p0, half p1) { return p0 * p1; }

_CLC_OVERLOAD _CLC_DEF half __clc_dot(half2 p0, half2 p1) {
  return p0.x * p1.x + p0.y * p1.y;
}

_CLC_OVERLOAD _CLC_DEF half __clc_dot(half3 p0, half3 p1) {
  return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z;
}

_CLC_OVERLOAD _CLC_DEF half __clc_dot(half4 p0, half4 p1) {
  return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z + p0.w * p1.w;
}

#endif
