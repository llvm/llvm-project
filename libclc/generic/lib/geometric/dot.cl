#include <clc/clc.h>
#include <clc/geometric/clc_dot.h>

_CLC_OVERLOAD _CLC_DEF float dot(float p0, float p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF float dot(float2 p0, float2 p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF float dot(float3 p0, float3 p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF float dot(float4 p0, float4 p1) {
  return __clc_dot(p0, p1);
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double dot(double p0, double p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF double dot(double2 p0, double2 p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF double dot(double3 p0, double3 p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF double dot(double4 p0, double4 p1) {
  return __clc_dot(p0, p1);
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half dot(half p0, half p1) { return __clc_dot(p0, p1); }

_CLC_OVERLOAD _CLC_DEF half dot(half2 p0, half2 p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF half dot(half3 p0, half3 p1) {
  return __clc_dot(p0, p1);
}

_CLC_OVERLOAD _CLC_DEF half dot(half4 p0, half4 p1) {
  return __clc_dot(p0, p1);
}

#endif
