#include <clc/internal/clc.h>

#define _CLC_DEFINE_ISEQUAL(RET_TYPE, FUNCTION, ARG1_TYPE, ARG2_TYPE)          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) {         \
    return (x == y);                                                           \
  }

_CLC_DEFINE_ISEQUAL(int, __clc_isequal, float, float)
_CLC_DEFINE_ISEQUAL(int2, __clc_isequal, float2, float2)
_CLC_DEFINE_ISEQUAL(int3, __clc_isequal, float3, float3)
_CLC_DEFINE_ISEQUAL(int4, __clc_isequal, float4, float4)
_CLC_DEFINE_ISEQUAL(int8, __clc_isequal, float8, float8)
_CLC_DEFINE_ISEQUAL(int16, __clc_isequal, float16, float16)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_isequal(double) returns an int, but the vector
// versions return long.
_CLC_DEFINE_ISEQUAL(int, __clc_isequal, double, double)
_CLC_DEFINE_ISEQUAL(long2, __clc_isequal, double2, double2)
_CLC_DEFINE_ISEQUAL(long3, __clc_isequal, double3, double3)
_CLC_DEFINE_ISEQUAL(long4, __clc_isequal, double4, double4)
_CLC_DEFINE_ISEQUAL(long8, __clc_isequal, double8, double8)
_CLC_DEFINE_ISEQUAL(long16, __clc_isequal, double16, double16)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_isequal(half) returns an int, but the vector
// versions return short.
_CLC_DEFINE_ISEQUAL(int, __clc_isequal, half, half)
_CLC_DEFINE_ISEQUAL(short2, __clc_isequal, half2, half2)
_CLC_DEFINE_ISEQUAL(short3, __clc_isequal, half3, half3)
_CLC_DEFINE_ISEQUAL(short4, __clc_isequal, half4, half4)
_CLC_DEFINE_ISEQUAL(short8, __clc_isequal, half8, half8)
_CLC_DEFINE_ISEQUAL(short16, __clc_isequal, half16, half16)

#endif

#undef _CLC_DEFINE_ISEQUAL
