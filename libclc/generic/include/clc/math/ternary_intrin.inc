_CLC_OVERLOAD float __CLC_FUNCTION(float, float, float) __asm(__CLC_INTRINSIC ".f32");
_CLC_OVERLOAD float2 __CLC_FUNCTION(float2, float2, float2) __asm(__CLC_INTRINSIC ".v2f32");
_CLC_OVERLOAD float3 __CLC_FUNCTION(float3, float3, float3) __asm(__CLC_INTRINSIC ".v3f32");
_CLC_OVERLOAD float4 __CLC_FUNCTION(float4, float4, float4) __asm(__CLC_INTRINSIC ".v4f32");
_CLC_OVERLOAD float8 __CLC_FUNCTION(float8, float8, float8) __asm(__CLC_INTRINSIC ".v8f32");
_CLC_OVERLOAD float16 __CLC_FUNCTION(float16, float16, float16) __asm(__CLC_INTRINSIC ".v16f32");

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_OVERLOAD double __CLC_FUNCTION(double, double, double) __asm(__CLC_INTRINSIC ".f64");
_CLC_OVERLOAD double2 __CLC_FUNCTION(double2, double2, double2) __asm(__CLC_INTRINSIC ".v2f64");
_CLC_OVERLOAD double3 __CLC_FUNCTION(double3, double3, double3) __asm(__CLC_INTRINSIC ".v3f64");
_CLC_OVERLOAD double4 __CLC_FUNCTION(double4, double4, double4) __asm(__CLC_INTRINSIC ".v4f64");
_CLC_OVERLOAD double8 __CLC_FUNCTION(double8, double8, double8) __asm(__CLC_INTRINSIC ".v8f64");
_CLC_OVERLOAD double16 __CLC_FUNCTION(double16, double16, double16) __asm(__CLC_INTRINSIC ".v16f64");
#endif

#undef __CLC_FUNCTION
#undef __CLC_INTRINSIC
