_CLC_OVERLOAD float __CLC_FUNCTION(float f) __asm(__CLC_INTRINSIC ".f32");
_CLC_OVERLOAD float2 __CLC_FUNCTION(float2 f) __asm(__CLC_INTRINSIC ".v2f32");
_CLC_OVERLOAD float3 __CLC_FUNCTION(float3 f) __asm(__CLC_INTRINSIC ".v3f32");
_CLC_OVERLOAD float4 __CLC_FUNCTION(float4 f) __asm(__CLC_INTRINSIC ".v4f32");
_CLC_OVERLOAD float8 __CLC_FUNCTION(float8 f) __asm(__CLC_INTRINSIC ".v8f32");
_CLC_OVERLOAD float16 __CLC_FUNCTION(float16 f) __asm(__CLC_INTRINSIC ".v16f32");

#ifdef cl_khr_fp64
_CLC_OVERLOAD double __CLC_FUNCTION(double d) __asm(__CLC_INTRINSIC ".f64");
_CLC_OVERLOAD double2 __CLC_FUNCTION(double2 d) __asm(__CLC_INTRINSIC ".v2f64");
_CLC_OVERLOAD double3 __CLC_FUNCTION(double3 d) __asm(__CLC_INTRINSIC ".v3f64");
_CLC_OVERLOAD double4 __CLC_FUNCTION(double4 d) __asm(__CLC_INTRINSIC ".v4f64");
_CLC_OVERLOAD double8 __CLC_FUNCTION(double8 d) __asm(__CLC_INTRINSIC ".v8f64");
_CLC_OVERLOAD double16 __CLC_FUNCTION(double16 d) __asm(__CLC_INTRINSIC ".v16f64");
#endif

#undef __CLC_FUNCTION
#undef __CLC_INTRINSIC
