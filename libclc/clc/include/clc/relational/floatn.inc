
#define __CLC_FLOATN float
#define __CLC_INTN int
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN float2
#define __CLC_INTN int2
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN float3
#define __CLC_INTN int3
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN float4
#define __CLC_INTN int4
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN float8
#define __CLC_INTN int8
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN float16
#define __CLC_INTN int16
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#undef __CLC_FLOAT
#undef __CLC_INT

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_FLOATN double
#define __CLC_INTN int
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN double2
#define __CLC_INTN long2
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN double3
#define __CLC_INTN long3
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN double4
#define __CLC_INTN long4
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN double8
#define __CLC_INTN long8
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN double16
#define __CLC_INTN long16
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_FLOATN half
#define __CLC_INTN int
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN half2
#define __CLC_INTN short2
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN half3
#define __CLC_INTN short3
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN half4
#define __CLC_INTN short4
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN half8
#define __CLC_INTN short8
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#define __CLC_FLOATN half16
#define __CLC_INTN short16
#include __CLC_BODY
#undef __CLC_INTN
#undef __CLC_FLOATN

#endif

#undef __CLC_BODY
