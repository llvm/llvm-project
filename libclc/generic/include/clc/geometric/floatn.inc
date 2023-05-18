#define __CLC_FLOAT float
#define __CLC_FPSIZE 32

#define __CLC_FLOATN float
#define __CLC_SCALAR
#include __CLC_BODY
#undef __CLC_FLOATN
#undef __CLC_SCALAR

#define __CLC_FLOATN float2
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN float3
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN float4
#include __CLC_BODY
#undef __CLC_FLOATN

#undef __CLC_FLOAT
#undef __CLC_FPSIZE

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_FLOAT double
#define __CLC_FPSIZE 64

#define __CLC_FLOATN double
#define __CLC_SCALAR
#include __CLC_BODY
#undef __CLC_FLOATN
#undef __CLC_SCALAR

#define __CLC_FLOATN double2
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN double3
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN double4
#include __CLC_BODY
#undef __CLC_FLOATN

#undef __CLC_FLOAT
#undef __CLC_FPSIZE

#endif
#endif

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_FLOAT half
#define __CLC_FPSIZE 16

#define __CLC_FLOATN half
#define __CLC_SCALAR
#include __CLC_BODY
#undef __CLC_FLOATN
#undef __CLC_SCALAR

#define __CLC_FLOATN half2
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN half3
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN half4
#include __CLC_BODY
#undef __CLC_FLOATN

#undef __CLC_FLOAT
#undef __CLC_FPSIZE

#endif
#endif

#undef __CLC_BODY
