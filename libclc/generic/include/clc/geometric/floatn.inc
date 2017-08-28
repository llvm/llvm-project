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

#define __CLC_FLOATN float8
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN float16
#include __CLC_BODY
#undef __CLC_FLOATN

#undef __CLC_FLOAT
#undef __CLC_FPSIZE

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64

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

#define __CLC_FLOATN double8
#include __CLC_BODY
#undef __CLC_FLOATN

#define __CLC_FLOATN double16
#include __CLC_BODY
#undef __CLC_FLOATN

#undef __CLC_FLOAT
#undef __CLC_FPSIZE

#endif
#endif

#undef __CLC_BODY
