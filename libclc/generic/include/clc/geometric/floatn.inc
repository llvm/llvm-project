#define __CLC_FLOAT float

#define __CLC_FLOATN float
#include __CLC_BODY
#undef __CLC_FLOATN

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

#ifdef cl_khr_fp64

#define __CLC_FLOAT double

#define __CLC_FLOATN double
#include __CLC_BODY
#undef __CLC_FLOATN

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

#endif

#undef __CLC_BODY
