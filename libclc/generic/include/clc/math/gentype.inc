#define __CLC_SCALAR_GENTYPE float
#define __CLC_FPSIZE 32

#define __CLC_GENTYPE float
#define __CLC_SCALAR
#include __CLC_BODY
#undef __CLC_GENTYPE
#undef __CLC_SCALAR

#define __CLC_GENTYPE float2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float3
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float16
#include __CLC_BODY
#undef __CLC_GENTYPE

#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE

#ifdef cl_khr_fp64
#define __CLC_SCALAR_GENTYPE double
#define __CLC_FPSIZE 64

#define __CLC_SCALAR
#define __CLC_GENTYPE double
#include __CLC_BODY
#undef __CLC_GENTYPE
#undef __CLC_SCALAR

#define __CLC_GENTYPE double2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double3
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double16
#include __CLC_BODY
#undef __CLC_GENTYPE

#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE
#endif

#undef __CLC_BODY
