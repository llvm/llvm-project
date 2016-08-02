#define __CLC_SCALAR_GENTYPE float
#define __CLC_FPSIZE 32

#define __CLC_GENTYPE float
#define __CLC_INTN int
#define __CLC_SCALAR
#include __CLC_BODY
#undef __CLC_GENTYPE
#undef __CLC_INTN
#undef __CLC_SCALAR

#define __CLC_GENTYPE float2
#define __CLC_INTN int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE float3
#define __CLC_INTN int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE float4
#define __CLC_INTN int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE float8
#define __CLC_INTN int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE float16
#define __CLC_INTN int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64
#define __CLC_SCALAR_GENTYPE double
#define __CLC_FPSIZE 64

#define __CLC_SCALAR
#define __CLC_GENTYPE double
#define __CLC_INTN int
#include __CLC_BODY
#undef __CLC_GENTYPE
#undef __CLC_INTN
#undef __CLC_SCALAR

#define __CLC_GENTYPE double2
#define __CLC_INTN int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE double3
#define __CLC_INTN int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE double4
#define __CLC_INTN int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE double8
#define __CLC_INTN int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#define __CLC_GENTYPE double16
#define __CLC_INTN int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_INTN

#undef __CLC_FPSIZE
#undef __CLC_SCALAR_GENTYPE
#endif

#undef __CLC_BODY
#endif
