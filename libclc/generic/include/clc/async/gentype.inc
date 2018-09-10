
#define __CLC_GENTYPE char
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float2
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

#define __CLC_GENTYPE long
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long16
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong16
#include __CLC_BODY
#undef __CLC_GENTYPE

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_GENTYPE double
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double2
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

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16: enable

#define __CLC_GENTYPE half
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE half2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE half4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE half8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE half16
#include __CLC_BODY
#undef __CLC_GENTYPE

#endif

#undef __CLC_BODY
