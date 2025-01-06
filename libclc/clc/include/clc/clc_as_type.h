#ifndef __CLC_CLC_AS_TYPE_H__
#define __CLC_CLC_AS_TYPE_H__

#define __clc_as_char(x) __builtin_astype(x, char)
#define __clc_as_uchar(x) __builtin_astype(x, uchar)
#define __clc_as_short(x) __builtin_astype(x, short)
#define __clc_as_ushort(x) __builtin_astype(x, ushort)
#define __clc_as_int(x) __builtin_astype(x, int)
#define __clc_as_uint(x) __builtin_astype(x, uint)
#define __clc_as_long(x) __builtin_astype(x, long)
#define __clc_as_ulong(x) __builtin_astype(x, ulong)
#define __clc_as_float(x) __builtin_astype(x, float)

#define __clc_as_char2(x) __builtin_astype(x, char2)
#define __clc_as_uchar2(x) __builtin_astype(x, uchar2)
#define __clc_as_short2(x) __builtin_astype(x, short2)
#define __clc_as_ushort2(x) __builtin_astype(x, ushort2)
#define __clc_as_int2(x) __builtin_astype(x, int2)
#define __clc_as_uint2(x) __builtin_astype(x, uint2)
#define __clc_as_long2(x) __builtin_astype(x, long2)
#define __clc_as_ulong2(x) __builtin_astype(x, ulong2)
#define __clc_as_float2(x) __builtin_astype(x, float2)

#define __clc_as_char3(x) __builtin_astype(x, char3)
#define __clc_as_uchar3(x) __builtin_astype(x, uchar3)
#define __clc_as_short3(x) __builtin_astype(x, short3)
#define __clc_as_ushort3(x) __builtin_astype(x, ushort3)
#define __clc_as_int3(x) __builtin_astype(x, int3)
#define __clc_as_uint3(x) __builtin_astype(x, uint3)
#define __clc_as_long3(x) __builtin_astype(x, long3)
#define __clc_as_ulong3(x) __builtin_astype(x, ulong3)
#define __clc_as_float3(x) __builtin_astype(x, float3)

#define __clc_as_char4(x) __builtin_astype(x, char4)
#define __clc_as_uchar4(x) __builtin_astype(x, uchar4)
#define __clc_as_short4(x) __builtin_astype(x, short4)
#define __clc_as_ushort4(x) __builtin_astype(x, ushort4)
#define __clc_as_int4(x) __builtin_astype(x, int4)
#define __clc_as_uint4(x) __builtin_astype(x, uint4)
#define __clc_as_long4(x) __builtin_astype(x, long4)
#define __clc_as_ulong4(x) __builtin_astype(x, ulong4)
#define __clc_as_float4(x) __builtin_astype(x, float4)

#define __clc_as_char8(x) __builtin_astype(x, char8)
#define __clc_as_uchar8(x) __builtin_astype(x, uchar8)
#define __clc_as_short8(x) __builtin_astype(x, short8)
#define __clc_as_ushort8(x) __builtin_astype(x, ushort8)
#define __clc_as_int8(x) __builtin_astype(x, int8)
#define __clc_as_uint8(x) __builtin_astype(x, uint8)
#define __clc_as_long8(x) __builtin_astype(x, long8)
#define __clc_as_ulong8(x) __builtin_astype(x, ulong8)
#define __clc_as_float8(x) __builtin_astype(x, float8)

#define __clc_as_char16(x) __builtin_astype(x, char16)
#define __clc_as_uchar16(x) __builtin_astype(x, uchar16)
#define __clc_as_short16(x) __builtin_astype(x, short16)
#define __clc_as_ushort16(x) __builtin_astype(x, ushort16)
#define __clc_as_int16(x) __builtin_astype(x, int16)
#define __clc_as_uint16(x) __builtin_astype(x, uint16)
#define __clc_as_long16(x) __builtin_astype(x, long16)
#define __clc_as_ulong16(x) __builtin_astype(x, ulong16)
#define __clc_as_float16(x) __builtin_astype(x, float16)

#ifdef cl_khr_fp64
#define __clc_as_double(x) __builtin_astype(x, double)
#define __clc_as_double2(x) __builtin_astype(x, double2)
#define __clc_as_double3(x) __builtin_astype(x, double3)
#define __clc_as_double4(x) __builtin_astype(x, double4)
#define __clc_as_double8(x) __builtin_astype(x, double8)
#define __clc_as_double16(x) __builtin_astype(x, double16)
#endif

#ifdef cl_khr_fp16
#define __clc_as_half(x) __builtin_astype(x, half)
#define __clc_as_half2(x) __builtin_astype(x, half2)
#define __clc_as_half3(x) __builtin_astype(x, half3)
#define __clc_as_half4(x) __builtin_astype(x, half4)
#define __clc_as_half8(x) __builtin_astype(x, half8)
#define __clc_as_half16(x) __builtin_astype(x, half16)
#endif

#endif // __CLC_CLC_AS_TYPE_H__
