#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/relational/clc_isnan.h>

// This file provides OpenCL C implementations of __clc_nextafter for
// targets that don't support the clang builtin.

#define CLC_AS_TYPE(x) __clc_as_##x

#define NEXTAFTER(FLOAT_TYPE, UINT_TYPE, INT_TYPE, INT_TYPE_SCALAR)            \
  _CLC_OVERLOAD _CLC_DEF FLOAT_TYPE __clc_nextafter(FLOAT_TYPE x,              \
                                                    FLOAT_TYPE y) {            \
    const UINT_TYPE sign_bit = (UINT_TYPE)1                                    \
                               << (sizeof(INT_TYPE_SCALAR) * 8 - 1);           \
    const UINT_TYPE sign_bit_mask = sign_bit - (UINT_TYPE)1;                   \
    INT_TYPE ix = CLC_AS_TYPE(INT_TYPE)(x);                                    \
    UINT_TYPE ax = CLC_AS_TYPE(UINT_TYPE)(ix) & sign_bit_mask;                 \
    INT_TYPE mx = CLC_AS_TYPE(INT_TYPE)(sign_bit) - ix;                        \
    mx = CLC_AS_TYPE(INT_TYPE)(ix) < (INT_TYPE)0 ? mx : ix;                    \
    INT_TYPE iy = CLC_AS_TYPE(INT_TYPE)(y);                                    \
    UINT_TYPE ay = CLC_AS_TYPE(UINT_TYPE)(iy) & sign_bit_mask;                 \
    INT_TYPE my = CLC_AS_TYPE(INT_TYPE)(sign_bit) - iy;                        \
    my = iy < (INT_TYPE)0 ? my : iy;                                           \
    INT_TYPE t = mx + (mx < my ? (INT_TYPE)1 : (INT_TYPE)-1);                  \
    INT_TYPE r = CLC_AS_TYPE(INT_TYPE)(sign_bit) - t;                          \
    r = t < (INT_TYPE)0 ? r : t;                                               \
    r = __clc_isnan(x) ? ix : r;                                               \
    r = __clc_isnan(y) ? CLC_AS_TYPE(INT_TYPE)(iy) : r;                        \
    r = ((ax | ay) == (UINT_TYPE)0 || ix == iy) ? iy : r;                      \
    return CLC_AS_TYPE(FLOAT_TYPE)(r);                                         \
  }

NEXTAFTER(float, uint, int, int)
NEXTAFTER(float2, uint2, int2, int)
NEXTAFTER(float3, uint3, int3, int)
NEXTAFTER(float4, uint4, int4, int)
NEXTAFTER(float8, uint8, int8, int)
NEXTAFTER(float16, uint16, int16, int)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

NEXTAFTER(double, ulong, long, long)
NEXTAFTER(double2, ulong2, long2, long)
NEXTAFTER(double3, ulong3, long3, long)
NEXTAFTER(double4, ulong4, long4, long)
NEXTAFTER(double8, ulong8, long8, long)
NEXTAFTER(double16, ulong16, long16, long)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

NEXTAFTER(half, ushort, short, short)
NEXTAFTER(half2, ushort2, short2, short)
NEXTAFTER(half3, ushort3, short3, short)
NEXTAFTER(half4, ushort4, short4, short)
NEXTAFTER(half8, ushort8, short8, short)
NEXTAFTER(half16, ushort16, short16, short)

#endif
