//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_fabs.h>
#include <clc/relational/clc_isnan.h>

// This file provides OpenCL C implementations of __clc_nextafter for
// targets that don't support the clang builtin.

#define __CLC_CLC_AS_TYPE(x) __clc_as_##x

#define __CLC_NEXTAFTER(FLOAT_TYPE, UINT_TYPE, INT_TYPE, INT_TYPE_SCALAR)      \
  _CLC_OVERLOAD _CLC_DEF FLOAT_TYPE __clc_nextafter(FLOAT_TYPE x,              \
                                                    FLOAT_TYPE y) {            \
    const UINT_TYPE sign_bit = (UINT_TYPE)1                                    \
                               << (sizeof(INT_TYPE_SCALAR) * 8 - 1);           \
    UINT_TYPE ix = __CLC_CLC_AS_TYPE(UINT_TYPE)(x);                            \
    FLOAT_TYPE absx = __clc_fabs(x);                                           \
    UINT_TYPE mxu = sign_bit - ix;                                             \
    INT_TYPE mx = __CLC_CLC_AS_TYPE(INT_TYPE)(mxu);                            \
    mx = __CLC_CLC_AS_TYPE(INT_TYPE)(ix) < (INT_TYPE)0                         \
             ? mx                                                              \
             : __CLC_CLC_AS_TYPE(INT_TYPE)(ix);                                \
    UINT_TYPE iy = __CLC_CLC_AS_TYPE(UINT_TYPE)(y);                            \
    FLOAT_TYPE absy = __clc_fabs(y);                                           \
    UINT_TYPE myu = sign_bit - iy;                                             \
    INT_TYPE my = __CLC_CLC_AS_TYPE(INT_TYPE)(myu);                            \
    my = __CLC_CLC_AS_TYPE(INT_TYPE)(iy) < (INT_TYPE)0                         \
             ? my                                                              \
             : __CLC_CLC_AS_TYPE(INT_TYPE)(iy);                                \
    INT_TYPE t = mx + (mx < my ? (INT_TYPE)1 : (INT_TYPE) - 1);                \
    UINT_TYPE r = sign_bit - __CLC_CLC_AS_TYPE(UINT_TYPE)(t);                  \
    r = (t < (INT_TYPE)0 || (t == (INT_TYPE)0 && mx < my))                     \
            ? r                                                                \
            : __CLC_CLC_AS_TYPE(UINT_TYPE)(t);                                 \
    r = __clc_isnan(x) ? ix : r;                                               \
    r = __clc_isnan(y) ? iy : r;                                               \
    r = ((__CLC_CLC_AS_TYPE(UINT_TYPE)(absx) |                                 \
          __CLC_CLC_AS_TYPE(UINT_TYPE)(absy)) == (UINT_TYPE)0 ||               \
         ix == iy)                                                             \
            ? iy                                                               \
            : r;                                                               \
    return __CLC_CLC_AS_TYPE(FLOAT_TYPE)(r);                                   \
  }

__CLC_NEXTAFTER(float, uint, int, int)
__CLC_NEXTAFTER(float2, uint2, int2, int)
__CLC_NEXTAFTER(float3, uint3, int3, int)
__CLC_NEXTAFTER(float4, uint4, int4, int)
__CLC_NEXTAFTER(float8, uint8, int8, int)
__CLC_NEXTAFTER(float16, uint16, int16, int)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__CLC_NEXTAFTER(double, ulong, long, long)
__CLC_NEXTAFTER(double2, ulong2, long2, long)
__CLC_NEXTAFTER(double3, ulong3, long3, long)
__CLC_NEXTAFTER(double4, ulong4, long4, long)
__CLC_NEXTAFTER(double8, ulong8, long8, long)
__CLC_NEXTAFTER(double16, ulong16, long16, long)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__CLC_NEXTAFTER(half, ushort, short, short)
__CLC_NEXTAFTER(half2, ushort2, short2, short)
__CLC_NEXTAFTER(half3, ushort3, short3, short)
__CLC_NEXTAFTER(half4, ushort4, short4, short)
__CLC_NEXTAFTER(half8, ushort8, short8, short)
__CLC_NEXTAFTER(half16, ushort16, short16, short)

#endif
