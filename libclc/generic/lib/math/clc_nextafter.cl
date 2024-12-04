#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/relational/clc_isnan.h>

// This file provides OpenCL C implementations of nextafter for
// targets that don't support the clang builtin.

#define AS_TYPE(x) as_##x

#define NEXTAFTER(FLOAT_TYPE, UINT_TYPE, INT_TYPE)                             \
  _CLC_OVERLOAD _CLC_DEF FLOAT_TYPE __clc_nextafter(FLOAT_TYPE x,              \
                                                    FLOAT_TYPE y) {            \
    const UINT_TYPE sign_bit = (UINT_TYPE)1 << (sizeof(INT_TYPE) * 8 - 1);     \
    const UINT_TYPE sign_bit_mask = sign_bit - 1;                              \
    INT_TYPE ix = AS_TYPE(INT_TYPE)(x);                                        \
    INT_TYPE ax = ix & sign_bit_mask;                                          \
    INT_TYPE mx = sign_bit - ix;                                               \
    mx = ix < 0 ? mx : ix;                                                     \
    INT_TYPE iy = AS_TYPE(INT_TYPE)(y);                                        \
    INT_TYPE ay = iy & sign_bit_mask;                                          \
    INT_TYPE my = sign_bit - iy;                                               \
    my = iy < 0 ? my : iy;                                                     \
    INT_TYPE t = mx + (mx < my ? 1 : -1);                                      \
    INT_TYPE r = sign_bit - t;                                                 \
    r = t < 0 ? r : t;                                                         \
    r = __clc_isnan(x) ? ix : r;                                               \
    r = __clc_isnan(y) ? iy : r;                                               \
    r = ((ax | ay) == 0 | ix == iy) ? iy : r;                                  \
    return AS_TYPE(FLOAT_TYPE)(r);                                             \
  }

NEXTAFTER(float, uint, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __clc_nextafter, float,
                      float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

NEXTAFTER(double, ulong, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __clc_nextafter, double,
                      double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

NEXTAFTER(half, ushort, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __clc_nextafter, half, half)
#endif
