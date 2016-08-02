#include <clc/clc.h>
#include "../clcmacro.h"

// This file provides OpenCL C implementations of nextafter for targets that
// don't support the clang builtin.

#define FLT_NAN 0.0f/0.0f

#define NEXTAFTER(FLOAT_TYPE, UINT_TYPE, NAN, ZERO, NEXTAFTER_ZERO) \
_CLC_OVERLOAD _CLC_DEF FLOAT_TYPE __clc_nextafter(FLOAT_TYPE x, FLOAT_TYPE y) { \
  union {                     \
    FLOAT_TYPE f;             \
    UINT_TYPE i;              \
  } next;                     \
  if (isnan(x) || isnan(y)) { \
    return NAN;               \
  }                           \
  if (x == y) {               \
    return y;                 \
  }                           \
  next.f = x;                 \
  if (x < y) {                \
    next.i++;                 \
  } else {                    \
    if (next.f == ZERO) {     \
    next.i = NEXTAFTER_ZERO;  \
    } else {                  \
      next.i--;               \
    }                         \
  }                           \
  return next.f;              \
}

NEXTAFTER(float, uint, FLT_NAN, 0.0f, 0x80000001)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __clc_nextafter, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DBL_NAN 0.0/0.0

NEXTAFTER(double, ulong, DBL_NAN, 0.0, 0x8000000000000001)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __clc_nextafter, double, double)
#endif
