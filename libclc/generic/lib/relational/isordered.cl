#include <clc/clc.h>
#include "relational.h"

#define _CLC_DEFINE_ISORDERED(RET_TYPE, FUNCTION, ARG1_TYPE, ARG2_TYPE) \
_CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) { \
  return isequal(x, x) && isequal(y, y); \
} \

_CLC_DEFINE_ISORDERED(int, isordered, float, float)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(int, isordered, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isordered(double, double) returns an int, but the vector versions
// return long.

_CLC_DEFINE_ISORDERED(int, isordered, double, double)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, isordered, double, double)

#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isordered(half, half) returns an int, but the vector versions
// return short.

_CLC_DEFINE_ISORDERED(int, isordered, half, half)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, isordered, half, half)

#endif

#undef _CLC_DEFINE_ISORDERED
