#include <clc/clc.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, isinf, __builtin_isinf, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isinf(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int isinf(double x) {
  return __builtin_isinf(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, isinf, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isinf(half) returns an int, but the vector versions
// return short.
_CLC_DEF _CLC_OVERLOAD int isinf(half x) {
  return __builtin_isinf(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, isinf, half)
#endif
