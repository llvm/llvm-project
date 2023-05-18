#include <clc/clc.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, isfinite, __builtin_isfinite, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isfinite(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int isfinite(double x) {
  return __builtin_isfinite(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, isfinite, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isfinite(half) returns an int, but the vector versions
// return short.
_CLC_DEF _CLC_OVERLOAD int isfinite(half x) {
  return __builtin_isfinite(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, isfinite, half)

#endif
