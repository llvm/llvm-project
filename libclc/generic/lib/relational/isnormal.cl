#include <clc/clc.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, isnormal, __builtin_isnormal, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isnormal(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int isnormal(double x) {
  return __builtin_isnormal(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, isnormal, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isnormal(half) returns an int, but the vector versions
// return short.
_CLC_DEF _CLC_OVERLOAD int isnormal(half x) {
  return __builtin_isnormal(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, isnormal, half)

#endif
