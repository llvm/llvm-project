#include <clc/clc.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, isnan, __builtin_isnan, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isnan(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int isnan(double x) {
  return __builtin_isnan(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, isnan, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isnan(half) returns an int, but the vector versions
// return short.
_CLC_DEF _CLC_OVERLOAD int isnan(half x) {
  return __builtin_isnan(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, isnan, half)

#endif
