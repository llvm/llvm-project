#include <clc/clc.h>
#include "relational.h"

#define _CLC_DEFINE_ISNOTEQUAL(RET_TYPE, FUNCTION, ARG1_TYPE, ARG2_TYPE) \
_CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) { \
  return (x != y); \
} \

_CLC_DEFINE_ISNOTEQUAL(int, isnotequal, float, float)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(int, isnotequal, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isnotequal(double, double) returns an int, but the vector versions
// return long.

_CLC_DEFINE_ISNOTEQUAL(int, isnotequal, double, double)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, isnotequal, double, double)

#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isnotequal(half, half) returns an int, but the vector versions
// return short.

_CLC_DEFINE_ISNOTEQUAL(int, isnotequal, half, half)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, isnotequal, half, half)

#endif

#undef _CLC_DEFINE_ISNOTEQUAL
