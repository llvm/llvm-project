#include <clc/internal/clc.h>
#include <clc/relational/relational.h>

// Note: It would be nice to use __builtin_islessequal with vector inputs, but
// it seems to only take scalar values as input, which will produce incorrect
// output for vector input types.

_CLC_DEFINE_RELATIONAL_BINARY(int, __clc_islessequal, __builtin_islessequal,
                              float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_islessequal(double, double) returns an int, but
// the vector versions return long.

_CLC_DEF _CLC_OVERLOAD int __clc_islessequal(double x, double y) {
  return __builtin_islessequal(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, __clc_islessequal, double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_islessequal(half, half) returns an int, but the
// vector versions return short.

_CLC_DEF _CLC_OVERLOAD int __clc_islessequal(half x, half y) {
  return __builtin_islessequal(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, __clc_islessequal, half, half)

#endif
