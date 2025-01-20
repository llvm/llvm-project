#include <clc/internal/clc.h>
#include <clc/relational/relational.h>

// Note: It would be nice to use __builtin_isunordered with vector inputs, but
// it seems to only take scalar values as input, which will produce incorrect
// output for vector input types.

_CLC_DEFINE_RELATIONAL_BINARY(int, __clc_isunordered, __builtin_isunordered,
                              float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_isunordered(double, double) returns an int, but
// the vector versions return long.

_CLC_DEF _CLC_OVERLOAD int __clc_isunordered(double x, double y) {
  return __builtin_isunordered(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, __clc_isunordered, double, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_isunordered(half, half) returns an int, but the
// vector versions return short.

_CLC_DEF _CLC_OVERLOAD int __clc_isunordered(half x, half y) {
  return __builtin_isunordered(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, __clc_isunordered, half, half)

#endif
