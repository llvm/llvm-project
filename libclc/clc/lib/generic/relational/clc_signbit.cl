#include <clc/internal/clc.h>
#include <clc/relational/relational.h>

_CLC_DEFINE_RELATIONAL_UNARY(int, __clc_signbit, __builtin_signbitf, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_signbit(double) returns an int, but the vector
// versions return long.

_CLC_DEF _CLC_OVERLOAD int __clc_signbit(double x) {
  return __builtin_signbit(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, __clc_signbit, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_signbit(half) returns an int, but the vector
// versions return short.

_CLC_DEF _CLC_OVERLOAD int __clc_signbit(half x) {
  return __builtin_signbit(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, __clc_signbit, half)

#endif
