#include <clc/internal/clc.h>
#include <clc/relational/relational.h>

#define _CLC_RELATIONAL_OP(X, Y) (X) <= (Y)

_CLC_DEFINE_SIMPLE_RELATIONAL_BINARY(int, int, __clc_islessequal, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_islessequal(double, double) returns an int, but
// the vector versions return long.
_CLC_DEFINE_SIMPLE_RELATIONAL_BINARY(int, long, __clc_islessequal, double,
                                     double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_islessequal(half, half) returns an int, but the
// vector versions return short.
_CLC_DEFINE_SIMPLE_RELATIONAL_BINARY(int, short, __clc_islessequal, half, half)

#endif

#undef _CLC_RELATIONAL_OP
