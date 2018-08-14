#include <clc/clc.h>
#include "relational.h"

//Note: It would be nice to use __builtin_isless with vector inputs, but it seems to only take scalar values as
//      input, which will produce incorrect output for vector input types.

_CLC_DEFINE_RELATIONAL_BINARY(int, isless, __builtin_isless, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isless(double, double) returns an int, but the vector versions
// return long.

_CLC_DEF _CLC_OVERLOAD int isless(double x, double y){
	return __builtin_isless(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, isless, double, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isless(half, half) returns an int, but the vector versions
// return short.

_CLC_DEF _CLC_OVERLOAD int isless(half x, half y){
	return __builtin_isless(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, isless, half, half)

#endif
