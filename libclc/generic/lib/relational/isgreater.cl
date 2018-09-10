#include <clc/clc.h>
#include "relational.h"

//Note: It would be nice to use __builtin_isgreater with vector inputs, but it seems to only take scalar values as
//      input, which will produce incorrect output for vector input types.

_CLC_DEFINE_RELATIONAL_BINARY(int, isgreater, __builtin_isgreater, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isgreater(double, double) returns an int, but the vector versions
// return long.

_CLC_DEF _CLC_OVERLOAD int isgreater(double x, double y){
	return __builtin_isgreater(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(long, isgreater, double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of isgreater(half, half) returns an int, but the vector versions
// return short.

_CLC_DEF _CLC_OVERLOAD int isgreater(half x, half y){
	return __builtin_isgreater(x, y);
}

_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(short, isgreater, half, half)

#endif
