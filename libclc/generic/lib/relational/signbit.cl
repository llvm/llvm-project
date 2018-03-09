#include <clc/clc.h>
#include "relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, signbit, __builtin_signbitf, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of signbit(double) returns an int, but the vector versions
// return long.

_CLC_DEF _CLC_OVERLOAD int signbit(double x){
	return __builtin_signbit(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, signbit, double)

#endif
