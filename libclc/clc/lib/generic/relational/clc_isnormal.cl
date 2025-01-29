#include <clc/internal/clc.h>
#include <clc/relational/relational.h>

_CLC_DEFINE_ISFPCLASS(int, int, __clc_isnormal, fcNormal, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_isnormal(double) returns an int, but the vector
// versions return long.
_CLC_DEFINE_ISFPCLASS(int, long, __clc_isnormal, fcNormal, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_isnormal(half) returns an int, but the vector
// versions return short.
_CLC_DEFINE_ISFPCLASS(int, short, __clc_isnormal, fcNormal, half)

#endif
