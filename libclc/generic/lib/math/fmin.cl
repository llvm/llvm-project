#include <clc/clc.h>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define FUNCTION __clc_fmin
#define FUNCTION_IMPL(x, y) ((y) < (x) ? (y) : (x))

#define __CLC_BODY <binary_impl.inc>
#include <clc/math/gentype.inc>
