#include <clc/clc.h>

#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, fmin, __builtin_fminf, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmin, __builtin_fmin, double, double);

#endif

#define __CLC_BODY <fmin.inc>
#include <clc/math/gentype.inc>
