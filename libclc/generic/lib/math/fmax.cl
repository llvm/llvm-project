#include <clc/clc.h>

#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, fmax, __builtin_fmaxf, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmax, __builtin_fmax, double, double);

#endif

#define __CLC_BODY <fmax.inc>
#include <clc/math/gentype.inc>
