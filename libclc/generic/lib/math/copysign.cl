#include <clc/clc.h>
#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, copysign, __builtin_copysignf, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, copysign, __builtin_copysign, double, double)

#endif
