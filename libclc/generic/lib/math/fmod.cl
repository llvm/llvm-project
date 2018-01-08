#include <clc/clc.h>
#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, fmod, __builtin_fmodf, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmod, __builtin_fmod, double, double)

#endif
