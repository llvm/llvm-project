#include <clc/clc.h>
#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, nextafter, __builtin_nextafterf, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, nextafter, __builtin_nextafter, double, double)

#endif
