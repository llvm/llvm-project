#include <clc/clc.h>
#include "../lib/clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, nextafter, __clc_nextafter, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEFINE_BINARY_BUILTIN(double, nextafter, __clc_nextafter, double, double)
#endif
