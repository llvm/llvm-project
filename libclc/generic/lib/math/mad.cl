#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_mad.h>

_CLC_DEFINE_TERNARY_BUILTIN(float, mad, __clc_mad, float, float, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_TERNARY_BUILTIN(double, mad, __clc_mad, double, double, double)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_TERNARY_BUILTIN(half, mad, __clc_mad, half, half, half)

#endif
