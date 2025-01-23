#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_nextafter.h>

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(float, nextafter, __clc_nextafter,
                                        float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(double, nextafter, __clc_nextafter,
                                        double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(half, nextafter, __clc_nextafter, half,
                                        half)

#endif
