#include <clc/clcmacro.h>
#include <clc/utils.h>

#ifndef __CLC_BUILTIN
#define __CLC_BUILTIN __CLC_XCONCAT(__clc_, __CLC_FUNCTION)
#endif

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(float, __CLC_FUNCTION, __CLC_BUILTIN,
                                        float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(double, __CLC_FUNCTION, __CLC_BUILTIN,
                                        double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(half, __CLC_FUNCTION, __CLC_BUILTIN,
                                        half, half)

#endif
