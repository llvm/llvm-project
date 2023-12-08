#include "../clcmacro.h"
#include "utils.h"

#ifndef __CLC_BUILTIN
#define __CLC_BUILTIN __CLC_XCONCAT(__clc_, __CLC_FUNCTION)
#endif

_CLC_DEFINE_UNARY_BUILTIN(float, __CLC_FUNCTION, __CLC_BUILTIN, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_UNARY_BUILTIN(double, __CLC_FUNCTION, __CLC_BUILTIN, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN(half, __CLC_FUNCTION, __CLC_BUILTIN, half)

#endif
