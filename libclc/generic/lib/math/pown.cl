#include <clc/clc.h>
#include "../clcmacro.h"

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, pown, float, int)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, pown, double, int)
#endif
