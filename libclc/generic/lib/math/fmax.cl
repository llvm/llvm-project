#include <clc/clc.h>

#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, fmax, __builtin_fmaxf, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmax, __builtin_fmax, double, double);

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half fmax(half x, half y)
{
   if (isnan(x))
      return y;
   if (isnan(y))
      return x;
   return (x < y) ? y : x;
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, fmax, half, half)

#endif

#define __CLC_BODY <fmax.inc>
#include <clc/math/gentype.inc>
