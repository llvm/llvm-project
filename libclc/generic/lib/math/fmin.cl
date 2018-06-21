#include <clc/clc.h>

#include "../clcmacro.h"

_CLC_DEFINE_BINARY_BUILTIN(float, fmin, __builtin_fminf, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmin, __builtin_fmin, double, double);

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half fmin(half x, half y)
{
   if (isnan(x))
      return y;
   if (isnan(y))
      return x;
   return (y < x) ? y : x;
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, fmin, half, half)

#endif

#define __CLC_BODY <fmin.inc>
#include <clc/math/gentype.inc>
