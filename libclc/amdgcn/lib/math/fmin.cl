#include <clc/clc.h>

#include "../../../generic/lib/clcmacro.h"

_CLC_DEF _CLC_OVERLOAD float fmin(float x, float y)
{
   /* fcanonicalize removes sNaNs and flushes denormals if not enabled.
    * Otherwise fmin instruction flushes the values for comparison,
    * but outputs original denormal */
   x = __builtin_canonicalizef(x);
   y = __builtin_canonicalizef(y);
   return __builtin_fminf(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, fmin, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double fmin(double x, double y)
{
   x = __builtin_canonicalize(x);
   y = __builtin_canonicalize(y);
   return __builtin_fmin(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, fmin, double, double)

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

#define __CLC_BODY <../../../generic/lib/math/fmin.inc>
#include <clc/math/gentype.inc>
