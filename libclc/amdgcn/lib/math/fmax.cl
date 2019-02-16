#include <clc/clc.h>

#include "../../../generic/lib/clcmacro.h"

_CLC_DEF _CLC_OVERLOAD float fmax(float x, float y)
{
   /* fcanonicalize removes sNaNs and flushes denormals if not enabled.
    * Otherwise fmax instruction flushes the values for comparison,
    * but outputs original denormal */
   x = __builtin_canonicalizef(x);
   y = __builtin_canonicalizef(y);
   return __builtin_fmaxf(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, fmax, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double fmax(double x, double y)
{
   x = __builtin_canonicalize(x);
   y = __builtin_canonicalize(y);
   return __builtin_fmax(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, fmax, double, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half fmax(half x, half y)
{
   if (isnan(x))
      return y;
   if (isnan(y))
      return x;
   return (y < x) ? x : y;
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, fmax, half, half)

#endif

#define __CLC_BODY <../../../generic/lib/math/fmax.inc>
#include <clc/math/gentype.inc>
