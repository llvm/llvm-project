#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_DEF _CLC_OVERLOAD float fmin(float x, float y)
{
   /* fcanonicalize removes sNaNs and flushes denormals if not enabled.
    * Otherwise fmin instruction flushes the values for comparison,
    * but outputs original denormal */
   x = __clc_flush_denormal_if_not_supported(x);
   y = __clc_flush_denormal_if_not_supported(y);
   return __builtin_fminf(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, fmin, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double fmin(double x, double y)
{
   return __builtin_fmin(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, fmin, double, double)

#endif

#define __CLC_BODY <../../../generic/lib/math/fmin.inc>
#include <clc/math/gentype.inc>
