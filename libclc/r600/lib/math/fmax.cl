#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_DEF _CLC_OVERLOAD float fmax(float x, float y)
{
   /* Flush denormals if not enabled. Otherwise fmax instruction flushes
    * the values for comparison, but outputs original denormal */
   x = __clc_flush_denormal_if_not_supported(x);
   y = __clc_flush_denormal_if_not_supported(y);
   return __builtin_fmaxf(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, fmax, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double fmax(double x, double y)
{
   return __builtin_fmax(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, fmax, double, double)

#endif

#define __CLC_BODY <../../../generic/lib/math/fmax.inc>
#include <clc/math/gentype.inc>
