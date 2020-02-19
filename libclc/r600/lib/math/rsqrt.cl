#include <clc/clc.h>

#include "../../../generic/lib/clcmacro.h"

_CLC_OVERLOAD _CLC_DEF float rsqrt(float x)
{
    return __builtin_r600_recipsqrt_ieeef(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, rsqrt, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double rsqrt(double x)
{
    return __builtin_r600_recipsqrt_ieee(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, rsqrt, double);

#endif
