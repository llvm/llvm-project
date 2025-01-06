#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float rsqrt(float x)
{
    return 1.0f / sqrt(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, rsqrt, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double rsqrt(double x)
{
    return 1.0 / sqrt(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, rsqrt, double);

#endif
