#include "math.h"
#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float logb(float x) {
    int ax = as_int(x) & EXSIGNBIT_SP32;
    float s = -118 - clz(ax);
    float r = (ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    r = ax >= PINFBITPATT_SP32 ? as_float(ax) : r;
    r = ax < 0x00800000 ? s : r;
    r = ax == 0 ? as_float(NINFBITPATT_SP32) : r;
    return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, logb, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double logb(double x) {
    long ax = as_long(x) & EXSIGNBIT_DP64;
    double s = -1011L - clz(ax);
    double r = (int) (ax >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;
    r = ax >= PINFBITPATT_DP64 ? as_double(ax) : r;
    r = ax < 0x0010000000000000L ? s : r;
    r = ax == 0L ? as_double(NINFBITPATT_DP64) : r;
    return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, logb, double)
#endif

_CLC_DEFINE_UNARY_BUILTIN_FP16(logb)
