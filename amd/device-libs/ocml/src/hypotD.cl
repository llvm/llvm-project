
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(hypot)(double x, double y)
{
    x = BUILTIN_ABS_F64(x);
    y = BUILTIN_ABS_F64(y);
    double u = BUILTIN_MAX_F64(x, y);
    double v = BUILTIN_MIN_F64(x, y);
    bool z = u == 0.0;

    int e;
    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F64(u) - 1;
        e = BUILTIN_MEDIAN3_S32(e, -1022, 1022);
        u = BUILTIN_FLDEXP_F64(u, -e);
        v = BUILTIN_FLDEXP_F64(v, -e);
    } else {
        e = (int)(as_int2(u).hi >> 20) - EXPBIAS_DP64;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -1022), 1022);
        double sc = as_double((ulong)(EXPBIAS_DP64 - e) << EXPSHIFTBITS_DP64);
        u *= sc;
        v *= sc;
    }

    double ret = MATH_FAST_SQRT(MATH_MAD(u, u, v*v));

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F64(ret, e);
    } else {
        double sc = as_double((ulong)(EXPBIAS_DP64 + e) << EXPSHIFTBITS_DP64);
        ret *= sc;
    }

    ret = z ? 0.0 : ret;

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) |
              BUILTIN_CLASS_F64(y, CLASS_QNAN|CLASS_SNAN) ?
              as_double(QNANBITPATT_DP64) : ret;

        ret = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF) |
              BUILTIN_CLASS_F64(y, CLASS_NINF|CLASS_PINF) ?
              as_double(PINFBITPATT_DP64) : ret;
    }

    return ret;
}

