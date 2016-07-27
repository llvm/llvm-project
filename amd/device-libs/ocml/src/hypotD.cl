/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(hypot)(double x, double y)
{
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double t = BUILTIN_MAX_F64(a, b);

    int e;
    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F64(t) - 1;
        e = BUILTIN_CLAMP_S32(e, -1022, 1022);
        a = BUILTIN_FLDEXP_F64(a, -e);
        b = BUILTIN_FLDEXP_F64(b, -e);
    } else {
        e = (int)(AS_INT2(t).hi >> 20) - EXPBIAS_DP64;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -1022), 1022);
        double sc = AS_DOUBLE((ulong)(EXPBIAS_DP64 - e) << EXPSHIFTBITS_DP64);
        a *= sc;
        b *= sc;
    }

    double ret = MATH_FAST_SQRT(MATH_MAD(a, a, b*b));
    ret = t == 0.0 ? 0.0 : ret;

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F64(ret, e);
    } else {
        ret *= AS_DOUBLE((ulong)(EXPBIAS_DP64 + e) << EXPSHIFTBITS_DP64);
    }

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) |
              BUILTIN_CLASS_F64(y, CLASS_QNAN|CLASS_SNAN) ?  AS_DOUBLE(QNANBITPATT_DP64) : ret;

        ret = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF) |
              BUILTIN_CLASS_F64(y, CLASS_NINF|CLASS_PINF) ?  AS_DOUBLE(PINFBITPATT_DP64) : ret;
    }

    return ret;
}

