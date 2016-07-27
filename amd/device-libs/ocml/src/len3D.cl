/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(len3)(double x, double y, double z)
{
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double c = BUILTIN_ABS_F64(z);

    double a1 = BUILTIN_MAX_F64(a, b);
    double b1 = BUILTIN_MIN_F64(a, b);

    a         = BUILTIN_MAX_F64(a1, c);
    double c1 = BUILTIN_MIN_F64(a1, c);

    b         = BUILTIN_MAX_F64(b1, c1);
    c         = BUILTIN_MIN_F64(b1, c1);

    int e;
    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F64(a) - 1;
        e = BUILTIN_CLAMP_S32(e, -1022, 1022);
        a = BUILTIN_FLDEXP_F64(a, -e);
        b = BUILTIN_FLDEXP_F64(b, -e);
        c = BUILTIN_FLDEXP_F64(c, -e);
    } else {
        e = (int)(AS_INT2(a).hi >> 20) - EXPBIAS_DP64;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -1022), 1022);
        double sc = AS_DOUBLE((ulong)(EXPBIAS_DP64 - e) << EXPSHIFTBITS_DP64);
        a *= sc;
        b *= sc;
        c *= sc;
    }

    double ret = MATH_FAST_SQRT(MATH_MAD(a, a, MATH_MAD(b, b, c*c)));
    ret = a == 0.0 ? 0.0 : ret;

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F64(ret, e);
    } else {
        double sc = AS_DOUBLE((ulong)(EXPBIAS_DP64 + e) << EXPSHIFTBITS_DP64);
        ret *= sc;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) |
               BUILTIN_CLASS_F64(y, CLASS_QNAN|CLASS_SNAN) |
               BUILTIN_CLASS_F64(z, CLASS_QNAN|CLASS_SNAN)) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = (BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F64(y, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F64(z, CLASS_PINF|CLASS_NINF)) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
    }

    return ret;
}

