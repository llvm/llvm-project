/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(len4)(double x, double y, double z, double w)
{
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double c = BUILTIN_ABS_F64(z);
    double d = BUILTIN_ABS_F64(w);

    double a1 = BUILTIN_MAX_F64(a, b);
    double b1 = BUILTIN_MIN_F64(a, b);

    double c1 = BUILTIN_MAX_F64(c, d);
    double d1 = BUILTIN_MIN_F64(c, d);

    a         = BUILTIN_MAX_F64(a1, c1);
    double c2 = BUILTIN_MIN_F64(a1, c1);

    double b2 = BUILTIN_MAX_F64(b1, d1);
    d         = BUILTIN_MIN_F64(b1, d1);

    b         = BUILTIN_MAX_F64(b2, c2);
    c         = BUILTIN_MIN_F64(b2, c2);

    int e = BUILTIN_FREXP_EXP_F64(a);
    a = BUILTIN_FLDEXP_F64(a, -e);
    b = BUILTIN_FLDEXP_F64(b, -e);
    c = BUILTIN_FLDEXP_F64(c, -e);
    d = BUILTIN_FLDEXP_F64(d, -e);

    double ret = BUILTIN_FLDEXP_F64(MATH_FAST_SQRT(MATH_MAD(a, a, MATH_MAD(b, b, MATH_MAD(c, c, d*d)))), e);
    ret = a == 0.0 ? 0.0 : ret;

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISNAN_F64(x) | BUILTIN_ISNAN_F64(y) |
               BUILTIN_ISNAN_F64(z) | BUILTIN_ISNAN_F64(w)) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = (BUILTIN_ISINF_F64(x) |
               BUILTIN_ISINF_F64(y) |
               BUILTIN_ISINF_F64(z) |
               BUILTIN_ISINF_F64(w)) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
    }

    return ret;
}

