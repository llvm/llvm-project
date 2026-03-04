/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(rlen3)(double x, double y, double z)
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

    int e = BUILTIN_FREXP_EXP_F64(a);
    a = BUILTIN_FLDEXP_F64(a, -e);
    b = BUILTIN_FLDEXP_F64(b, -e);
    c = BUILTIN_FLDEXP_F64(c, -e);

    double d2 = MATH_MAD(a, a, MATH_MAD(b, b, c*c));
    double v = BUILTIN_AMDGPU_RSQRT_F64(d2);
    double u = MATH_MAD(-d2*v, v, 1.0);
    v = MATH_MAD(v*u, MATH_MAD(u, 0.375, 0.5), v);
    double ret = BUILTIN_FLDEXP_F64(v, -e);

    if (!FINITE_ONLY_OPT()) {
        ret = a == 0.0 ? PINF_F64 : ret;

        ret = (BUILTIN_ISNAN_F64(x) |
               BUILTIN_ISNAN_F64(y) |
               BUILTIN_ISNAN_F64(z)) ? QNAN_F64 : ret;

        ret = (BUILTIN_ISINF_F64(x) |
               BUILTIN_ISINF_F64(y) |
               BUILTIN_ISINF_F64(z)) ? 0.0 : ret;
    }

    return ret;
}

