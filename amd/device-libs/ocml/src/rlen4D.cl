/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(rlen4)(double x, double y, double z, double w)
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

    double l2 = MATH_MAD(a, a, MATH_MAD(b, b, MATH_MAD(c, c, d*d)));
    double v = BUILTIN_AMDGPU_RSQRT_F64(l2);
    double u = MATH_MAD(-l2*v, v, 1.0);
    v = MATH_MAD(v*u, MATH_MAD(u, 0.375, 0.5), v);
    double ret = BUILTIN_FLDEXP_F64(v, -e);

    if (!FINITE_ONLY_OPT()) {
        ret = a == 0.0 ? PINF_F64 : ret;

        ret = (BUILTIN_ISNAN_F64(x) |
               BUILTIN_ISNAN_F64(y) |
               BUILTIN_ISNAN_F64(z) |
               BUILTIN_ISNAN_F64(w)) ? QNAN_F64 : ret;

        ret = (BUILTIN_ISINF_F64(x) |
               BUILTIN_ISINF_F64(y) |
               BUILTIN_ISINF_F64(z) |
               BUILTIN_ISINF_F64(w)) ? 0.0 : ret;
    }

    return ret;
}

