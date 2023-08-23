/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(hypot)(double x, double y)
{
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double t = BUILTIN_MAX_F64(a, b);
    int e = BUILTIN_FREXP_EXP_F64(t);
    a = BUILTIN_FLDEXP_F64(a, -e);
    b = BUILTIN_FLDEXP_F64(b, -e);
    double ret = BUILTIN_FLDEXP_F64(MATH_FAST_SQRT(MATH_MAD(a, a, b*b)), e);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISUNORDERED_F64(x, y) ? QNAN_F64 : ret;

        ret = (BUILTIN_ISINF_F64(x) | BUILTIN_ISINF_F64(y)) ? PINF_F64 : ret;
    }

    return ret;
}

