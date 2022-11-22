/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

double
MATH_MANGLE(frexp)(double x, __private int *ep)
{
    int e = BUILTIN_FREXP_EXP_F64(x);
    double r = BUILTIN_FREXP_MANT_F64(x);

    if (HAVE_BUGGY_FREXP_INSTRUCTIONS()) {
        bool isfinite = BUILTIN_ISFINITE_F64(x);
        *ep = isfinite ? e : 0;
        return isfinite ? r : x;
    }

    *ep = e;
    return r;
}

