/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

float
MATH_MANGLE(frexp)(float x, __private int *ep)
{
    int e = BUILTIN_FREXP_EXP_F32(x);
    float r = BUILTIN_FREXP_MANT_F32(x);

    if (HAVE_BUGGY_FREXP_INSTRUCTIONS()) {
        bool isfinite = BUILTIN_ISFINITE_F32(x);
        *ep = isfinite ? e : 0;
        return isfinite ? r : x;
    }

    *ep = e;
    return r;
}

