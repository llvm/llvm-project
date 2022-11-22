/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS half2
MATH_MANGLE2(frexp)(half2 x, __private int2 *ep)
{
    int elo, ehi;
    half2 r;
    r.lo = MATH_MANGLE(frexp)(x.lo, &elo);
    r.hi = MATH_MANGLE(frexp)(x.hi, &ehi);
    *ep = (int2)(elo, ehi);
    return r;
}

REQUIRES_16BIT_INSTS half
MATH_MANGLE(frexp)(half x, __private int *ep)
{
    int e = (int)BUILTIN_FREXP_EXP_F16(x);
    half r = BUILTIN_FREXP_MANT_F16(x);

    if (HAVE_BUGGY_FREXP_INSTRUCTIONS()) {
        bool isfinite = BUILTIN_ISFINITE_F16(x);
        *ep = isfinite ? e : 0;
        return isfinite ? r : x;
    }

    *ep = e;
    return r;
}

