/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

half2
MATH_MANGLE2(frexp)(half2 x, __private int2 *ep)
{
    int elo, ehi;
    half2 r;
    r.lo = MATH_MANGLE(frexp)(x.lo, &elo);
    r.hi = MATH_MANGLE(frexp)(x.hi, &ehi);
    *ep = (int2)(elo, ehi);
    return r;
}

half
MATH_MANGLE(frexp)(half x, __private int *ep)
{
    return BUILTIN_FREXP_F16(x, ep);
}

