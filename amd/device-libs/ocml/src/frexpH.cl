/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half2
MATH_MANGLE2(frexp)(half2 x, __private int2 *ep)
{
    int elo, ehi;
    half2 r;
    r.lo = MATH_MANGLE(frexp)(x.lo, &elo);
    r.hi = MATH_MANGLE(frexp)(x.hi, &ehi);
    *ep = (int2)(elo, ehi);
    return r;
}

INLINEATTR half
MATH_MANGLE(frexp)(half x, __private int *ep)
{
    int e = (int)BUILTIN_FREXP_EXP_F16(x);
    half r = BUILTIN_FREXP_MANT_F16(x);
    bool c = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF|CLASS_SNAN|CLASS_QNAN);
    *ep = c ? 0 : e;
    return c ? x : r;
}

