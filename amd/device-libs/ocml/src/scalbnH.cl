/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS CONSTATTR half2
MATH_MANGLE2(scalbn)(half2 x, int2 n)
{
    return (half2)(MATH_MANGLE(ldexp)(x.lo, n.lo), MATH_MANGLE(ldexp)(x.hi, n.hi));
}

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(scalbn)(half x, int n)
{
    return MATH_MANGLE(ldexp)(x, n);
}

