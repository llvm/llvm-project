/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half2
MATH_MANGLE2(fract)(half2 x, __private half2 *ip)
{
    *ip = BUILTIN_FLOOR_2F16(x);
    return (half2)(BUILTIN_FRACTION_F16(x.lo), BUILTIN_FRACTION_F16(x.hi));
}

INLINEATTR half
MATH_MANGLE(fract)(half x, __private half *ip)
{
    *ip = BUILTIN_FLOOR_F16(x);
    return  BUILTIN_FRACTION_F16(x);
}

