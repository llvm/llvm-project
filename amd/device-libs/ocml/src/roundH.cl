/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half2
MATH_MANGLE2(round)(half2 x)
{
    half2 t = BUILTIN_TRUNC_2F16(x);
    half2 d = BUILTIN_ABS_2F16(x - t);
    half2 o = BUILTIN_COPYSIGN_2F16(1.0h, x);
    o.lo = d.lo >= 0.5h ? o.lo : 0.0h;
    o.hi = d.hi >= 0.5h ? o.hi : 0.0h;
    return t + o;
}

CONSTATTR INLINEATTR half
MATH_MANGLE(round)(half x)
{
    half t = BUILTIN_TRUNC_F16(x);
    half d = BUILTIN_ABS_F16(x - t);
    half o = BUILTIN_COPYSIGN_F16(1.0h, x);
    return t + (d >= 0.5h ? o : 0.0h);
}

