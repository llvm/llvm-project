/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(round)(half x)
{
    half t = BUILTIN_TRUNC_F16(x);
    half d = BUILTIN_ABS_F16(x - t);
    half o = BUILTIN_COPYSIGN_F16(1.0h, x);
    return t + (d >= 0.5h ? o : 0.0h);
}

