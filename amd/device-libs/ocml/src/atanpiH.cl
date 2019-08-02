
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

extern CONSTATTR half MATH_PRIVATE(atanpired)(half);

CONSTATTR UGEN(atanpi)

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(atanpi)(half x)
{
    half v = BUILTIN_ABS_F16(x);
    bool g = v > 1.0h;

    half vi = MATH_FAST_RCP(v);
    v = g ? vi : v;

    half a = MATH_PRIVATE(atanpired)(v);

    half y = 0.5h - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F16(a, x);
}


