
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

extern CONSTATTR half MATH_PRIVATE(atanred)(half);

CONSTATTR UGEN(atanpi)

CONSTATTR INLINEATTR half
MATH_MANGLE(atanpi)(half x)
{
    const half pi = 0x1.921fb6p+1h;

    half v = BUILTIN_ABS_F16(x);
    bool g = v > 1.0h;

    half vi = MATH_FAST_RCP(v);
    v = g ? vi : v;

    half a = MATH_PRIVATE(atanred)(v);
    a = MATH_FAST_DIV(a, pi);

    half y = 0.5h - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F16(a, x);
}


