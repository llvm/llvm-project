/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half
MATH_PRIVATE(sincospired)(half x, __private half *cp)
{

    half t = x * x;

    half sx = MATH_MAD(t, 0x1.b84p+0h, -0x1.46cp+2h);
    sx = x * t * sx;
    sx = MATH_MAD(x, 0x1.92p+1h, sx);

    half cx = MATH_MAD(t, 0x1.fbp+1h, -0x1.3bcp+2h);
    cx = MATH_MAD(t, cx, 1.0h);

    *cp = cx;
    return sx;
}

