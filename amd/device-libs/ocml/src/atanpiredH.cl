/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_PRIVATE(atanpired)(half v)
{
    const half ch = 0x1.45cp-2h;
    const half cl = 0x1.85cp-13h;
    half t = v * v;
    half y = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 0x1.f04p-8h, -0x1.dfp-6h), 0x1.e3p-5h), -0x1.b08p-4h);
    half ph = v * ch;
    half pl = MATH_MAD(v, ch, -ph);
    half r = MATH_MAD(v, MATH_MAD(t, y, cl), pl) + ph;
    return r;
}

