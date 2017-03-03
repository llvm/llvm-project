/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

INLINEATTR short
MATH_PRIVATE(trigpired)(half x, __private half *r)
{
    half t = 2.0h * BUILTIN_FRACTION_F16(0.5h * x);
    x = x > 1.0h ? t : x;
    t = BUILTIN_RINT_F16(2.0h * x);
    *r = MATH_MAD(t, -0.5h, x);
    return (short)t & (short)0x3;
}

