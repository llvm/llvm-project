/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(log1p)

PUREATTR INLINEATTR half
MATH_MANGLE(log1p)(half x)
{
    half ret;
    ret = (half)(BUILTIN_LOG2_F32((float)x + 1.0f) * 0x1.62e430p-1f);
    half p = MATH_MAD(x, x*MATH_MAD(x, 0x1.555556p-2h, -0.5h), x);
    ret = BUILTIN_ABS_F16(x) < 0x1.0p-6h ? p : ret;

    return ret;
}

