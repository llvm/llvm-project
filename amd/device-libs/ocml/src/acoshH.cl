/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR INLINEATTR half
MATH_MANGLE(acosh)(half hx)
{
    half ret;
    if (AMD_OPT()) {
        float x = (float)hx;
        float t = x + BUILTIN_SQRT_F32(BUILTIN_MAD_F32(x, x, -1.0f));
        ret =  (half)(BUILTIN_LOG2_F32(t) * 0x1.62e430p-1f);
        ret = hx < 1.0h ? AS_HALF((short)QNANBITPATT_HP16) : ret;
    } else {
        ret =  (half)MATH_UPMANGLE(acosh)((float)hx);
    }
    return ret;
}

