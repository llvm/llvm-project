/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(atanh)

CONSTATTR half
MATH_MANGLE(atanh)(half hx)
{
    half ret;
    float x = (float)BUILTIN_ABS_F16(hx);
    float t = (1.0f + x) * BUILTIN_RCP_F32(1.0f - x);
    ret = (half)(BUILTIN_LOG2_F32(t) * 0x1.62e430p-2f);
    ret = x < 0x1.0p-7f ? x : ret;

    if (!FINITE_ONLY_OPT()) {
        ret = x == 1.0f ? AS_HALF((short)PINFBITPATT_HP16) : ret;
        ret = (x > 1.0f) | BUILTIN_ISNAN_F32(x) ? AS_HALF((short)QNANBITPATT_HP16) : ret;
    }

    return BUILTIN_COPYSIGN_F16(ret, hx);
}

