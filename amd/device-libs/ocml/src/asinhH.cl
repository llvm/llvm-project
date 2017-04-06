/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(asinh)

PUREATTR INLINEATTR half
MATH_MANGLE(asinh)(half hx)
{
    half ret;
    float x = (float)BUILTIN_ABS_F16(hx);
    float t = x + BUILTIN_SQRT_F32(BUILTIN_MAD_F32(x, x, 1.0f));
    ret = BUILTIN_COPYSIGN_F16((half)(BUILTIN_LOG2_F32(t) * 0x1.62e430p-1f), hx);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F16(hx, CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? hx : ret;
    }

    return ret;
}

