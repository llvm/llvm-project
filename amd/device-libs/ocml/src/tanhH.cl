/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(tanh)

CONSTATTR half
MATH_MANGLE(tanh)(half hx)
{
    float x = (float)hx * 0x1.715476p+0f;
    float a = BUILTIN_EXP2_F32(x);
    float b = BUILTIN_EXP2_F32(-x);
    half one = BUILTIN_COPYSIGN_F16(1.0h, hx);
    half ret = (half)((a - b) * BUILTIN_RCP_F32(a + b));
    return BUILTIN_ABS_F16(hx) > 4.5h ? one : ret;
}

