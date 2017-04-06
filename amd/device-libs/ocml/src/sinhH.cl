/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(sinh)

PUREATTR INLINEATTR half
MATH_MANGLE(sinh)(half hx)
{
    float x = (float)hx * 0x1.715476p+0f;
    return (half)(0.5f * (BUILTIN_EXP2_F32(x) - BUILTIN_EXP2_F32(-x)));
}

