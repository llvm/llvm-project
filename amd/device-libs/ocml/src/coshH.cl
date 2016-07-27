/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR INLINEATTR half
MATH_MANGLE(cosh)(half hx)
{
    half ret;
    if (AMD_OPT()) {
        float x = (float)hx * 0x1.715476p+0f;
        ret = (half)(0.5f * (BUILTIN_EXP2_F32(x) + BUILTIN_EXP2_F32(-x)));
    } else {
        ret = (half)MATH_UPMANGLE(cosh)((float)hx);
    }
    return ret;
}

