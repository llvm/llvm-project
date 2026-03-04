/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(acosh)

CONSTATTR half
MATH_MANGLE(acosh)(half hx)
{
    half ret;
    float x = (float)hx;
    float t = x + BUILTIN_AMDGPU_SQRT_F32(BUILTIN_MAD_F32(x, x, -1.0f));
    ret =  (half)(BUILTIN_AMDGPU_LOG2_F32(t) * 0x1.62e430p-1f);

    if (!FINITE_ONLY_OPT()) {
        ret = hx < 1.0h ? QNAN_F16 : ret;
    }

    return ret;
}

