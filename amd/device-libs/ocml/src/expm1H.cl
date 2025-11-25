/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(expm1)

CONSTATTR half
MATH_MANGLE(expm1)(half x)
{
    half ret;
    ret = (half)(BUILTIN_AMDGPU_EXP2_F32((float)x * 0x1.715476p+0f) - 1.0f);
    half p = BUILTIN_FMA_F16(x, x*BUILTIN_FMA_F16(x, 0x1.555556p-3h, 0.5h), x);
    ret = BUILTIN_ABS_F16(x) < 0x1.0p-6h ? p : ret;
    return ret;
}

