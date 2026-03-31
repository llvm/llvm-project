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
    half ax = BUILTIN_ABS_F16(hx);
    float x = (float)ax;
    float t = (1.0f + x) * BUILTIN_AMDGPU_RCP_F32(1.0f - x);
    half ret = (half)(BUILTIN_AMDGPU_LOG2_F32(t) * 0x1.62e430p-2f);
    ret = ax < 0x1.0p-7h ? ax : ret;

    if (!FINITE_ONLY_OPT()) {
        ret = ax == 1.0h ? PINF_F16 : ret;
        ret = (ax > 1.0h) | BUILTIN_ISNAN_F16(hx) ? QNAN_F16 : ret;
    }

    return BUILTIN_COPYSIGN_F16(ret, hx);
}
