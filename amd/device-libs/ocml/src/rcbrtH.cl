/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(rcbrt)

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(rcbrt)(half x)
{
    half ret = (half)BUILTIN_AMDGPU_EXP2_F32(-0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32((float)BUILTIN_ABS_F16(x)));

    half xi = MATH_FAST_RCP(x);

    // Is normal or subnormal
    ret = ((x != 0.0h) & BUILTIN_ISFINITE_F16(x)) ? ret : xi;

    return BUILTIN_COPYSIGN_F16(ret, x);
}

