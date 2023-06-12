/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(cbrt)

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(cbrt)(half x)
{
    half ret = (half)BUILTIN_AMDGPU_EXP2_F32(0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32((float)BUILTIN_ABS_F16(x)));
    ret = BUILTIN_COPYSIGN_F16(ret, x);

    // Is normal or subnormal.
    return ((x != 0.0h) & BUILTIN_ISFINITE_F16(x)) ? ret : x;
}

