/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(exp)

CONSTATTR half
MATH_MANGLE(exp)(half x)
{
    return (half)BUILTIN_AMDGPU_EXP2_F32((float)x * 0x1.715476p+0f);
}

