/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(hypot)

CONSTATTR half
MATH_MANGLE(hypot)(half x, half y)
{
    float fx = (float)x;
    float fy = (float)y;
    float d2 = BUILTIN_MAD_F32(fx, fx, fy*fy);

    half ret = (half)BUILTIN_AMDGPU_SQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISINF_F16(x) | BUILTIN_ISINF_F16(y)) ? PINF_F16 : ret;
    }

    return ret;
}

