/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(len3)(half x, half y, half z)
{
    float fx = (float)x;
    float fy = (float)y;
    float fz = (float)z;

    float d2 = MATH_MAD(fx, fx, MATH_MAD(fy, fy, fz*fz));

    half ret = (half)BUILTIN_SQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISINF_F16(x) |
               BUILTIN_ISINF_F16(y) |
               BUILTIN_ISINF_F16(z)) ? AS_HALF((ushort)PINFBITPATT_HP16) : ret;
    }

    return ret;
}

