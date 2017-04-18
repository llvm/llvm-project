/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(rlen3)(half x, half y, half z)
{
    float fx = (float)x;
    float fy = (float)y;
    float fz = (float)z;

    float d2;
    if (HAVE_FAST_FMA32()) {
        d2 = BUILTIN_FMA_F32(fx, fx, BUILTIN_FMA_F32(fy, fy, fz*fz));
    } else {
        d2 = BUILTIN_MAD_F32(fx, fx, BUILTIN_MAD_F32(fy, fy, fz*fz));
    }

    half ret = (half)BUILTIN_RSQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F16(y, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F16(z, CLASS_PINF|CLASS_NINF)) ? 0.0h : ret;
    }

    return ret;
}

