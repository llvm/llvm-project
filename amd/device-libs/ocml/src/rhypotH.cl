/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(rhypot)

CONSTATTR half
MATH_MANGLE(rhypot)(half x, half y)
{
    float fx = (float)x;
    float fy = (float)y;

    float d2;
    if (HAVE_FAST_FMA32()) {
        d2 = BUILTIN_FMA_F32(fx, fx, fy*fy);
    } else {
        d2 = fx*fx + fy*fy;
    }

    half ret = (half)BUILTIN_RSQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F16(x) |
              BUILTIN_ISINF_F16(y) ?
              0.0h : ret;
    }

    return ret;
}

