/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(hypot)

CONSTATTR INLINEATTR half
MATH_MANGLE(hypot)(half x, half y)
{
    float fx = (float)x;
    float fy = (float)y;

    float d2;
    if (HAVE_FAST_FMA32()) {
        d2 = BUILTIN_FMA_F32(fx, fx, fy*fy);
    } else {
        d2 = fx*fx + fy*fy;
    }

    half ret = (half)BUILTIN_SQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) |
              BUILTIN_CLASS_F16(y, CLASS_PINF|CLASS_NINF) ?
              AS_HALF((ushort)PINFBITPATT_HP16) : ret;
    }

    return ret;
}

