/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(logb)(float x)
{
    float ret = (float)(BUILTIN_FREXP_EXP_F32(x) - 1);

    if (!FINITE_ONLY_OPT()) {
        float ax = BUILTIN_ABS_F32(x);
        ret = BUILTIN_ISFINITE_F32(ax) ? ret : ax;
        ret = x == 0.0f ? NINF_F32 : ret;
    }

    return ret;
}

