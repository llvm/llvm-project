/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR int
MATH_MANGLE(ilogb)(float x)
{
    int r = BUILTIN_FREXP_EXP_F32(x) - 1;

    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_ISNAN_F32(x) ? FP_ILOGBNAN : r;
        r = BUILTIN_ISINF_F32(x) ? INT_MAX : r;
    }

    r = x == 0.0f ? FP_ILOGB0 : r;

    return r;
}

