/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(logb)(double x)
{
    double ret = (double)(BUILTIN_FREXP_EXP_F64(x) - 1);

    if (!FINITE_ONLY_OPT()) {
        double ax = BUILTIN_ABS_F64(x);
        ret = BUILTIN_ISFINITE_F64(ax) ? ret : ax;
        ret = x == 0.0 ? NINF_F64 : ret;
    }

    return ret;
}

