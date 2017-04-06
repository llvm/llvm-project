/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(ilogb)(double x)
{
    int r = BUILTIN_FREXP_EXP_F64(x) - 1;

    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? FP_ILOGBNAN : r;
        r = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF) ? INT_MAX : r;
    }

    r = x == 0.0 ? FP_ILOGB0 : r;
    return r;
}

