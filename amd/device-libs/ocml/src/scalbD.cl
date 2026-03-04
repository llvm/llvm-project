/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(scalb)(double x, double y)
{
    double t = BUILTIN_MIN_F64(BUILTIN_MAX_F64(y, -0x1.0p+20), 0x1.0p+20);
    double ret = MATH_MANGLE(ldexp)(x, (int)BUILTIN_RINT_F64(t));

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISUNORDERED_F64(x, y) ? QNAN_F64 : ret;
        ret = ((x == 0.0) & (y == PINF_F64)) ? QNAN_F64 : ret;
        ret = (BUILTIN_ISINF_F64(x) & (y == NINF_F64)) ? QNAN_F64 : ret;
    }

    return ret;
}

