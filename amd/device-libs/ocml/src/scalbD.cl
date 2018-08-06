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
        ret = (BUILTIN_ISNAN_F64(x) | BUILTIN_ISNAN_F64(y)) ?  AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = (BUILTIN_CLASS_F64(x, CLASS_NZER|CLASS_PZER) & BUILTIN_CLASS_F64(y, CLASS_PINF)) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = (BUILTIN_ISINF_F64(x) & BUILTIN_CLASS_F64(y, CLASS_NINF)) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
    }

    return ret;
}

