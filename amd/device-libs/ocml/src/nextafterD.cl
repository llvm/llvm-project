/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(nextafter)(double x, double y)
{
    long ix = AS_LONG(x);
    long mx = SIGNBIT_DP64 - ix;
    mx = ix < 0 ? mx : ix;
    long iy = AS_LONG(y);
    long my = SIGNBIT_DP64 - iy;
    my = iy < 0 ? my : iy;
    long t = mx + (mx < my ? 1 : -1);
    long r = SIGNBIT_DP64 - t;
    r = t < 0 ? r : t;
    r = (mx == -1L && mx < my) ? SIGNBIT_DP64 : r;

    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_ISNAN_F64(x) ? ix : r;
        r = BUILTIN_ISNAN_F64(y) ? iy : r;
    }

    r = (ix == iy || (AS_LONG(BUILTIN_ABS_F64(x)) | AS_LONG(BUILTIN_ABS_F64(y))) == 0L) ? iy : r;
    return AS_DOUBLE(r);
}

