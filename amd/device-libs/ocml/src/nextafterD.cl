/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(nextafter)(double x, double y)
{
    long ix = AS_LONG(x);
    long ax = ix & EXSIGNBIT_DP64;
    long mx = SIGNBIT_DP64 - ix;
    mx = ix < 0 ? mx : ix;
    long iy = AS_LONG(y);
    long ay = iy & EXSIGNBIT_DP64;
    long my = SIGNBIT_DP64 - iy;
    my = iy < 0 ? my : iy;
    long t = mx + (mx < my ? 1 : -1);
    long r = SIGNBIT_DP64 - t;
    r = t < 0 ? r : t;
    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? ix : r;
        r = BUILTIN_CLASS_F64(y, CLASS_QNAN|CLASS_SNAN) ? iy : r;
    }
    r = (ax|ay) == 0L | ix == iy ? iy : r;
    return AS_DOUBLE(r);
}

