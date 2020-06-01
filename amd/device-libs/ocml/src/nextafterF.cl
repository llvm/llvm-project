/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(nextafter)(float x, float y)
{
    int ix = AS_INT(x);
    int mx = SIGNBIT_SP32 - ix;
    mx = ix < 0 ? mx : ix;
    int iy = AS_INT(y);
    int my = SIGNBIT_SP32 - iy;
    my = iy < 0 ? my : iy;
    int t = mx + (mx < my ? 1 : -1);
    int r = SIGNBIT_SP32 - t;
    r = t < 0 ? r : t;
    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_ISNAN_F32(x) ? ix : r;
        r = BUILTIN_ISNAN_F32(y) ? iy : r;
    }

    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    r = ((AS_INT(ax) | AS_INT(ay)) == 0 | ix == iy) ? iy : r;
    return AS_FLOAT(r);
}

