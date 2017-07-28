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
    int ax = ix & EXSIGNBIT_SP32;
    int mx = SIGNBIT_SP32 - ix;
    mx = ix < 0 ? mx : ix;
    int iy = AS_INT(y);
    int ay = iy & EXSIGNBIT_SP32;
    int my = SIGNBIT_SP32 - iy;
    my = iy < 0 ? my : iy;
    int t = mx + (mx < my ? 1 : -1);
    int r = SIGNBIT_SP32 - t;
    r = t < 0 ? r : t;
    if (!FINITE_ONLY_OPT()) {
        r = ax > PINFBITPATT_SP32 ? ix : r;
        r = ay > PINFBITPATT_SP32 ? iy : r;
    }
    r = (ax | ay) == 0 | ix == iy ? iy : r;
    return AS_FLOAT(r);
}

