/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(nextafter)

CONSTATTR INLINEATTR half
MATH_MANGLE(nextafter)(half x, half y)
{
    short ix = AS_SHORT(x);
    short ax = ix & (short)EXSIGNBIT_HP16;
    short mx = (short)SIGNBIT_HP16 - ix;
    mx = ix < (short)0 ? mx : ix;
    short iy = AS_SHORT(y);
    short ay = iy & (short)EXSIGNBIT_HP16;
    short my = (short)SIGNBIT_HP16 - iy;
    my = iy < (short)0 ? my : iy;
    short t = mx + (mx < my ? (short)1 : (short)-1);
    short r = (short)SIGNBIT_HP16 - t;
    r = t < (short)0 ? r : t;
    if (!FINITE_ONLY_OPT()) {
        r = ax > (short)PINFBITPATT_HP16 ? ix : r;
        r = ay > (short)PINFBITPATT_HP16 ? iy : r;
    }
    r = (ax | ay) == (short)0 | ix == iy ? iy : r;
    return AS_HALF(r);
}

