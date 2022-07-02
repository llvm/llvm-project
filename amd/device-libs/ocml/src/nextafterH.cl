/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(nextafter)

CONSTATTR half
MATH_MANGLE(nextafter)(half x, half y)
{
    short ix = AS_SHORT(x);
    short mx = (short)SIGNBIT_HP16 - ix;
    mx = ix < (short)0 ? mx : ix;
    short iy = AS_SHORT(y);
    short my = (short)SIGNBIT_HP16 - iy;
    my = iy < (short)0 ? my : iy;
    short t = mx + (mx < my ? (short)1 : (short)-1);
    short r = (short)SIGNBIT_HP16 - t;
    r = t < (short)0 ? r : t;
    r = (mx == (short)-1 && mx < my) ? (short)SIGNBIT_HP16 : r;

    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_ISNAN_F16(x) ? ix : r;
        r = BUILTIN_ISNAN_F16(y) ? iy : r;
    }

    r = (ix == iy || (AS_SHORT(BUILTIN_ABS_F16(x)) | AS_SHORT(BUILTIN_ABS_F16(y))) == (short)0) ? iy : r;
    return AS_HALF(r);
}

