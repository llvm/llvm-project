/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(succ)(half x)
{
    short ix = AS_SHORT(x);
    short mx = (short)SIGNBIT_HP16 - ix;
    mx = ix < (short)0 ? mx : ix;
    short t = mx + (short)(x != PINF_F16 && !BUILTIN_ISNAN_F16(x));
    short r = (short)SIGNBIT_HP16 - t;
    r = t < (short)0 ? r : t;
    r = mx == (short)-1 ? (short)SIGNBIT_HP16 : r;
    return AS_HALF(r);
}

