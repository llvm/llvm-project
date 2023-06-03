/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(pred)(half x)
{
    short ix = AS_SHORT(x);
    short mx = (short)SIGNBIT_HP16 - ix;
    mx = ix < (short)0 ? mx : ix;
    short t = mx - (short)(x != NINF_F16 && !BUILTIN_ISNAN_F16(x));
    short r = (short)SIGNBIT_HP16 - t;
    r = t < (short)0 ? r : t;
    return AS_HALF(r);
}

