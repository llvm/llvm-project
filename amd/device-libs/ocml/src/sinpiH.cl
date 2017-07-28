/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

UGEN(sinpi)

half
MATH_MANGLE(sinpi)(half x)
{
    struct redret r =  MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    short s = AS_SHORT((r.i & (short)1) == (short)0 ? sc.s : sc.c);
    s ^= (r.i > (short)1 ? (short)0x8000 : (short)0) ^ (AS_SHORT(x) & (short)0x8000);

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : s;
    }

    return AS_HALF(s);
}

