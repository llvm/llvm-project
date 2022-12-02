/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

UGEN(sinpi)

REQUIRES_16BIT_INSTS half
MATH_MANGLE(sinpi)(half x)
{
    struct redret r =  MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    short s = AS_SHORT((r.i & (short)1) == (short)0 ? sc.s : sc.c);
    s ^= (r.i > (short)1 ? (short)0x8000 : (short)0) ^ (AS_SHORT(x) & (short)0x8000);

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F16(x) ? s : (short)QNANBITPATT_HP16;
    }

    return AS_HALF(s);
}

