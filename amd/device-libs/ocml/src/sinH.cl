/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(sin)

REQUIRES_16BIT_INSTS half
MATH_MANGLE(sin)(half x)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc =  MATH_PRIVATE(sincosred)(r.hi);

    short s = AS_SHORT((r.i & (short)1) == (short)0 ? sc.s : sc.c);
    s ^= (r.i > (short)1 ? (short)0x8000 : (short)0) ^ (AS_SHORT(x) & (short)0x8000);

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F16(ax) ? s : (short)QNANBITPATT_HP16;
    }

    return AS_HALF(s);
}

