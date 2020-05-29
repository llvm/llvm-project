/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(cos)

REQUIRES_16BIT_INSTS half
MATH_MANGLE(cos)(half x)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
    sc.s = -sc.s;

    short c =  AS_SHORT((r.i & 1) == (short)0 ? sc.c : sc.s);
    c ^= r.i > 1 ? (short)0x8000 : (short)0;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_ISFINITE_F16(ax) ? c : (short)QNANBITPATT_HP16;
    }

    return AS_HALF(c);
}

