/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(cos)

half
MATH_MANGLE(cos)(half x)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
    sc.s = -sc.s;

    half c = (r.i & 1) == (short)0 ? sc.c : sc.s;

    short flip = r.i > 1 ? (short)SIGNBIT_HP16 : 0;
    c = AS_HALF((short)(AS_SHORT(c) ^ flip));

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_ISFINITE_F16(ax) ? c : QNAN_F16;
    }

    return c;
}

