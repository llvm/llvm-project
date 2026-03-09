/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(sin)

half
MATH_MANGLE(sin)(half x)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc =  MATH_PRIVATE(sincosred)(r.hi);

    half s = (r.i & (short)1) == (short)0 ? sc.s : sc.c;
    short flip = r.i > (short)1 ? (short)SIGNBIT_HP16 : (short)0;

    s = AS_HALF((short)(AS_SHORT(s) ^ (flip ^ (AS_SHORT(x) & (short)SIGNBIT_HP16))));

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F16(ax) ? s : QNAN_F16;
    }

    return AS_HALF(s);
}

