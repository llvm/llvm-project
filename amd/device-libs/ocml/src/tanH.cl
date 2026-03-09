/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(tan)

half
MATH_MANGLE(tan)(half x)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    half t = MATH_PRIVATE(tanred)(r.hi, r.i & (short)1);

    t = AS_HALF((short)(AS_SHORT(t) ^ (AS_SHORT(x) & SIGNBIT_HP16)));

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISFINITE_F16(ax) ? t : QNAN_F16;
    }

    return t;
}

