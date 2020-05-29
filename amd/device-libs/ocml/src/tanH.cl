/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(tan)

REQUIRES_16BIT_INSTS half
MATH_MANGLE(tan)(half x)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    short t = AS_SHORT(MATH_PRIVATE(tanred)(r.hi, r.i & (short)1));
    t ^= AS_SHORT(x) & (short)0x8000;

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISFINITE_F16(ax) ? t : (short)QNANBITPATT_HP16;
    }

    return AS_HALF(t);
}

