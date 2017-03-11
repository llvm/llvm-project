/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(tan)

INLINEATTR half
MATH_MANGLE(tan)(half x)
{
    half r;
    short i = MATH_PRIVATE(trigred)(&r, BUILTIN_ABS_F16(x));

    short t = AS_SHORT(MATH_PRIVATE(tanred)(r, i & 1));
    t ^= AS_SHORT(x) & (short)0x8000;

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : t;
    }

    return AS_HALF(t);
}

