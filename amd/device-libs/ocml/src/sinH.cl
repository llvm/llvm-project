/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(sin)

INLINEATTR half
MATH_MANGLE(sin)(half x)
{
    half r;
    short i = MATH_PRIVATE(trigred)(&r, BUILTIN_ABS_F16(x));

    half cc;
    half ss = MATH_PRIVATE(sincosred)(r, &cc);

    short s = AS_SHORT((i & (short)1) == (short)0 ? ss : cc);
    s ^= (i > (short)1 ? (short)0x8000 : 0) ^ (AS_SHORT(x) & (short)0x8000);

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : s;
    }

    return AS_HALF(s);
}

