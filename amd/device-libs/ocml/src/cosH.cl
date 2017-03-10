/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(cos)

INLINEATTR half
MATH_MANGLE(cos)(half x)
{
    half r;
    short i = MATH_PRIVATE(trigred)(&r, BUILTIN_ABS_F16(x));

    half cc;
    half ss = -MATH_PRIVATE(sincosred)(r, &cc);

    short c =  AS_SHORT((i & 1) == 0 ? cc : ss);
    c ^= i > 1 ? (short)0x8000 : (short)0;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : c;
    }

    return AS_HALF(c);
}

