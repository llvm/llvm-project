/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_MANGLE(sin)(half x)
{
    half y = BUILTIN_ABS_F16(x);

    half r;
    int regn = MATH_PRIVATE(trigred)(&r, y);

    half cc;
    half ss = MATH_PRIVATE(sincosred)(r, &cc);

    half s = (regn & 1) != 0 ? cc : ss;
    half ns = -s;
    s = (regn > 1) ^ (x < 0.0h) ? ns : s;

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_HALF((short)QNANBITPATT_HP16) : s;
    }

    return s;
}

