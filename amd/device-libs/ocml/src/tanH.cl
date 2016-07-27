/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_MANGLE(tan)(half x)
{
    half y = BUILTIN_ABS_F16(x);

    half r;
    int regn = MATH_PRIVATE(trigred)(&r, y);

    half t = MATH_PRIVATE(tanred)(r, regn);
    half nt = -t;
    t = x < 0.0h ? nt : t;

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_HALF((short)QNANBITPATT_HP16) : t;
    }

    return t;
}

