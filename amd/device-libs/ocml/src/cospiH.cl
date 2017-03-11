/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

UGEN(cospi)

INLINEATTR half
MATH_MANGLE(cospi)(half x)
{
    half t;
    int i = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x), &t);

    half cc;
    half ss = -MATH_PRIVATE(sincospired)(t, &cc);

    short c =  AS_SHORT((i & (short)1) == (short)0 ? cc : ss);
    c ^= i > (short)1 ? (short)0x8000 : (short)0;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : c;
    }

    return AS_HALF(c);
}

