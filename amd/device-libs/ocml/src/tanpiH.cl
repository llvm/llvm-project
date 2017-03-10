/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

CONSTATTR UGEN(tanpi)

CONSTATTR INLINEATTR half
MATH_MANGLE(tanpi)(half x)
{
    half r;
    short i = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x), &r);

    short t = AS_SHORT(MATH_PRIVATE(tanpired)(r, i & (short)1));
    t ^= (((i == (short)1) | (i == (short)2)) & (r == 0.0h)) ? (short)0x8000 : (short)0;
    t ^= AS_SHORT(x) & (short)0x8000;

    if (!FINITE_ONLY_OPT()) {
        t =  BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : t;
    }

    return AS_HALF(t);
}

