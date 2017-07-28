/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

CONSTATTR UGEN(tanpi)

CONSTATTR half
MATH_MANGLE(tanpi)(half x)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x));
    short t = AS_SHORT(MATH_PRIVATE(tanpired)(r.hi, r.i & (short)1));
    t ^= (((r.i == (short)1) | (r.i == (short)2)) & (r.hi == 0.0h)) ? (short)0x8000 : (short)0;
    t ^= AS_SHORT(x) & (short)0x8000;

    if (!FINITE_ONLY_OPT()) {
        t =  BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? (short)QNANBITPATT_HP16 : t;
    }

    return AS_HALF(t);
}

