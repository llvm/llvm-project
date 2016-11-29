/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(logb)(half x)
{
    half ret;

    if (AMD_OPT()) {
        ret = (half)(BUILTIN_FREXP_EXP_F16(x) - (short)1);
    } else {
        int ix = (int)AS_USHORT(x);
        int r = ((ix >> 10) & 0x1f) - EXPBIAS_HP16;
        int rs = 7 - (int)MATH_CLZI(ix & MANTBITS_HP16);
        ret = (half)(r == -EXPBIAS_HP16 ? rs : r);
    }

    if (!FINITE_ONLY_OPT()) {
        half ax = BUILTIN_ABS_F16(x);
        ret = BUILTIN_CLASS_F16(ax, CLASS_PINF|CLASS_SNAN|CLASS_QNAN) ? ax : ret;
        ret = x == 0.0h ? AS_HALF((short)NINFBITPATT_HP16) : ret;
    }

    return ret;
}

