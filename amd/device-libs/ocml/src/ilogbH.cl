/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(ilogb)(half x)
{
    int r;

    if (AMD_OPT()) {
        r = BUILTIN_FREXP_EXP_F16(x) - 1;
    } else {
        int ix = (int)AS_USHORT(x);
        r = ((ix >> 10) & 0x1f) - EXPBIAS_HP16;
        int rs = 7 - (int)MATH_CLZI(ix & MANTBITS_HP16);
        r = BUILTIN_CLASS_F16(x, CLASS_PSUB|CLASS_NSUB) ? rs : r;
    }

    if (!FINITE_ONLY_OPT()) {
        r = BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN) ? FP_ILOGBNAN : r;
        r = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) ? INT_MAX : r;
        r = x == 0.0h ? FP_ILOGB0 : r;
    } else {
	r = x == 0.0h ? FP_ILOGB0 : r;
    }

    return r;
}

