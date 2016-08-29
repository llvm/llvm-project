/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(ilogb)(float x)
{
    uint ux = AS_UINT(x) & EXSIGNBIT_SP32;
    int r;
    if (AMD_OPT()) {
        r = BUILTIN_FREXP_EXP_F32(x) - 1;
    } else {
        r = (int)(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        int rs = -118 - (int)MATH_CLZI(ux);
        r = ux < 0x00800000u ? rs : r;
    }

    if (!FINITE_ONLY_OPT()) {
        r = ux > PINFBITPATT_SP32 ? FP_ILOGBNAN : r;
        r = ux == PINFBITPATT_SP32 ? INT_MAX : r;
        r = x == 0.0f ? FP_ILOGB0 : r;
    } else {
	r = x == 0.0f ? FP_ILOGB0 : r;
    }

    return r;
}

