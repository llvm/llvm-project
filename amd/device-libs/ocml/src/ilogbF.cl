
#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(ilogb)(float x)
{
    uint ux = as_uint(x) & EXSIGNBIT_SP32;
    int r;
    if (AMD_OPT()) {
        r = BUILTIN_FREXP_EXP_F32(x) - 1;
    } else {
        r = (int)(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        int rs = -118 - (int)MATH_CLZI(ux);
        r = ux < 0x00800000u ? rs : r;
    }

    if (!FINITE_ONLY_OPT()) {
        r = x == 0.0f | ux > PINFBITPATT_SP32 ? (int)0x80000000 : r;
        r = ux == PINFBITPATT_SP32 ? 0x7fffffff : r;
    } else {
	r = x == 0.0f ? (int)0x80000000 : r;
    }

    return r;
}

