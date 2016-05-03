
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(hypot)(float x, float y)
{
    uint uax = as_uint(x) & EXSIGNBIT_SP32;
    uint uay = as_uint(y) & EXSIGNBIT_SP32;
    uint ux = BUILTIN_MAX_U32(uax, uay);
    uint uy = BUILTIN_MIN_U32(uax, uay);

    int e;

    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F32(as_float(ux)) - 1;
        e = BUILTIN_MEDIAN3_S32(e, -126, 126);
        x = BUILTIN_FLDEXP_F32(as_float(ux), -e);
        y = BUILTIN_FLDEXP_F32(as_float(uy), -e);
    } else {
        e = (int)(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -126), 126);

        float sc = as_float((EXPBIAS_SP32 - e) << EXPSHIFTBITS_SP32);
        x = as_float(ux) * sc;
        y = as_float(uy) * sc;
    }

    float ret = MATH_FAST_SQRT(MATH_MAD(x, x, y*y));

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F32(ret, e);
    } else {
        float sc = as_float((EXPBIAS_SP32 + e) << EXPSHIFTBITS_SP32);
        ret *= sc;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = ux > PINFBITPATT_SP32 ? as_float(ux) : ret;
        ret = ux == PINFBITPATT_SP32 | uy == PINFBITPATT_SP32 ? as_float(PINFBITPATT_SP32) : ret;
    }

    return ret;
}

