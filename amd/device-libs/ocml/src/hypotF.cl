
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(hypot)(float x, float y)
{
    uint uax = AS_UINT(x) & EXSIGNBIT_SP32;
    uint uay = AS_UINT(y) & EXSIGNBIT_SP32;
    uint ux = BUILTIN_MAX_U32(uax, uay);
    uint uy = BUILTIN_MIN_U32(uax, uay);

    int e;

    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F32(AS_FLOAT(ux)) - 1;
        e = BUILTIN_CLAMP_S32(e, -126, 126);
        x = BUILTIN_FLDEXP_F32(AS_FLOAT(ux), -e);
        y = BUILTIN_FLDEXP_F32(AS_FLOAT(uy), -e);
    } else {
        e = (int)(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -126), 126);

        float sc = AS_FLOAT((EXPBIAS_SP32 - e) << EXPSHIFTBITS_SP32);
        x = AS_FLOAT(ux) * sc;
        y = AS_FLOAT(uy) * sc;
    }

    float ret = MATH_FAST_SQRT(MATH_MAD(x, x, y*y));

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F32(ret, e);
    } else {
        float sc = AS_FLOAT((EXPBIAS_SP32 + e) << EXPSHIFTBITS_SP32);
        ret *= sc;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = ux > PINFBITPATT_SP32 ? AS_FLOAT(ux) : ret;
        ret = ux == PINFBITPATT_SP32 | uy == PINFBITPATT_SP32 ? AS_FLOAT(PINFBITPATT_SP32) : ret;
    }

    return ret;
}

