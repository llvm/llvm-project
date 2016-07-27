/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(hypot)(float x, float y)
{
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float t = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a), AS_UINT(b)));

    int e;
    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F32(t) - 1;
        e = BUILTIN_CLAMP_S32(e, -126, 126);
        a = BUILTIN_FLDEXP_F32(a, -e);
        b = BUILTIN_FLDEXP_F32(b, -e);
    } else {
        e = (int)(AS_UINT(t) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -126), 126);
        float sc = AS_FLOAT((EXPBIAS_SP32 - e) << EXPSHIFTBITS_SP32);
        a *= sc;
        b *= sc;
    }

    float ret = MATH_FAST_SQRT(MATH_MAD(a, a, b*b));

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F32(ret, e);
    } else {
        ret *= AS_FLOAT((EXPBIAS_SP32 + e) << EXPSHIFTBITS_SP32);
    }

    if (!FINITE_ONLY_OPT()) {
        ret = AS_UINT(t) > PINFBITPATT_SP32 ? t : ret;
        ret = (BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F32(y, CLASS_PINF|CLASS_NINF)) ?
              AS_FLOAT(PINFBITPATT_SP32) : ret;
    }

    return ret;
}

