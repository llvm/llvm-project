/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(len3)(float x, float y, float z)
{
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float c = BUILTIN_ABS_F32(z);

    float a1 = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a), AS_UINT(b)));
    float b1 = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(a), AS_UINT(b)));

    a        = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a1), AS_UINT(c)));
    float c1 = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(a1), AS_UINT(c)));

    b        = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(b1), AS_UINT(c1)));
    c        = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(b1), AS_UINT(c1)));

    int e;
    if (AMD_OPT()) {
        e = BUILTIN_FREXP_EXP_F32(a) - 1;
        e = BUILTIN_CLAMP_S32(e, -126, 126);
        a = BUILTIN_FLDEXP_F32(a, -e);
        b = BUILTIN_FLDEXP_F32(b, -e);
        c = BUILTIN_FLDEXP_F32(c, -e);
    } else {
        e = (int)(AS_INT(a) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        e = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e, -126), 126);
        float sc = AS_FLOAT((EXPBIAS_SP32 - e) << EXPSHIFTBITS_SP32);
        a *= sc;
        b *= sc;
        c *= sc;
    }

    float ret = MATH_FAST_SQRT(MATH_MAD(a, a, MATH_MAD(b, b, c*c)));

    if (AMD_OPT()) {
        ret = BUILTIN_FLDEXP_F32(ret, e);
    } else {
        ret *= AS_FLOAT((EXPBIAS_SP32 + e) << EXPSHIFTBITS_SP32);
    }

    if (!FINITE_ONLY_OPT()) {
        ret = AS_UINT(a) > PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = (BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F32(y, CLASS_PINF|CLASS_NINF) |
               BUILTIN_CLASS_F32(z, CLASS_PINF|CLASS_NINF)) ? AS_FLOAT(PINFBITPATT_SP32) : ret;
    }

    return ret;
}

