/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(hypot)(float x, float y)
{
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float t = BUILTIN_MAX_F32(a, b);
    int e = BUILTIN_FREXP_EXP_F32(t) ;
    a = BUILTIN_FLDEXP_F32(a, -e);
    b = BUILTIN_FLDEXP_F32(b, -e);
    float ret = BUILTIN_FLDEXP_F32(MATH_FAST_SQRT(MATH_MAD(a, a, b*b)), e);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F32(t) ? PINF_F32 : ret;
    }

    return ret;
}

