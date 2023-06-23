/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(rlen4)(float x, float y, float z, float w)
{
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float c = BUILTIN_ABS_F32(z);
    float d = BUILTIN_ABS_F32(w);

    float a1 = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a), AS_UINT(b)));
    float b1 = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(a), AS_UINT(b)));

    float c1 = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(c), AS_UINT(d)));
    float d1 = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(c), AS_UINT(d)));

    a        = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a1), AS_UINT(c1)));
    float c2 = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(a1), AS_UINT(c1)));

    float b2 = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(b1), AS_UINT(d1)));
    d        = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(b1), AS_UINT(d1)));

    b        = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(b2), AS_UINT(c2)));
    c        = AS_FLOAT(BUILTIN_MIN_U32(AS_UINT(b2), AS_UINT(c2)));

    int e = BUILTIN_FREXP_EXP_F32(a);
    a = BUILTIN_FLDEXP_F32(a, -e);
    b = BUILTIN_FLDEXP_F32(b, -e);
    c = BUILTIN_FLDEXP_F32(c, -e);
    d = BUILTIN_FLDEXP_F32(d, -e);

    float ret = BUILTIN_FLDEXP_F32(BUILTIN_AMDGPU_RSQRT_F32(MATH_MAD(a, a, MATH_MAD(b, b, MATH_MAD(c, c, d*d)))), -e);

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISINF_F32(x) |
               BUILTIN_ISINF_F32(y) |
               BUILTIN_ISINF_F32(z) |
               BUILTIN_ISINF_F32(w)) ? 0.0f : ret;
    }

    return ret;
}

