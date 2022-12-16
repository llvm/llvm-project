/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE(csqrt)(float2 z)
{
    float a = BUILTIN_ABS_F32(z.x);
    float b = BUILTIN_ABS_F32(z.y);
    int e = BUILTIN_FREXP_EXP_F32(BUILTIN_MAX_F32(a, b));
    float as = BUILTIN_FLDEXP_F32(a, -e);
    float bs = BUILTIN_FLDEXP_F32(b, -e);
    float p = MATH_FAST_SQRT(MATH_MAD(as, as, bs*bs));
    int k = (e & 1) ^ 1; 
    p = BUILTIN_FLDEXP_F32(p + as, k);
    p = BUILTIN_FLDEXP_F32(MATH_FAST_SQRT(p), (e >> 1) - k);
    float q = BUILTIN_FLDEXP_F32(MATH_DIV(b, p), -1);
    q = p == 0.0f ? p : q;
    bool l = z.x < 0.0f;
    float rr = l ? q : p;
    float ri = l ? p : q;

    if (!FINITE_ONLY_OPT()) {
        bool i = BUILTIN_ISINF_F32(b);
        rr = i ? b : rr;
        ri = i ? b : ri;
        ri = z.x == NINF_F32 ? a : ri;
        rr = z.x == PINF_F32 ? a : rr;
    }

    return (float2)(rr, BUILTIN_COPYSIGN_F32(ri, z.y));
}

