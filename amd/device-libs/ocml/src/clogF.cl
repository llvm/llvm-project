/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float MATH_PRIVATE(lnep)(float2 a, int ea);

CONSTATTR float2
MATH_MANGLE(clog)(float2 z)
{
    float x = z.s0;
    float y = z.s1;
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float t = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a), AS_UINT(b)));
    int e = BUILTIN_FREXP_EXP_F32(t) ;
    a = BUILTIN_FLDEXP_F32(a, -e);
    b = BUILTIN_FLDEXP_F32(b, -e);
    float rr = 0.5f * MATH_PRIVATE(lnep)(add(sqr(a), sqr(b)), 2*e);
    float ri = MATH_MANGLE(atan2)(y, x);
    

    if (!FINITE_ONLY_OPT()) {
        rr = ((x == 0.0f) & (y == 0.0f)) ? NINF_F32 : rr;
        rr = (BUILTIN_ISINF_F32(x) | BUILTIN_ISINF_F32(y)) ? PINF_F32 : rr;
    }

    return (float2)(rr, ri);
}

