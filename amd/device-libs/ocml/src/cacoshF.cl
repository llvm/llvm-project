/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float4 MATH_PRIVATE(epcsqrtep)(float4 z);
extern CONSTATTR float MATH_PRIVATE(lnep)(float2 a, int ea);

CONSTATTR float2
MATH_MANGLE(cacosh)(float2 z)
{
    float x = BUILTIN_ABS_F32(z.x);
    float y = BUILTIN_ABS_F32(z.y);

    float2 l2, t;
    int e = 0;
    bool b = true;

    if (x < 0x1.0p+25f && y < 0x1.0p+25f) {
        if (x >= 1.0f || y >= 0x1.0p-24f || y > (1.0f - x)*0x1.0p-12f) {
            float4 z2p1 = (float4)(add(mul(add(y,x), sub(y,x)), 1.0f), mul(y,x)*2.0f);
            float4 rz2m1 = MATH_PRIVATE(epcsqrtep)(z2p1);
            rz2m1 = (float4)(csgn(rz2m1.hi, (float2)z.x), csgn(rz2m1.lo, (float2)z.y));
            float4 s = (float4)(add(rz2m1.lo, z.x), add(rz2m1.hi, z.y));
            l2 = add(sqr(s.lo), sqr(s.hi));
            t = (float2)(s.s1, z.y == 0.0f ? z.y : s.s3);
        } else {
            b = false;
            float r = MATH_SQRT(BUILTIN_FMA_F32(-x, x, 1.0f));
            l2 = con(MATH_DIV(y, r), 0.0f);
            t = (float2)(z.x, BUILTIN_COPYSIGN_F32(r, z.y));
        }
    } else {
        e = BUILTIN_FREXP_EXP_F32(AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(x), AS_UINT(y))));
        x = BUILTIN_FLDEXP_F32(x, -e);
        y = BUILTIN_FLDEXP_F32(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
        t = z;
    }

    float rr;
    if (b) {
        rr = 0.5f * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    float ri = MATH_MANGLE(atan2)(t.y, t.x);

    if (!FINITE_ONLY_OPT()) {
        rr = (BUILTIN_ISINF_F32(z.x) | BUILTIN_ISINF_F32(z.y)) ? PINF_F32 : rr;
    }

    return (float2)(rr, ri);
}

