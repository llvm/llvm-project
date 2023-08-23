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
MATH_MANGLE(casinh)(float2 z)
{
    float x = BUILTIN_ABS_F32(z.x);
    float y = BUILTIN_ABS_F32(z.y);

    float2 l2, t;
    int e = 0;
    bool b = true;

    if (x < 0x1.0p+25f && y < 0x1.0p+25f) {
        if (y >= 1.0f || x >= 0x1.0p-24f || x > (1.0f - y)*0x1.0p-12f) {
            float4 z2p1 = (float4)(add(mul(add(x,y), sub(x,y)), 1.0f), mul(y,x)*2.0f);
            float4 rz2p1 = MATH_PRIVATE(epcsqrtep)(z2p1);
            float4 s = (float4)(add(rz2p1.lo, x), add(rz2p1.hi, y));
            l2 = add(sqr(s.lo), sqr(s.hi));
            t = (float2)(s.s1, s.s3);
        } else {
            b = false;
            float r = MATH_SQRT(BUILTIN_FMA_F32(-y, y, 1.0f));
            l2 = con(MATH_DIV(x, r), 0.0f);
            t = (float2)(r, y);
        }
    } else {
        t = (float2)(x, y);
        e = BUILTIN_FREXP_EXP_F32(AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(x), AS_UINT(y))));
        x = BUILTIN_FLDEXP_F32(x, -e);
        y = BUILTIN_FLDEXP_F32(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
    }

    float rr;
    if (b) {
        rr = 0.5f * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    rr = BUILTIN_COPYSIGN_F32(rr, z.x);
    float ri = BUILTIN_COPYSIGN_F32(MATH_MANGLE(atan2)(t.y, t.x), z.y);

    if (!FINITE_ONLY_OPT()) {
        float i = BUILTIN_COPYSIGN_F32(PINF_F32, z.x);
        rr = (BUILTIN_ISINF_F32(z.x) | BUILTIN_ISINF_F32(z.y)) ? i : rr;
    }

    return (float2)(rr, ri);
}

