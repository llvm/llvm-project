/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 z);

CONSTATTR float2
MATH_MANGLE(ctanh)(float2 z)
{
    float cy;
    float sy = MATH_MANGLE(sincos)(z.y, &cy);
    float cysy = cy*sy;
    float x = BUILTIN_ABS_F32(z.x);

    float rr, ri;
    if (x < 0x1.3687aap+3f) {
        float2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e430p-1, -0x1.05c610p-29f)));
        float2 er = rcp(e);
        er = ldx(er, -2);
        float2 cx = fadd(e, er);
        float2 sx = fsub(e, er);

        float cxhi = cx.hi;
        float sxhi = x < 0x1.0p-12f ? x : sx.hi;

        float d = MATH_MAD(cy, cy, sxhi*sxhi);
        rr = BUILTIN_COPYSIGN_F32(MATH_DIV(cxhi*sxhi, d), z.x);
        ri = MATH_DIV(cysy, d);
    } else {
        rr = BUILTIN_COPYSIGN_F32(1.0f, z.x);
        ri = 4.0f * cysy * MATH_MANGLE(exp)(-2.0f * x);
    }

    if (!FINITE_ONLY_OPT()) {
        bool xn = BUILTIN_ISNAN_F32(x);
        bool yin = !BUILTIN_ISFINITE_F32(z.y);
        bool ni = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR) & yin;
        rr = (ni | xn) ? QNAN_F32 : rr;
        ri = ni ? QNAN_F32 : ri;
        ri = (BUILTIN_ISINF_F32(x) & yin) ? 0.0f : ri;
        ri = (xn & (z.y == 0.0f)) ? z.y : ri;
    }

    return (float2)(rr, ri);
}

