/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float2
MATH_PRIVATE(epln)(float a)
{
    float m = BUILTIN_FREXP_MANT_F32(a);
    int b = m < (2.0f/3.0f);
    m = BUILTIN_FLDEXP_F32(m, b);
    int e = BUILTIN_FREXP_EXP_F32(a) - b;

    float2 x = div(m - 1.0f, fadd(1.0f, m));
    float2 s = sqr(x);
    float t = s.hi;
    float p = MATH_MAD(t, MATH_MAD(t, 0x1.ed89c2p-3f, 0x1.23e988p-2f), 0x1.999bdep-2f);

    // ln(2)*e + 2*x + x^3(c3 + x^2*p)
    float2 r = add(mul(con(0x1.62e430p-1f, -0x1.05c610p-29f), (float)e),
                   fadd(ldx(x,1),
                        mul(mul(s, x), 
                            fadd(con(0x1.555554p-1f,0x1.e72020p-29f),
                                 mul(s, p)))));

    return r;
}

