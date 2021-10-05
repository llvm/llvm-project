/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float
MATH_PRIVATE(lnep)(float2 a, int ea)
{
    int b = BUILTIN_FREXP_MANT_F32(a.hi) < (2.0f/3.0f);
    int e = BUILTIN_FREXP_EXP_F32(a.hi) - b;
    float2 m = ldx(a, -e);
    float2 x = div(fadd(-1.0f, m), fadd(1.0f, m));
    float s = x.hi * x.hi;
    float p = MATH_MAD(s, MATH_MAD(s, 0x1.36db58p-2f, 0x1.992b46p-2f), 0x1.5555b4p-1f);
    float2 r = add(mul(con(0x1.62e430p-1f, -0x1.05c610p-29f), (float)(e + ea)),
                   fadd(ldx(x,1), s * x.hi * p));
    return r.hi;
}

