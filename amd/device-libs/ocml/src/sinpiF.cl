/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

INLINEATTR float
MATH_MANGLE(sinpi)(float x)
{
    int ix = AS_INT(x); 
    int xsgn = ix & SIGNBIT_SP32;
    ix ^= xsgn;
    float ax = AS_FLOAT(ix);
    int iax = (int)ax;
    float r = BUILTIN_FRACTION_F32(ax);
    int xodd = xsgn ^ (iax & 0x1 ? SIGNBIT_SP32 : 0);
    int ir;

    // 2^23 <= |x| < Inf, the result is always integer
    if (!FINITE_ONLY_OPT()) {
        ir = ix < PINFBITPATT_SP32 ? xsgn : QNANBITPATT_SP32;
    } else {
	ir = xsgn;
    }

    // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

    // r < 1.0
    float a = 1.0f - r;
    int e = 0;

    // r <= 0.75
    bool c = r <= 0.75f;
    float ta = r - 0.5f;
    a = c ? ta : a;
    e = c ? 1 : e;

    // r < 0.5
    c = r < 0.5f;
    ta = 0.5f - r;
    a = c ? ta : a;

    // 0 < r <= 0.25
    c = r <= 0.25f;
    a = c ? r : a;
    e = c ? 0 : e;

    const float pi = 0x1.921fb6p+1f;
    float ca;
    float sa = MATH_PRIVATE(sincosred)(a*pi, &ca);

    int jr = xodd ^ AS_INT(e ? ca : sa);
    ir = ix < 0x4b000000 ? jr : ir;

    return AS_FLOAT(ir);
}

