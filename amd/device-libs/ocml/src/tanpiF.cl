/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(tanpi)(float x)
{
    int ix = AS_INT(x); 
    int xsgn = ix & SIGNBIT_SP32;
    int xnsgn = xsgn ^ SIGNBIT_SP32;
    ix ^= xsgn;
    float ax = AS_FLOAT(ix);
    int iax = (int)ax;
    float r = BUILTIN_FRACTION_F32(ax);
    int xodd = xsgn ^ (iax & 0x1 ? SIGNBIT_SP32 : 0);

    // Initialize with return for +-Inf and NaN
    int ir = QNANBITPATT_SP32;

    // 2^24 <= |x| < Inf, the result is always even integer
    ir = ix < PINFBITPATT_SP32 ? xsgn : ir;

    // 2^23 <= |x| < 2^24, the result is always integer
    ir = ix < 0x4b800000 ? xodd : ir;

    // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

    // r < 1.0
    float a = 1.0f - r;
    int e = 0;
    int s = xnsgn;

    // r <= 0.75
    bool c = r <= 0.75f;
    float ta = r - 0.5f;
    a = c ? ta : a;
    e = c ? 1 : e;
    s = c ? xsgn : s;

    // r < 0.5
    c = r < 0.5f;
    ta = 0.5f - r;
    a = c ? ta : a;
    s = c ? xnsgn : s;

    // 0 < r <= 0.25
    c = r <= 0.25f;
    a = c ? r : a;
    e = c ? 0 : e;
    s = c ? xsgn : s;

    const float pi = 0x1.921fb6p+1f;
    float t = MATH_PRIVATE(tanred)(a * pi, e);
    int jr = s ^ AS_INT(t);

    jr = r == 0.5f ? xodd | PINFBITPATT_SP32 : jr;

    ir = ix < 0x4b000000 ? jr : ir;

    return AS_FLOAT(ir);
}

