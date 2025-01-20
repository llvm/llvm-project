/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <clc/clc.h>
#include <clc/clcmacro.h>

#include "math.h"
#include "sincos_helpers.h"

_CLC_OVERLOAD _CLC_DEF float cos(float x)
{
    int ix = as_int(x);
    int ax = ix & 0x7fffffff;
    float dx = as_float(ax);

    float r0, r1;
    int regn = __clc_argReductionS(&r0, &r1, dx);

    float ss = -__clc_sinf_piby4(r0, r1);
    float cc =  __clc_cosf_piby4(r0, r1);

    float c =  (regn & 1) != 0 ? ss : cc;
    c = as_float(as_int(c) ^ ((regn > 1) << 31));

    c = ax >= PINFBITPATT_SP32 ? as_float(QNANBITPATT_SP32) : c;

    return c;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, cos, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double cos(double x) {
    x = fabs(x);

    double r, rr;
    int regn;

    if (x < 0x1.0p+47)
        __clc_remainder_piby2_medium(x, &r, &rr, &regn);
    else
        __clc_remainder_piby2_large(x, &r, &rr, &regn);

    double2 sc = __clc_sincos_piby4(r, rr);
    sc.lo = -sc.lo;

    int2 c = as_int2(regn & 1 ? sc.lo : sc.hi);
    c.hi ^= (regn > 1) << 31;

    return isnan(x) | isinf(x) ? as_double(QNANBITPATT_DP64) : as_double(c);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, cos, double);

#endif

_CLC_DEFINE_UNARY_BUILTIN_FP16(cos)
