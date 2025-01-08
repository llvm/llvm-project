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
#include "tables.h"

_CLC_DEF _CLC_OVERLOAD float __clc_tanpi(float x)
{
    int ix = as_int(x);
    int xsgn = ix & 0x80000000;
    int xnsgn = xsgn ^ 0x80000000;
    ix ^= xsgn;
    float ax = as_float(ix);
    int iax = (int)ax;
    float r = ax - iax;
    int xodd = xsgn ^ (iax & 0x1 ? 0x80000000 : 0);

    // Initialize with return for +-Inf and NaN
    int ir = 0x7fc00000;

    // 2^24 <= |x| < Inf, the result is always even integer
    ir = ix < 0x7f800000 ? xsgn : ir;

    // 2^23 <= |x| < 2^24, the result is always integer
    ir = ix < 0x4b800000 ? xodd : ir;

    // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

    // r < 1.0
    float a = 1.0f - r;
    int e = 0;
    int s = xnsgn;

    // r <= 0.75
    int c = r <= 0.75f;
    a = c ? r - 0.5f : a;
    e = c ? 1 : e;
    s = c ? xsgn : s;

    // r < 0.5
    c = r < 0.5f;
    a = c ? 0.5f - r : a;
    s = c ? xnsgn : s;

    // 0 < r <= 0.25
    c = r <= 0.25f;
    a = c ? r : a;
    e = c ? 0 : e;
    s = c ? xsgn : s;

    float t = __clc_tanf_piby4(a * M_PI_F, 0);
    float tr = -native_recip(t);
    int jr = s ^ as_int(e ? tr : t);

    jr = r == 0.5f ? xodd | 0x7f800000 : jr;

    ir = ix < 0x4b000000 ? jr : ir;

    return as_float(ir);
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_tanpi, float);

#ifdef cl_khr_fp64
#include "sincosD_piby4.h"

_CLC_DEF _CLC_OVERLOAD double __clc_tanpi(double x)
{
    long ix = as_long(x);
    long xsgn = ix & 0x8000000000000000L;
    long xnsgn = xsgn ^ 0x8000000000000000L;
    ix ^= xsgn;
    double ax = as_double(ix);
    long iax = (long)ax;
    double r = ax - iax;
    long xodd = xsgn ^ (iax & 0x1 ? 0x8000000000000000L : 0L);

    // Initialize with return for +-Inf and NaN
    long ir = 0x7ff8000000000000L;

    // 2^53 <= |x| < Inf, the result is always even integer
    ir = ix < 0x7ff0000000000000L ? xsgn : ir;

    // 2^52 <= |x| < 2^53, the result is always integer
    ir = ix < 0x4340000000000000L ? xodd : ir;

    // 0x1.0p-14 <= |x| < 2^53, result depends on which 0.25 interval

    // r < 1.0
    double a = 1.0 - r;
    int e = 0;
    long s = xnsgn;

    // r <= 0.75
    int c = r <= 0.75;
    double t = r - 0.5;
    a = c ? t : a;
    e = c ? 1 : e;
    s = c ? xsgn : s;

    // r < 0.5
    c = r < 0.5;
    t = 0.5 - r;
    a = c ? t : a;
    s = c ? xnsgn : s;

    // r <= 0.25
    c = r <= 0.25;
    a = c ? r : a;
    e = c ? 0 : e;
    s = c ? xsgn : s;

    double api = a * M_PI;
    double2 tt = __clc_tan_piby4(api, 0.0);
    long jr = s ^ as_long(e ? tt.hi : tt.lo);

    long si = xodd | 0x7ff0000000000000L;
    jr = r == 0.5 ? si : jr;

    ir = ix < 0x4330000000000000L ? jr : ir;

    return as_double(ir);
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_tanpi, double);
#endif
