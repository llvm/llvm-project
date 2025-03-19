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
#include <clc/math/math.h>

#include "sincospiF_piby4.h"
#ifdef cl_khr_fp64
#include "sincosD_piby4.h"
#endif

_CLC_OVERLOAD _CLC_DEF float sinpi(float x)
{
    int ix = as_int(x);
    int xsgn = ix & 0x80000000;
    ix ^= xsgn;
    float ax = as_float(ix);
    int iax = (int)ax;
    float r = ax - iax;
    int xodd = xsgn ^ (iax & 0x1 ? 0x80000000 : 0);

    // Initialize with return for +-Inf and NaN
    int ir = 0x7fc00000;

    // 2^23 <= |x| < Inf, the result is always integer
    ir = ix < 0x7f800000 ? xsgn : ir;

    // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

    // r < 1.0
    float a = 1.0f - r;
    int e = 0;

    // r <= 0.75
    int c = r <= 0.75f;
    a = c ? r - 0.5f : a;
    e = c ? 1 : e;

    // r < 0.5
    c = r < 0.5f;
    a = c ? 0.5f - r : a;

    // 0 < r <= 0.25
    c = r <= 0.25f;
    a = c ? r : a;
    e = c ? 0 : e;

    float2 t = __libclc__sincosf_piby4(a * M_PI_F);
    int jr = xodd ^ as_int(e ? t.hi : t.lo);

    ir = ix < 0x4b000000 ? jr : ir;

    return as_float(ir);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, sinpi, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double sinpi(double x)
{
    long ix = as_long(x);
    long xsgn = ix & 0x8000000000000000L;
    ix ^= xsgn;
    double ax = as_double(ix);
    long iax = (long)ax;
    double r = ax - (double)iax;
    long xodd = xsgn ^ (iax & 0x1L ? 0x8000000000000000L : 0L);

    // Initialize with return for +-Inf and NaN
    long ir = 0x7ff8000000000000L;

    // 2^23 <= |x| < Inf, the result is always integer
    ir = ix < 0x7ff0000000000000 ? xsgn : ir;

    // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

    // r < 1.0
    double a = 1.0 - r;
    int e = 0;

    //  r <= 0.75
    int c = r <= 0.75;
    double t = r - 0.5;
    a = c ? t : a;
    e = c ? 1 : e;

    // r < 0.5
    c = r < 0.5;
    t = 0.5 - r;
    a = c ? t : a;

    // r <= 0.25
    c = r <= 0.25;
    a = c ? r : a;
    e = c ? 0 : e;

    double api = a * M_PI;
    double2 sc = __libclc__sincos_piby4(api, 0.0);
    long jr = xodd ^ as_long(e ? sc.hi : sc.lo);

    ir = ax < 0x1.0p+52 ? jr : ir;

    return as_double(ir);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, sinpi, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_FP16(sinpi)

#endif
