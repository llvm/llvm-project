/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
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

_CLC_OVERLOAD _CLC_DEF float atanh(float x) {
    uint ux = as_uint(x);
    uint ax = ux & EXSIGNBIT_SP32;
    uint xs = ux ^ ax;

    // |x| > 1 or NaN
    float z = as_float(QNANBITPATT_SP32);

    // |x| == 1
    float t = as_float(xs | PINFBITPATT_SP32);
    z = ax == 0x3f800000U ? t : z;

    // 1/2 <= |x| < 1
    t = as_float(ax);
    t = MATH_DIVIDE(2.0f*t, 1.0f - t);
    t = 0.5f * log1p(t);
    t = as_float(xs | as_uint(t));
    z = ax < 0x3f800000U ? t : z;

    // |x| < 1/2
    t = x * x;
    float a = mad(mad(0.92834212715e-2f, t, -0.28120347286e0f), t, 0.39453629046e0f);
    float b = mad(mad(0.45281890445e0f, t, -0.15537744551e1f), t, 0.11836088638e1f);
    float p = MATH_DIVIDE(a, b);
    t = mad(x*t, p, x);
    z = ax < 0x3f000000 ? t : z;

    // |x| < 2^-13
    z = ax < 0x39000000U ? x : z;

    return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, atanh, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double atanh(double x) {
    double absx = fabs(x);

    double ret = absx == 1.0 ? as_double(PINFBITPATT_DP64) : as_double(QNANBITPATT_DP64);

    // |x| >= 0.5
    // Note that atanh(x) = 0.5 * ln((1+x)/(1-x))
    // For greater accuracy we use
    // ln((1+x)/(1-x)) = ln(1 + 2x/(1-x)) = log1p(2x/(1-x)).
    double r = 0.5 * log1p(2.0 * absx / (1.0 - absx));
    ret = absx < 1.0 ? r : ret;

    r = -ret;
    ret = x < 0.0 ? r : ret;

    // Arguments up to 0.5 in magnitude are
    // approximated by a [5,5] minimax polynomial
    double t = x * x;

    double pn = fma(t,
                    fma(t,
                        fma(t,
                            fma(t,
                                fma(t, -0.10468158892753136958e-3, 0.28728638600548514553e-1),
                                -0.28180210961780814148e0),
                            0.88468142536501647470e0),
                        -0.11028356797846341457e1),
                    0.47482573589747356373e0);

    double pd = fma(t,
                    fma(t,
                        fma(t,
                            fma(t,
                                fma(t, -0.35861554370169537512e-1, 0.49561196555503101989e0),
                                -0.22608883748988489342e1),
                            0.45414700626084508355e1),
                        -0.41631933639693546274e1),
                    0.14244772076924206909e1);

    r = fma(x*t, pn/pd, x);
    ret = absx < 0.5 ? r : ret;

    return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, atanh, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_FP16(atanh)

#endif
