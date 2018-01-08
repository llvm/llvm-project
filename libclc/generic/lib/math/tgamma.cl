/*
 * Copyright (c) 2016 Aaron Watry
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

#include "math.h"
#include "../clcmacro.h"

_CLC_OVERLOAD _CLC_DEF float tgamma(float x) {
    const float pi = 3.1415926535897932384626433832795f;
    float ax = fabs(x);
    float lg = lgamma(ax);
    float g = exp(lg);

    if (x < 0.0f) {
        float z = sinpi(x);
        g = g * ax * z;
        g = pi / g;
        g = g == 0 ? as_float(PINFBITPATT_SP32) : g;
        g = z == 0 ? as_float(QNANBITPATT_SP32) : g;
    }

    return g;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, tgamma, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double tgamma(double x) {
    const double pi = 3.1415926535897932384626433832795;
    double ax = fabs(x);
    double lg = lgamma(ax);
    double g = exp(lg);

    if (x < 0.0) {
        double z = sinpi(x);
        g = g * ax * z;
        g = pi / g;
        g = g == 0 ? as_double(PINFBITPATT_DP64) : g;
        g = z == 0 ? as_double(QNANBITPATT_DP64) : g;
    }

    return g;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, tgamma, double);

#endif
