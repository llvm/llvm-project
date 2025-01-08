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

_CLC_OVERLOAD _CLC_DEF float acos(float x) {
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.


    // Some constants and split constants.
    const float piby2 = 1.5707963705e+00F;
    const float pi = 3.1415926535897933e+00F;
    const float piby2_head = 1.5707963267948965580e+00F;
    const float piby2_tail = 6.12323399573676603587e-17F;

    uint ux = as_uint(x);
    uint aux = ux & ~SIGNBIT_SP32;
    int xneg = ux != aux;
    int xexp = (int)(aux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    float y = as_float(aux);

    // transform if |x| >= 0.5
    int transform = xexp >= -1;

    float y2 = y * y;
    float yt = 0.5f * (1.0f - y);
    float r = transform ? yt : y2;

    // Use a rational approximation for [0.0, 0.5]
    float a = mad(r,
                  mad(r,
                      mad(r, -0.00396137437848476485201154797087F, -0.0133819288943925804214011424456F),
                      -0.0565298683201845211985026327361F),
                  0.184161606965100694821398249421F);

    float b = mad(r, -0.836411276854206731913362287293F, 1.10496961524520294485512696706F);
    float u = r * MATH_DIVIDE(a, b);

    float s = MATH_SQRT(r);
    y = s;
    float s1 = as_float(as_uint(s) & 0xffff0000);
    float c = MATH_DIVIDE(mad(s1, -s1, r), s + s1);
    float rettn = mad(s + mad(y, u, -piby2_tail), -2.0f, pi);
    float rettp = 2.0F * (s1 + mad(y, u, c));
    float rett = xneg ? rettn : rettp;
    float ret = piby2_head - (x - mad(x, -u, piby2_tail));

    ret = transform ? rett : ret;
    ret = aux > 0x3f800000U ? as_float(QNANBITPATT_SP32) : ret;
    ret = ux == 0x3f800000U ? 0.0f : ret;
    ret = ux == 0xbf800000U ? pi : ret;
    ret = xexp < -26 ? piby2 : ret;
    return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, acos, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double acos(double x) {
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const double pi = 3.1415926535897933e+00;             /* 0x400921fb54442d18 */
    const double piby2 = 1.5707963267948965580e+00;       /* 0x3ff921fb54442d18 */
    const double piby2_head = 1.5707963267948965580e+00;  /* 0x3ff921fb54442d18 */
    const double piby2_tail = 6.12323399573676603587e-17; /* 0x3c91a62633145c07 */

    double y = fabs(x);
    int xneg = as_int2(x).hi < 0;
    int xexp = (as_int2(y).hi >> 20) - EXPBIAS_DP64;

    // abs(x) >= 0.5
    int transform = xexp >= -1;

    double rt = 0.5 * (1.0 - y);
    double y2 = y * y;
    double r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    double un = fma(r,
                    fma(r,
                        fma(r,
                            fma(r,
                                fma(r, 0.0000482901920344786991880522822991,
                                       0.00109242697235074662306043804220),
                                -0.0549989809235685841612020091328),
                            0.275558175256937652532686256258),
                        -0.445017216867635649900123110649),
                    0.227485835556935010735943483075);

    double ud = fma(r,
                    fma(r,
                        fma(r,
                            fma(r, 0.105869422087204370341222318533,
                                   -0.943639137032492685763471240072),
                            2.76568859157270989520376345954),
                        -3.28431505720958658909889444194),
                    1.36491501334161032038194214209);

    double u = r * MATH_DIVIDE(un, ud);

    // Reconstruct acos carefully in transformed region
    double s = sqrt(r);
    double ztn =  fma(-2.0, (s + fma(s, u, -piby2_tail)), pi);

    double s1 = as_double(as_ulong(s) & 0xffffffff00000000UL);
    double c = MATH_DIVIDE(fma(-s1, s1, r), s + s1);
    double ztp = 2.0 * (s1 + fma(s, u, c));
    double zt =  xneg ? ztn : ztp;
    double z = piby2_head - (x - fma(-x, u, piby2_tail));

    z =  transform ? zt : z;

    z = xexp < -56 ? piby2 : z;
    z = isnan(x) ? as_double((as_ulong(x) | QNANBITPATT_DP64)) : z;
    z = x == 1.0 ? 0.0 : z;
    z = x == -1.0 ? pi : z;

    return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, acos, double);

#endif // cl_khr_fp64

_CLC_DEFINE_UNARY_BUILTIN_FP16(acos)
