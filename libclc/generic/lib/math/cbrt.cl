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
#include <clc/math/tables.h>

_CLC_OVERLOAD _CLC_DEF float cbrt(float x) {

    uint xi = as_uint(x);
    uint axi = xi & EXSIGNBIT_SP32;
    uint xsign = axi ^ xi;
    xi = axi;

    int m = (xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;

    // Treat subnormals
    uint xisub = as_uint(as_float(xi | 0x3f800000) - 1.0f);
    int msub = (xisub >> EXPSHIFTBITS_SP32) - 253;
    int c = m == -127;
    xi = c ? xisub : xi;
    m = c ? msub : m;

    int m3 = m / 3;
    int rem = m - m3*3;
    float mf = as_float((m3 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);

    uint indx = (xi & 0x007f0000) + ((xi & 0x00008000) << 1);
    float f = as_float((xi & MANTBITS_SP32) | 0x3f000000) - as_float(indx | 0x3f000000);

    indx >>= 16;
    float r = f * USE_TABLE(log_inv_tbl, indx);
    float poly = mad(mad(r, 0x1.f9add4p-5f, -0x1.c71c72p-4f), r*r, r * 0x1.555556p-2f);

    // This could also be done with a 5-element table
    float remH = 0x1.428000p-1f;
    float remT = 0x1.45f31ap-14f;

    remH = rem == -1 ? 0x1.964000p-1f : remH;
    remT = rem == -1 ? 0x1.fea53ep-13f : remT;

    remH = rem ==  0 ? 0x1.000000p+0f : remH;
    remT = rem ==  0 ? 0x0.000000p+0f  : remT;

    remH = rem ==  1 ? 0x1.428000p+0f : remH;
    remT = rem ==  1 ? 0x1.45f31ap-13f : remT;

    remH = rem ==  2 ? 0x1.964000p+0f : remH;
    remT = rem ==  2 ? 0x1.fea53ep-12f : remT;

    float2 tv = USE_TABLE(cbrt_tbl, indx);
    float cbrtH = tv.s0;
    float cbrtT = tv.s1;

    float bH = cbrtH * remH;
    float bT = mad(cbrtH, remT, mad(cbrtT, remH, cbrtT*remT));

    float z = mad(poly, bH, mad(poly, bT, bT)) + bH;
    z *= mf;
    z = as_float(as_uint(z) | xsign);
    c = axi >= EXPBITS_SP32 | axi == 0;
    z = c ? x : z;
    return z;

}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, cbrt, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double cbrt(double x) {

    int return_x = isinf(x) | isnan(x) | x == 0.0;
    ulong ux = as_ulong(fabs(x));
    int m = (as_int2(ux).hi >> 20) - 1023;

    // Treat subnormals
    ulong uxs = as_ulong(as_double(0x3ff0000000000000UL | ux) - 1.0);
    int ms = m + (as_int2(uxs).hi >> 20) - 1022;

    int c = m == -1023;
    ux = c ? uxs : ux;
    m = c ? ms : m;

    int mby3 = m / 3;
    int rem = m - 3*mby3;

    double mf = as_double((ulong)(mby3 + 1023) << 52);

    ux &= 0x000fffffffffffffUL;
    double Y = as_double(0x3fe0000000000000UL | ux);

    // nearest integer
    int index = as_int2(ux).hi >> 11;
    index = (0x100 | (index >> 1)) + (index & 1);
    double F = (double)index * 0x1.0p-9;

    double f = Y - F;
    double r = f * USE_TABLE(cbrt_inv_tbl, index-256);

    double z = r * fma(r,
                       fma(r,
                           fma(r,
                               fma(r,
                                   fma(r, -0x1.8090d6221a247p-6, 0x1.ee7113506ac13p-6),
                                   -0x1.511e8d2b3183bp-5),
                               0x1.f9add3c0ca458p-5),
                           -0x1.c71c71c71c71cp-4),
                       0x1.5555555555555p-2);

    double2 tv = USE_TABLE(cbrt_rem_tbl, rem+2);
    double Rem_h = tv.s0;
    double Rem_t = tv.s1;

    tv = USE_TABLE(cbrt_dbl_tbl, index-256);
    double F_h = tv.s0;
    double F_t = tv.s1;

    double b_h = F_h * Rem_h; 
    double b_t = fma(Rem_t, F_h, fma(F_t, Rem_h, F_t*Rem_t));

    double ans = fma(z, b_h, fma(z, b_t, b_t)) + b_h;
    ans = copysign(ans*mf, x);
    return return_x ? x : ans;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, cbrt, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_FP16(cbrt)

#endif
