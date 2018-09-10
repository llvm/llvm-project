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

#include "math.h"
#include "tables.h"
#include "../clcmacro.h"

_CLC_OVERLOAD _CLC_DEF float log1p(float x)
{
    float w = x;
    uint ux = as_uint(x);
    uint ax = ux & EXSIGNBIT_SP32;

    // |x| < 2^-4
    float u2 = MATH_DIVIDE(x, 2.0f + x);
    float u = u2 + u2;
    float v = u * u;
    // 2/(5 * 2^5), 2/(3 * 2^3)
    float zsmall = mad(-u2, x, mad(v, 0x1.99999ap-7f, 0x1.555556p-4f) * v * u) + x;

    // |x| >= 2^-4
    ux = as_uint(x + 1.0f);

    int m = (int)((ux >> EXPSHIFTBITS_SP32) & 0xff) - EXPBIAS_SP32;
    float mf = (float)m;
    uint indx = (ux & 0x007f0000) + ((ux & 0x00008000) << 1);
    float F = as_float(indx | 0x3f000000);

    // x > 2^24
    float fg24 = F - as_float(0x3f000000 | (ux & MANTBITS_SP32));

    // x <= 2^24
    uint xhi = ux & 0xffff8000;
    float xh = as_float(xhi);
    float xt = (1.0f - xh) + w;
    uint xnm = ((~(xhi & 0x7f800000)) - 0x00800000) & 0x7f800000;
    xt = xt * as_float(xnm) * 0.5f;
    float fl24 = F - as_float(0x3f000000 | (xhi & MANTBITS_SP32)) - xt;

    float f = mf > 24.0f ? fg24 : fl24;

    indx = indx >> 16;
    float r = f * USE_TABLE(log_inv_tbl, indx);

    // 1/3, 1/2
    float poly = mad(mad(r, 0x1.555556p-2f, 0x1.0p-1f), r*r, r);

    const float LOG2_HEAD = 0x1.62e000p-1f;   // 0.693115234
    const float LOG2_TAIL = 0x1.0bfbe8p-15f;  // 0.0000319461833

    float2 tv = USE_TABLE(loge_tbl, indx);
    float z1 = mad(mf, LOG2_HEAD, tv.s0);
    float z2 = mad(mf, LOG2_TAIL, -poly) + tv.s1;
    float z = z1 + z2;

    z = ax < 0x3d800000U ? zsmall : z;



    // Edge cases
    z = ax >= PINFBITPATT_SP32 ? w : z;
    z = w  < -1.0f ? as_float(QNANBITPATT_SP32) : z;
    z = w == -1.0f ? as_float(NINFBITPATT_SP32) : z;
        //fix subnormals
        z = ax  < 0x33800000 ? x : z;

    return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, log1p, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double log1p(double x)
{
    // Computes natural log(1+x). Algorithm based on:
    // Ping-Tak Peter Tang
    // "Table-driven implementation of the logarithm function in IEEE
    // floating-point arithmetic"
    // ACM Transactions on Mathematical Software (TOMS)
    // Volume 16, Issue 4 (December 1990)
    // Note that we use a lookup table of size 64 rather than 128,
    // and compensate by having extra terms in the minimax polynomial
    // for the kernel approximation.

    // Process Inside the threshold now
    ulong ux = as_ulong(1.0 + x);
    int xexp = ((as_int2(ux).hi >> 20) & 0x7ff) - EXPBIAS_DP64;
    double f = as_double(ONEEXPBITS_DP64 | (ux & MANTBITS_DP64));

    int j = as_int2(ux).hi >> 13;
    j = ((0x80 | (j & 0x7e)) >> 1) + (j & 0x1);
    double f1 = (double)j * 0x1.0p-6;
    j -= 64;

    double f2temp = f - f1;
    double m2 = as_double(convert_ulong(0x3ff - xexp) << EXPSHIFTBITS_DP64);
    double f2l = fma(m2, x, m2 - f1);
    double f2g = fma(m2, x, -f1) + m2;
    double f2 = xexp <= MANTLENGTH_DP64-1 ? f2l : f2g;
    f2 = (xexp <= -2) | (xexp >= MANTLENGTH_DP64+8) ? f2temp : f2;

    double2 tv = USE_TABLE(ln_tbl, j);
    double z1 = tv.s0;
    double q = tv.s1;

    double u = MATH_DIVIDE(f2, fma(0.5, f2, f1));
    double v = u * u;

    double poly = v * fma(v,
                          fma(v, 2.23219810758559851206e-03, 1.24999999978138668903e-02),
                          8.33333333333333593622e-02);

    // log2_lead and log2_tail sum to an extra-precise version of log(2)
    const double log2_lead = 6.93147122859954833984e-01; /* 0x3fe62e42e0000000 */
    const double log2_tail = 5.76999904754328540596e-08; /* 0x3e6efa39ef35793c */

    double z2 = q + fma(u, poly, u);
    double dxexp = (double)xexp;
    double r1 = fma(dxexp, log2_lead, z1);
    double r2 = fma(dxexp, log2_tail, z2);
    double result1 = r1 + r2;

    // Process Outside the threshold now
    double r = x;
    u = r / (2.0 + r);
    double correction = r * u;
    u = u + u;
    v = u * u;
    r1 = r;

    poly = fma(v,
               fma(v,
                   fma(v, 4.34887777707614552256e-04, 2.23213998791944806202e-03),
                   1.25000000037717509602e-02),
               8.33333333333317923934e-02);

    r2 = fma(u*v, poly, -correction);

    // The values exp(-1/16)-1 and exp(1/16)-1
    const double log1p_thresh1 = -0x1.f0540438fd5c3p-5;
    const double log1p_thresh2 =  0x1.082b577d34ed8p-4;
    double result2 = r1 + r2;
    result2 = x < log1p_thresh1 | x > log1p_thresh2 ? result1 : result2;

    result2 = isinf(x) ? x : result2;
    result2 = x < -1.0 ? as_double(QNANBITPATT_DP64) : result2;
    result2 = x == -1.0 ? as_double(NINFBITPATT_DP64) : result2;
    return result2;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, log1p, double);

#endif // cl_khr_fp64
