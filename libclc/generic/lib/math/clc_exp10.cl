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

#include "config.h"
#include "math.h"
#include "tables.h"
#include "../clcmacro.h"

//    Algorithm:
//
//    e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
//
//    x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//    n = 64*m + j,   0 <= j < 64
//
//    e^x = 2^((64*m + j + f)/64)
//        = (2^m) * (2^(j/64)) * 2^(f/64)
//        = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
//
//    f = x*(64/ln(2)) - n
//    r = f*(ln(2)/64) = x - n*(ln(2)/64)
//
//    e^x = (2^m) * (2^(j/64)) * e^r
//
//    (2^(j/64)) is precomputed
//
//    e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//    e^r = 1 + q
//
//    q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//
//    e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) )

_CLC_DEF _CLC_OVERLOAD float __clc_exp10(float x)
{
    const float X_MAX =  0x1.344134p+5f; // 128*log2/log10 : 38.53183944498959
    const float X_MIN = -0x1.66d3e8p+5f; // -149*log2/log10 : -44.8534693539332

    const float R_64_BY_LOG10_2 = 0x1.a934f0p+7f; // 64*log10/log2 : 212.6033980727912
    const float R_LOG10_2_BY_64_LD = 0x1.340000p-8f; // log2/(64 * log10) lead : 0.004699707
    const float R_LOG10_2_BY_64_TL = 0x1.04d426p-18f; // log2/(64 * log10) tail : 0.00000388665057
    const float R_LN10 = 0x1.26bb1cp+1f;

    int return_nan = isnan(x);
    int return_inf = x > X_MAX;
    int return_zero = x < X_MIN;

    int n = convert_int(x * R_64_BY_LOG10_2);

    float fn = (float)n;
    int j = n & 0x3f;
    int m = n >> 6;
    int m2 = m << EXPSHIFTBITS_SP32;
    float r;

    r = R_LN10 * mad(fn, -R_LOG10_2_BY_64_TL, mad(fn, -R_LOG10_2_BY_64_LD, x));

    // Truncated Taylor series for e^r
    float z2 = mad(mad(mad(r, 0x1.555556p-5f, 0x1.555556p-3f), r, 0x1.000000p-1f), r*r, r);

    float two_to_jby64 = USE_TABLE(exp_tbl, j);
    z2 = mad(two_to_jby64, z2, two_to_jby64);

    float z2s = z2 * as_float(0x1 << (m + 149));
    float z2n = as_float(as_int(z2) + m2);
    z2 = m <= -126 ? z2s : z2n;


    z2 = return_inf ? as_float(PINFBITPATT_SP32) : z2;
    z2 = return_zero ? 0.0f : z2;
    z2 = return_nan ? x : z2;
    return z2;
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_exp10, float)

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_exp10(double x)
{
    const double X_MAX = 0x1.34413509f79ffp+8; // 1024*ln(2)/ln(10)
    const double X_MIN = -0x1.434e6420f4374p+8; // -1074*ln(2)/ln(10)

    const double R_64_BY_LOG10_2 = 0x1.a934f0979a371p+7; // 64*ln(10)/ln(2)
    const double R_LOG10_2_BY_64_LD = 0x1.3441350000000p-8; // head ln(2)/(64*ln(10))
    const double R_LOG10_2_BY_64_TL = 0x1.3ef3fde623e25p-37; // tail ln(2)/(64*ln(10))
    const double R_LN10 = 0x1.26bb1bbb55516p+1; // ln(10)

    int n = convert_int(x * R_64_BY_LOG10_2);

    double dn = (double)n;

    int j = n & 0x3f;
    int m = n >> 6;

    double r = R_LN10 * fma(-R_LOG10_2_BY_64_TL, dn, fma(-R_LOG10_2_BY_64_LD, dn, x));

    // 6 term tail of Taylor expansion of e^r
    double z2 = r * fma(r,
	                fma(r,
		            fma(r,
			        fma(r,
			            fma(r, 0x1.6c16c16c16c17p-10, 0x1.1111111111111p-7),
			            0x1.5555555555555p-5),
			        0x1.5555555555555p-3),
		            0x1.0000000000000p-1),
		        1.0);

    double2 tv = USE_TABLE(two_to_jby64_ep_tbl, j);
    z2 = fma(tv.s0 + tv.s1, z2, tv.s1) + tv.s0;

    int small_value = (m < -1022) || ((m == -1022) && (z2 < 1.0));

	int n1 = m >> 2;
	int n2 = m-n1;
	double z3= z2 * as_double(((long)n1 + 1023) << 52);
	z3 *= as_double(((long)n2 + 1023) << 52);

    z2 = ldexp(z2, m);
    z2 = small_value ? z3: z2;

    z2 = isnan(x) ? x : z2;

    z2 = x > X_MAX ? as_double(PINFBITPATT_DP64) : z2;
    z2 = x < X_MIN ? 0.0 : z2;

    return z2;
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_exp10, double)
#endif
