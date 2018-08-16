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

_CLC_OVERLOAD _CLC_DEF float sinh(float x)
{
    // After dealing with special cases the computation is split into regions as follows.
    // abs(x) >= max_sinh_arg:
    // sinh(x) = sign(x)*Inf
    // abs(x) >= small_threshold:
    // sinh(x) = sign(x)*exp(abs(x))/2 computed using the splitexp and scaleDouble functions as for exp_amd().
    // abs(x) < small_threshold:
    // compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
    // sinh(x) is then sign(x)*z.

    const float max_sinh_arg = 0x1.65a9fap+6f;
    const float small_threshold = 0x1.0a2b24p+3f;

    uint ux = as_uint(x);
    uint aux = ux & EXSIGNBIT_SP32;
    uint xs = ux ^ aux;
    float y = as_float(aux);

    // We find the integer part y0 of y and the increment dy = y - y0. We then compute
    // z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)
    // where sinh(y0) and cosh(y0) are tabulated above.
    int ind = (int) y;
    ind = (uint)ind > 36U ? 0 : ind;

    float dy = y - ind;
    float dy2 = dy * dy;

    float sdy = mad(dy2,
                    mad(dy2,
                        mad(dy2,
                            mad(dy2,
                                mad(dy2,
                                    mad(dy2, 0.7746188980094184251527126e-12f, 0.160576793121939886190847e-9f),
                                    0.250521176994133472333666e-7f),
                                0.275573191913636406057211e-5f),
                            0.198412698413242405162014e-3f),
                         0.833333333333329931873097e-2f),
                    0.166666666666666667013899e0f);
    sdy = mad(sdy, dy*dy2, dy);

    float cdy = mad(dy2,
                    mad(dy2,
                        mad(dy2,
                            mad(dy2,
                                mad(dy2,
                                    mad(dy2, 0.1163921388172173692062032e-10f, 0.208744349831471353536305e-8f),
                                    0.275573350756016588011357e-6f),
                                0.248015872460622433115785e-4f),
                            0.138888888889814854814536e-2f),
                        0.416666666666660876512776e-1f),
                    0.500000000000000005911074e0f);
    cdy = mad(cdy, dy2, 1.0f);

    float2 tv = USE_TABLE(sinhcosh_tbl, ind);
    float z = mad(tv.s1, sdy, tv.s0 * cdy);
    z = as_float(xs | as_uint(z));

    // When y is large enough so that the negative exponential is negligible,
    // so sinh(y) is approximated by sign(x)*exp(y)/2.
    float t = exp(y - 0x1.62e500p-1f);
    float zsmall = mad(0x1.a0210ep-18f, t, t);
    zsmall = as_float(xs | as_uint(zsmall));
    z = y >= small_threshold ? zsmall : z;

    // Corner cases
    float zinf = as_float(PINFBITPATT_SP32 | xs);
    z = y >= max_sinh_arg ? zinf : z;
    z = aux > PINFBITPATT_SP32 | aux < 0x38800000U ? x : z;

    return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, sinh, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double sinh(double x)
{
    // After dealing with special cases the computation is split into
    // regions as follows:
    //
    // abs(x) >= max_sinh_arg:
    // sinh(x) = sign(x)*Inf
    //
    // abs(x) >= small_threshold:
    // sinh(x) = sign(x)*exp(abs(x))/2 computed using the
    // splitexp and scaleDouble functions as for exp_amd().
    //
    // abs(x) < small_threshold:
    // compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
    // sinh(x) is then sign(x)*z.

    const double max_sinh_arg = 7.10475860073943977113e+02; // 0x408633ce8fb9f87e

    // This is where exp(-x) is insignificant compared to exp(x) = ln(2^27)
    const double small_threshold = 0x1.2b708872320e2p+4;

    double y = fabs(x);

    // In this range we find the integer part y0 of y
    // and the increment dy = y - y0. We then compute
    // z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)
    // where sinh(y0) and cosh(y0) are obtained from tables

    int ind = min((int)y, 36);
    double dy = y - ind;
    double dy2 = dy * dy;

    double sdy = dy * dy2 *
	         fma(dy2,
		     fma(dy2,
			 fma(dy2,
			     fma(dy2,
				 fma(dy2,
				     fma(dy2, 0.7746188980094184251527126e-12, 0.160576793121939886190847e-9),
				     0.250521176994133472333666e-7),
				 0.275573191913636406057211e-5),
			     0.198412698413242405162014e-3),
			 0.833333333333329931873097e-2),
		     0.166666666666666667013899e0);

    double cdy = dy2 * fma(dy2,
	                   fma(dy2,
			       fma(dy2,
				   fma(dy2,
				       fma(dy2,
					   fma(dy2, 0.1163921388172173692062032e-10, 0.208744349831471353536305e-8),
					   0.275573350756016588011357e-6),
				       0.248015872460622433115785e-4),
				   0.138888888889814854814536e-2),
			       0.416666666666660876512776e-1),
			   0.500000000000000005911074e0);

    // At this point sinh(dy) is approximated by dy + sdy.
    // Shift some significant bits from dy to sdy.
    double sdy1 = as_double(as_ulong(dy) & 0xfffffffff8000000UL);
    double sdy2 = sdy + (dy - sdy1);

    double2 tv = USE_TABLE(cosh_tbl, ind);
    double cl = tv.s0;
    double ct = tv.s1;
    tv = USE_TABLE(sinh_tbl, ind);
    double sl = tv.s0;
    double st = tv.s1;

    double z = fma(cl, sdy1, fma(sl, cdy, fma(cl, sdy2, fma(ct, sdy1, fma(st, cdy, ct*sdy2)) + st))) + sl;

    // Other cases
    z = (y < 0x1.0p-28) | isnan(x) | isinf(x) ? y : z;

    double t = exp(y - 0x1.62e42fefa3800p-1);
    t = fma(t, -0x1.ef35793c76641p-45, t);
    z = y >= small_threshold ? t : z;
    z = y >= max_sinh_arg ? as_double(PINFBITPATT_DP64) : z;

    return copysign(z, x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, sinh, double)

#endif
