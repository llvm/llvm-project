/*
 * Copyright (c) 2014, 2015 Advanced Micro Devices, Inc.
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

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF double __clc_exp_helper(double x, double x_min, double x_max, double r, int n) {

    int j = n & 0x3f;
    int m = n >> 6;

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

    z2 = x > x_max ? as_double(PINFBITPATT_DP64) : z2;
    z2 = x < x_min ? 0.0 : z2;

    return z2;
}

#endif // cl_khr_fp64
