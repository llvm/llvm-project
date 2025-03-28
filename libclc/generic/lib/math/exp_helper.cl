//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>

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
