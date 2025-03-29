//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float exp2(float x) {

    // Reduce x
    const float ln2HI = 0x1.62e300p-1f;
    const float ln2LO = 0x1.2fefa2p-17f;

    float t = rint(x);
    int p = (int)t;
    float tt = x - t;
    float hi = tt * ln2HI;
    float lo = tt * ln2LO;

    // Evaluate poly
    t = hi + lo;
    tt  = t*t;
    float v = mad(tt,
                  -mad(tt,
		       mad(tt,
		           mad(tt,
			       mad(tt, 0x1.637698p-25f, -0x1.bbd41cp-20f),
                               0x1.1566aap-14f),
                           -0x1.6c16c2p-9f),
                       0x1.555556p-3f),
                  t);

    float y = 1.0f - (((-lo) - MATH_DIVIDE(t * v, 2.0f - v)) - hi);

    // Scale by 2^p
    float r =  as_float(as_int(y) + (p << 23));

    const float ulim =  128.0f;
    const float llim = -126.0f;

    r = x < llim ? 0.0f : r;
    r = x < ulim ? r : as_float(0x7f800000);
    return isnan(x) ? x : r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, exp2, float)

#ifdef cl_khr_fp64

#include "exp_helper.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double exp2(double x) {
    const double R_LN2 = 0x1.62e42fefa39efp-1; // ln(2)
    const double R_1_BY_64 = 1.0 / 64.0;

    int n = convert_int(x * 64.0);
    double r = R_LN2 * fma(-R_1_BY_64, (double)n, x); 
    return __clc_exp_helper(x, -1074.0, 1024.0, r, n);
}


_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, exp2, double)

#endif
