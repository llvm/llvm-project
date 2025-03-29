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
