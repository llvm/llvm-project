//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sincos_helpers.h"
#include <clc/clc.h>
#include <clc/clc_convert.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_sincos_helpers.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>
#include <clc/relational/clc_select.h>

// FP32 and FP16 versions.
#define __CLC_BODY <cos.inc>
#include <clc/math/gentype.inc>

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double cos(double x) {
    x = fabs(x);

    double r, rr;
    int regn;

    if (x < 0x1.0p+47)
        __clc_remainder_piby2_medium(x, &r, &rr, &regn);
    else
        __clc_remainder_piby2_large(x, &r, &rr, &regn);

    double2 sc = __clc_sincos_piby4(r, rr);
    sc.lo = -sc.lo;

    int2 c = as_int2(regn & 1 ? sc.lo : sc.hi);
    c.hi ^= (regn > 1) << 31;

    return isnan(x) | isinf(x) ? as_double(QNANBITPATT_DP64) : as_double(c);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, cos, double);

#endif
