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

_CLC_OVERLOAD _CLC_DEF int ilogb(float x) {
    uint ux = as_uint(x);
    uint ax = ux & EXSIGNBIT_SP32;
    int rs = -118 - (int) clz(ux & MANTBITS_SP32);
    int r = (int) (ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
    r = ax < 0x00800000U ? rs : r;
    r = ax == 0 ? FP_ILOGB0 : r;

    // We could merge those 2 tests and have:
    //
    //    r = ax >= EXPBITS_SP32 ? 0x7fffffff : r
    //
    // since FP_ILOGBNAN is set to INT_MAX, but it's clearer this way and
    // FP_ILOGBNAN can change without requiring changes to ilogb() code.
    r = ax > EXPBITS_SP32 ? FP_ILOGBNAN : r;
    r = ax == EXPBITS_SP32 ? 0x7fffffff : r;
    return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, ilogb, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF int ilogb(double x) {
    ulong ux = as_ulong(x);
    ulong ax = ux & ~SIGNBIT_DP64;
    int r = (int) (ax >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;
    int rs = -1011 - (int) clz(ax & MANTBITS_DP64);
    r = ax < 0x0010000000000000UL ? rs : r;
    r = ax == 0UL ? FP_ILOGB0 : r;

    // We could merge those 2 tests and have:
    //
    //    r = ax >= 0x7ff0000000000000UL ? 0x7fffffff : r
    //
    // since FP_ILOGBNAN is set to INT_MAX, but it's clearer this way and
    // FP_ILOGBNAN can change without requiring changes to ilogb() code.
    r = ax > 0x7ff0000000000000UL ? FP_ILOGBNAN : r;
    r = ax == 0x7ff0000000000000UL ? 0x7fffffff : r;
    return r;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, ilogb, double);

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF int ilogb(half x) {
    return ilogb((float)x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, ilogb, half);

#endif
