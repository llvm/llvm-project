/*
 * Copyright (c) 2015 Advanced Micro Devices, Inc.
 * Copyright (c) 2016 Aaron Watry
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
#include "../clcmacro.h"
#include "math.h"

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
