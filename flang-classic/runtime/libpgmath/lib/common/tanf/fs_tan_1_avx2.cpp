
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#ifndef __TAN_F_SCALAR_H__
#define __TAN_F_SCALAR_H__


#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "common_tanf.h"

extern "C" float __attribute__ ((noinline)) __fs_tan_1_avx2(float const a);


/* Payne-Hanek style argument reduction. */
static float
reduction_slowpath(float const a, int32_t *h)
{
    uint2 m;
    uint32_t ia = float_as_int(a);
    uint32_t s = ia & 0x80000000;
    uint32_t result[7];
    uint32_t hi, lo;
    uint32_t e;
    int32_t idx;
    int32_t q;
    e = ((ia >> 23) & 0xff) - 127;
    ia = (ia << 8) | 0x80000000;

    /* compute x * 2/pi */
    idx = 4 - ((e >> 5) & 3);

    hi = 0;
    for (q = 0; q < 6; q++) {
        m = umad32wide(i2opi_f[q], ia, hi);
        lo = m.x;
        hi = m.y;
        result[q] = lo;
    }
    result[q] = hi;

    e = e & 31;
    /* shift result such that hi:lo<63:63> is the least significant
       integer bit, and hi:lo<62:0> are the fractional bits of the result
    */
    uint64_t p = ((uint64_t)result[idx + 2] << 32) | result[idx + 1];

    if (e) {
        q = 32 - e;
        p = (p << e) | (result[idx] >> q);
    }

    /* fraction */
    q = (result[idx + 2] << e) & 0x80000000;
    p &= 0x7fffffffffffffffULL;

    if (p & 0x4000000000000000ULL) {
        p |= 0x8000000000000000ULL;
        q ^= 0x80000000;
    }
    *h = q;

    double d = (double)(int64_t)p;
    d *= PI_2_M64;
    float r = (float)d;

    return int_as_float(float_as_int(r) ^ s);
}

float __attribute__ ((noinline))
__fs_tan_1_avx2(float x)
{

    float p, k, r, s, t;
    int h = 0;

    p   = int_as_float(float_as_int(x) & 0x7fffffff);
    if (float_as_int(p) > float_as_int(THRESHOLD_F)) {
        x = float_as_int(p) >= 0x7f800000 ? x * 0.0f : reduction_slowpath(x, &h);
    } else {
        k = FMAF(x, _2_OVER_PI_F, 12582912.0f);
        h = float_as_int(k) << 31;
        k -= 12582912.0f;
        x = FMAF(k, -PI_2_HI_F, x);
        x = FMAF(k, -PI_2_MI_F, x);
        x = FMAF(k, -PI_2_LO_F, x);
    }
    s = x * x;
    r = A_F;
    r = FMAF(r, s, B_F);
    r = FMAF(r, s, C_F);
    r = FMAF(r, s, D_F);
    r = FMAF(r, s, E_F);
    r = FMAF(r, s, F_F);
    t = s * x;
    r = FMAF(r, t, x);

    if (h) r = -1.0f / r;

    return r;
}

#endif // __TAN_F_SCALAR_H__

