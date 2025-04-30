
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <immintrin.h>
#include <common.h>

#if !(defined _CPU)
#error: please define _CPU - specific suffix to a function name
#endif

#define _JOIN2(a,b) a##b
#define JOIN2(a,b) _JOIN2(a,b)

#define log10_scalar JOIN2(__fs_log10_1_,_CPU)
#define FMAF __builtin_fmaf

extern "C" float log10_scalar(float);


float __attribute__ ((noinline)) log10_scalar(float a_input)
{
    float a, m, e, b, t;
    int mu, eu;

    unsigned u = float_as_int(a_input);
    u -= 0x800000;
    if (__builtin_expect(u >= 0x7f000000, 0)) {
        int exp_offset = 0;
        if (a_input != a_input) return a_input + a_input; // NaN
        if (a_input < 0.0f) return CANONICAL_NAN; // negative
        if (a_input == 0.0f) return NINF; // zero
        if (a_input == PINF) return PINF; // +infinity
        a_input *= TWO_TO_24_F; // denormals
        exp_offset += 24;
        mu = float_as_int(a_input);
        mu -= float_as_int(MAGIC_F_LEGACY[0]);
        eu = (mu >> 23) - exp_offset;
        mu &= MANTISSA_MASK[0];
        mu += float_as_int(MAGIC_F_LEGACY[0]);
        m = int_as_float(mu);
        e = (float)eu;
        goto core;
    }
    mu = float_as_int(a_input);
    mu -= float_as_int(MAGIC_F_LEGACY[0]);
    eu = mu >> 23;
    mu &= MANTISSA_MASK[0];
    mu += float_as_int(MAGIC_F_LEGACY[0]);
    m = int_as_float(mu);
    e = (float)eu;
core:
    e = e * LOG10_2_F[0];

    m = m - 1.0f;

    t = c0[0];
    t = FMAF(t, m, c1[0]);
    t = FMAF(t, m, c2[0]);
    t = FMAF(t, m, c3[0]);
    t = FMAF(t, m, c4[0]);
    t = FMAF(t, m, c5[0]);
    t = FMAF(t, m, c6[0]);
    t = FMAF(t, m, c7[0]);
    t = FMAF(t, m, c8[0]);
    t = FMAF(t, m, e);

    return t;
}

