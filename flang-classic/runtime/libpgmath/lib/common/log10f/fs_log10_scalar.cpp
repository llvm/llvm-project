
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

extern "C" float log10_scalar(float);


float __attribute__ ((noinline)) log10_scalar(float a_input)
{
    float res;
    __m128 a, m, e, b, t, c0, c1, c2, c3;
    int i, mu, eu;

#ifdef __AVX512F__
    a = _mm_set_ss(a_input);
    m = _mm_getmant_ss(a, a, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
    e = _mm_getexp_ss(a, a);
    b = _mm_getexp_ss(m, m);
    e = _mm_sub_ss(e, b);
#else
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
        mu -= float_as_int(MAGIC_F[0]);
        eu = (mu >> 23) - exp_offset;
        mu &= MANTISSA_MASK[0];
        mu += float_as_int(MAGIC_F[0]);
        m = _mm_set_ss(int_as_float(mu));
        e = _mm_set_ss((float)eu);
        goto core;
    }
    mu = float_as_int(a_input);
    mu -= float_as_int(MAGIC_F[0]);
    eu = (mu >> 23);
    mu &= MANTISSA_MASK[0];
    mu += float_as_int(MAGIC_F[0]);
    m = _mm_set_ss(int_as_float(mu));
    e = _mm_set_ss((float)eu);
#endif
core:
    e = _mm_mul_ss(e, *(__m128*)LOG10_2_F);
    i = _mm_cvtsi128_si32((__m128i)m);

    c0 = _mm_load_ps(coeffs + ((i >> 17) & 0x3c));
    c1 = _mm_permute_ps(c0, 1);
    c2 = _mm_permute_ps(c0, 2);
    c3 = _mm_permute_ps(c0, 3);

    m = _mm_sub_ss(m, _mm_set_ss(1.0f));
    t = c0;
    t = _mm_fmadd_ss(t, m, c1);
    t = _mm_fmadd_ss(t, m, c2);
    t = _mm_fmadd_ss(t, m, c3);
    t = _mm_fmadd_ss(t, m, e);
    res = _mm_cvtss_f32(t);

    return res;
}

