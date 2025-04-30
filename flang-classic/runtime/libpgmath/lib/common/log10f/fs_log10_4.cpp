
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

#define log10_vec128 JOIN2(__fs_log10_4_,_CPU)

extern "C" __m128 log10_vec128(__m128);


/*  #include "../scalar/log10_scalar.h"  */

__m128 __attribute__ ((noinline)) log10_vec128(__m128 a_input)
{
    __m128 a, m, e, b, t;
    __m128i idx, cmp, mp, ep;

#ifdef __AVX512VL__
    a = a_input;
    m = _mm_getmant_ps(a, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
    e = _mm_getexp_ps(a);
    b = _mm_getexp_ps(m);
    e = _mm_sub_ps(e, b);

    e = _mm_mul_ps(e, *(__m128*)LOG10_2_F);

    idx = _mm_srli_epi32((__m128i)m, 19);
    m = _mm_sub_ps(m, *(__m128*)ONE_F);

    idx = _mm_and_si128(idx, _mm_set1_epi32(0xf));
    __m128 c0 = _mm_i32gather_ps(coeffs0, idx, 4);
    __m128 c1 = _mm_i32gather_ps(coeffs1, idx, 4);
    __m128 c2 = _mm_i32gather_ps(coeffs2, idx, 4);
    __m128 c3 = _mm_i32gather_ps(coeffs3, idx, 4);

    t = c0;
    t = _mm_fmadd_ps(t, m, c1);
    t = _mm_fmadd_ps(t, m, c2);
    t = _mm_fmadd_ps(t, m, c3);
    t = _mm_fmadd_ps(t, m, e);
#else
    __m128i u;
    m = (__m128)_mm_sub_epi32((__m128i)a_input, *(__m128i*)MAGIC_F);
    e = (__m128)_mm_srai_epi32((__m128i)m, 23);
    m = _mm_and_ps(m, *(__m128*)MANTISSA_MASK);
    m = (__m128)_mm_add_epi32((__m128i)m, *(__m128i*)MAGIC_F);

    e = _mm_cvtepi32_ps((__m128i)e);
    e = _mm_mul_ps(e, *(__m128*)LOG10_2_F);

    idx = _mm_srli_epi32((__m128i)m, 19);
    m = _mm_sub_ps(m, *(__m128*)ONE_F);

    idx = _mm_and_si128(idx, _mm_set1_epi32(0xf));
    __m128 c0 = _mm_i32gather_ps(coeffs0, idx, 4);
    __m128 c1 = _mm_i32gather_ps(coeffs1, idx, 4);
    __m128 c2 = _mm_i32gather_ps(coeffs2, idx, 4);
    __m128 c3 = _mm_i32gather_ps(coeffs3, idx, 4);

    t = c0;
    t = _mm_fmadd_ps(t, m, c1);
    t = _mm_fmadd_ps(t, m, c2);
    t = _mm_fmadd_ps(t, m, c3);
    t = _mm_fmadd_ps(t, m, e);

    u = _mm_add_epi32((__m128i)a_input, _mm_set1_epi32(0x800000));
    u = _mm_cmpgt_epi32(_mm_set1_epi32(0x1000000), u);
    if (__builtin_expect(!_mm_testz_si128(u, u), 0)) {
        __m128i inf_mask = _mm_cmpeq_epi32((__m128i)a_input, _mm_set1_epi32(0x7f800000));
        __m128i den_mask = _mm_cmpgt_epi32(_mm_set1_epi32(0x800000), (__m128i)a_input);
        __m128i neg_mask = _mm_cmpgt_epi32(_mm_set1_epi32(0), (__m128i)a_input);
        __m128i zer_mask = (__m128i)_mm_cmp_ps(_mm_set1_ps(0.0f), a_input, _CMP_EQ_OQ);
        __m128i nan_mask = (__m128i)_mm_cmp_ps(a_input, a_input, _CMP_UNORD_Q);

        __m128 inf_out = _mm_set1_ps(PINF);
        __m128 neg_out = _mm_set1_ps(CANONICAL_NAN);
        __m128 zer_out = _mm_set1_ps(NINF);
        __m128 nan_out = _mm_add_ps(a_input, a_input);

        __m128 a2p24 = _mm_mul_ps(a_input, _mm_set1_ps(TWO_TO_24_F));
        m = (__m128)_mm_sub_epi32((__m128i)a2p24, *(__m128i*)MAGIC_F);
        e = (__m128)_mm_sub_epi32(_mm_srai_epi32((__m128i)m, 23), _mm_set1_epi32(24));
        m = _mm_and_ps(m, *(__m128*)MANTISSA_MASK);
        m = (__m128)_mm_add_epi32((__m128i)m, *(__m128i*)MAGIC_F);

        e = _mm_cvtepi32_ps((__m128i)e);
        e = _mm_mul_ps(e, *(__m128*)LOG10_2_F);

        idx = _mm_srli_epi32((__m128i)m, 19);
        m = _mm_sub_ps(m, *(__m128*)ONE_F);

        idx = _mm_and_si128(idx, _mm_set1_epi32(0xf));
        c0 = _mm_i32gather_ps(coeffs0, idx, 4);
        c1 = _mm_i32gather_ps(coeffs1, idx, 4);
        c2 = _mm_i32gather_ps(coeffs2, idx, 4);
        c3 = _mm_i32gather_ps(coeffs3, idx, 4);

        __m128 den_out = c0;
        den_out = _mm_fmadd_ps(den_out, m, c1);
        den_out = _mm_fmadd_ps(den_out, m, c2);
        den_out = _mm_fmadd_ps(den_out, m, c3);
        den_out = _mm_fmadd_ps(den_out, m, e);

        t = _mm_blendv_ps(t, inf_out, (__m128)inf_mask);
        t = _mm_blendv_ps(t, den_out, (__m128)den_mask);
        t = _mm_blendv_ps(t, neg_out, (__m128)neg_mask);
        t = _mm_blendv_ps(t, zer_out, (__m128)zer_mask);
        t = _mm_blendv_ps(t, nan_out, (__m128)nan_mask);
    }
#endif

    return t;
}

