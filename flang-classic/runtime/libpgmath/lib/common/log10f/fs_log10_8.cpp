
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

#define log10_vec256 JOIN2(__fs_log10_8_,_CPU)

extern "C" __m256 log10_vec256(__m256);


__m256 __attribute__ ((noinline)) log10_vec256(__m256 a_input)
{
    __m256 a, m, e, b, t;
    __m256i idx, cmp, mp, ep;

#ifdef __AVX512VL__
    a = a_input;
    m = _mm256_getmant_ps(a, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
    e = _mm256_getexp_ps(a);
    b = _mm256_getexp_ps(m);
    e = _mm256_sub_ps(e, b);

    e = _mm256_mul_ps(e, *(__m256*)LOG10_2_F);

    idx = _mm256_srli_epi32((__m256i)m, 19);
    m = _mm256_sub_ps(m, *(__m256*)ONE_F);

    idx = _mm256_and_si256(idx, _mm256_set1_epi32(0xf));
    __m256 c0 = _mm256_i32gather_ps(coeffs0, idx, 4);
    __m256 c1 = _mm256_i32gather_ps(coeffs1, idx, 4);
    __m256 c2 = _mm256_i32gather_ps(coeffs2, idx, 4);
    __m256 c3 = _mm256_i32gather_ps(coeffs3, idx, 4);

    t = c0;
    t = _mm256_fmadd_ps(t, m, c1);
    t = _mm256_fmadd_ps(t, m, c2);
    t = _mm256_fmadd_ps(t, m, c3);
    t = _mm256_fmadd_ps(t, m, e);
#else
    __m256i u;
    m = (__m256)_mm256_sub_epi32((__m256i)a_input, *(__m256i*)MAGIC_F);
    e = (__m256)_mm256_srai_epi32((__m256i)m, 23);
    m = _mm256_and_ps(m, *(__m256*)MANTISSA_MASK);
    m = (__m256)_mm256_add_epi32((__m256i)m, *(__m256i*)MAGIC_F);

    e = _mm256_cvtepi32_ps((__m256i)e);
    e = _mm256_mul_ps(e, *(__m256*)LOG10_2_F);

    idx = _mm256_srli_epi32((__m256i)m, 19);
    m = _mm256_sub_ps(m, *(__m256*)ONE_F);

    idx = _mm256_and_si256(idx, _mm256_set1_epi32(0xf));
    __m256 c0 = _mm256_i32gather_ps(coeffs0, idx, 4);
    __m256 c1 = _mm256_i32gather_ps(coeffs1, idx, 4);
    __m256 c2 = _mm256_i32gather_ps(coeffs2, idx, 4);
    __m256 c3 = _mm256_i32gather_ps(coeffs3, idx, 4);

    t = c0;
    t = _mm256_fmadd_ps(t, m, c1);
    t = _mm256_fmadd_ps(t, m, c2);
    t = _mm256_fmadd_ps(t, m, c3);
    t = _mm256_fmadd_ps(t, m, e);

    u = _mm256_add_epi32((__m256i)a_input, _mm256_set1_epi32(0x800000));
    u = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x1000000), u);
    if (__builtin_expect(!_mm256_testz_si256(u, u), 0)) {
        __m256i inf_mask = _mm256_cmpeq_epi32((__m256i)a_input, _mm256_set1_epi32(0x7f800000));
        __m256i den_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), (__m256i)a_input);
        __m256i neg_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(0), (__m256i)a_input);
        __m256i zer_mask = (__m256i)_mm256_cmp_ps(_mm256_set1_ps(0.0f), a_input, _CMP_EQ_OQ);
        __m256i nan_mask = (__m256i)_mm256_cmp_ps(a_input, a_input, _CMP_UNORD_Q);

        __m256 inf_out = _mm256_set1_ps(PINF);
        __m256 neg_out = _mm256_set1_ps(CANONICAL_NAN);
        __m256 zer_out = _mm256_set1_ps(NINF);
        __m256 nan_out = _mm256_add_ps(a_input, a_input);

        __m256 a2p24 = _mm256_mul_ps(a_input, _mm256_set1_ps(TWO_TO_24_F));
        m = (__m256)_mm256_sub_epi32((__m256i)a2p24, *(__m256i*)MAGIC_F);
        e = (__m256)_mm256_sub_epi32(_mm256_srai_epi32((__m256i)m, 23), _mm256_set1_epi32(24));
        m = _mm256_and_ps(m, *(__m256*)MANTISSA_MASK);
        m = (__m256)_mm256_add_epi32((__m256i)m, *(__m256i*)MAGIC_F);

        e = _mm256_cvtepi32_ps((__m256i)e);
        e = _mm256_mul_ps(e, *(__m256*)LOG10_2_F);

        idx = _mm256_srli_epi32((__m256i)m, 19);
        m = _mm256_sub_ps(m, *(__m256*)ONE_F);

        idx = _mm256_and_si256(idx, _mm256_set1_epi32(0xf));
        c0 = _mm256_i32gather_ps(coeffs0, idx, 4);
        c1 = _mm256_i32gather_ps(coeffs1, idx, 4);
        c2 = _mm256_i32gather_ps(coeffs2, idx, 4);
        c3 = _mm256_i32gather_ps(coeffs3, idx, 4);

        __m256 den_out = c0;
        den_out = _mm256_fmadd_ps(den_out, m, c1);
        den_out = _mm256_fmadd_ps(den_out, m, c2);
        den_out = _mm256_fmadd_ps(den_out, m, c3);
        den_out = _mm256_fmadd_ps(den_out, m, e);

        t = _mm256_blendv_ps(t, inf_out, (__m256)inf_mask);
        t = _mm256_blendv_ps(t, den_out, (__m256)den_mask);
        t = _mm256_blendv_ps(t, neg_out, (__m256)neg_mask);
        t = _mm256_blendv_ps(t, zer_out, (__m256)zer_mask);
        t = _mm256_blendv_ps(t, nan_out, (__m256)nan_mask);
    }
#endif

    return t;
}
