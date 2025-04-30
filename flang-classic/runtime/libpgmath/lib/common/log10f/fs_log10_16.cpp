
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

#define log10_vec512 JOIN2(__fs_log10_16_,_CPU)

extern "C" __m512 log10_vec512(__m512);

__m512 __attribute__ ((noinline)) log10_vec512(__m512 a)
{
    __m512 m, e, b, t;
    __m512i idx;

#ifdef __AVX512F__
    m = _mm512_getmant_ps(a, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
    e = _mm512_getexp_ps(a);
    b = _mm512_getexp_ps(m);
    e = _mm512_sub_ps(e, b);
    e = _mm512_mul_ps(e, *(__m512*)LOG10_2_F);

    idx = _mm512_srli_epi32((__m512i)m, 19);
    m = _mm512_sub_ps(m, *(__m512*)ONE_F);

    __m512 c0 = _mm512_permutexvar_ps(idx, *(__m512*)coeffs0);
    __m512 c1 = _mm512_permutexvar_ps(idx, *(__m512*)coeffs1);
    __m512 c2 = _mm512_permutexvar_ps(idx, *(__m512*)coeffs2);
    __m512 c3 = _mm512_permutexvar_ps(idx, *(__m512*)coeffs3);

    t = c0;
    t = _mm512_fmadd_ps(t, m, c1);
    t = _mm512_fmadd_ps(t, m, c2);
    t = _mm512_fmadd_ps(t, m, c3);
    t = _mm512_fmadd_ps(t, m, e);
#else
#warning NO AVX512!
#endif

    return t;
}
