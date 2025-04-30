/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#if defined(TARGET_LINUX_POWER)
#include "xmm2altivec.h"
#elif defined(TARGET_ARM64)
#include "arm64intrin.h"
#else
#include <immintrin.h>
#endif
#include "rslog_defs.h"

extern "C" __m128 __rvs_log_fma3(__m128);

#ifndef PRECISION
#define PRECISION 0
#endif

__m128 __rvs_log_fma3(__m128 a) {

#if PRECISION == 0
    __m128 const A = _mm_set1_ps(A0);
    __m128 const B = _mm_set1_ps(B0);
    __m128 const C = _mm_set1_ps(C0);
    __m128 const D = _mm_set1_ps(D0);
    __m128 const E = _mm_set1_ps(E0);
#endif
#if PRECISION == 1
    __m128 const A = _mm_set1_ps(A1);
    __m128 const B = _mm_set1_ps(B1);
    __m128 const C = _mm_set1_ps(C1);
    __m128 const D = _mm_set1_ps(D1);
    __m128 const E = _mm_set1_ps(E1);
    __m128 const F = _mm_set1_ps(F1);
    __m128 const G = _mm_set1_ps(G1);
#endif
#if PRECISION == 2
    __m128 const A = _mm_set1_ps(A2);
    __m128 const B = _mm_set1_ps(B2);
    __m128 const C = _mm_set1_ps(C2);
    __m128 const D = _mm_set1_ps(D2);
    __m128 const E = _mm_set1_ps(E2);
    __m128 const F = _mm_set1_ps(F2);
    __m128 const G = _mm_set1_ps(G2);
    __m128 const H = _mm_set1_ps(H2);
#endif

    __m128 const PARTITION_CONST = _mm_set1_ps(PARTITION_CONST_F);
    __m128 const TWO_TO_M126     = _mm_set1_ps(TWO_TO_M126_F);
    __m128 const LN2             = _mm_set1_ps(LN2_F);

    __m128i const N_INF      = _mm_set1_epi32(0xff800000);
    __m128i const P_INF      = _mm_set1_epi32(0x7f800000);
    __m128i const NINF2NAN   = _mm_set1_epi32(CANONICAL_NAN_I ^ 0xff800000);
    __m128i const bit_mask2  = _mm_set1_epi32(0x807fffff);
    __m128i const offset     = _mm_set1_epi32(0x3f000000);
    __m128i const exp_offset = _mm_set1_epi32(126);
    __m128  const ZERO       = _mm_set1_ps(0.0f);
    __m128  const ONE        = _mm_set1_ps(1.0f);

    __m128 e = (__m128)_mm_srli_epi32((__m128i)a, 23);
           e = (__m128)_mm_sub_epi32((__m128i)e, exp_offset);
           e = _mm_cvtepi32_ps((__m128i)e);

    __m128i im = _mm_and_si128(bit_mask2, (__m128i)a);
    __m128   m = (__m128)_mm_add_epi32(im, offset);

    __m128 cmp = (__m128)_mm_cmpgt_epi32((__m128i)PARTITION_CONST, (__m128i)m);

    __m128 fixe = _mm_and_ps(cmp, LN2);
    e = _mm_fmsub_ps(e, LN2, fixe);

    __m128i fixm = _mm_and_si128((__m128i)cmp, _mm_set1_epi32(0x00800000));
    m = (__m128)_mm_add_epi32((__m128i)m, fixm);
    m = _mm_sub_ps(m, ONE);

    __m128 t =                A;
    t = _mm_fmadd_ps(t, m, B);
    t = _mm_fmadd_ps(t, m, C);
    t = _mm_fmadd_ps(t, m, D);
    t = _mm_fmadd_ps(t, m, E);
#if PRECISION >= 1
    t = _mm_fmadd_ps(t, m, F);
    t = _mm_fmadd_ps(t, m, G);
#endif
#if PRECISION >= 2
    t = _mm_fmadd_ps(t, m, H);
#endif
    t = _mm_fmadd_ps(t, m, e);

    __m128 mask0, mask1;
    mask0 = _mm_cmp_ps(a, TWO_TO_M126, _CMP_NGE_UQ);
    mask1 = (__m128)_mm_cmpeq_epi32((__m128i)a, P_INF);

#if defined(TARGET_LINUX_POWER)
    if (__builtin_expect(_vec_any_nz((__m128i)_mm_or_ps(mask0, mask1)) ,0))
#else
    if (__builtin_expect(_mm_movemask_ps(_mm_or_ps(mask0, mask1)) ,0))
#endif
    {
        // [0.0, FLT_MIN) u nan -> -inf
        __m128 spec = _mm_and_ps(mask0, (__m128)N_INF);

        // (-oo, 0.0) -> nan
        __m128 neg = _mm_cmp_ps(a, ZERO, _CMP_LT_OQ);
        neg = _mm_and_ps(neg, (__m128)NINF2NAN);

        // nan -> nan, inf -> inf
        __m128 unord = _mm_cmp_ps(a, (__m128)P_INF, _CMP_NLT_UQ);
        unord = _mm_and_ps(unord, a);

        spec = _mm_xor_ps(spec, neg);
        spec = _mm_add_ps(spec, unord);
        t = _mm_add_ps(t, spec);
   }

    return t;
}
