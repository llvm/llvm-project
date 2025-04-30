/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#if defined(TARGET_LINUX_POWER)
#error "Source cannot be compiled for POWER architectures"
#include "xmm2altivec.h"
#else
#include <immintrin.h>
#endif
#include "rslog_defs.h"

extern "C" __m256 __rvs_log_fma3_256(__m256);

#ifndef PRECISION
#define PRECISION 0
#endif

__m256 __rvs_log_fma3_256(__m256 a) {


#if PRECISION == 0
    __m256 const A = _mm256_set1_ps(A0);
    __m256 const B = _mm256_set1_ps(B0);
    __m256 const C = _mm256_set1_ps(C0);
    __m256 const D = _mm256_set1_ps(D0);
    __m256 const E = _mm256_set1_ps(E0);
#endif
#if PRECISION == 1
    __m256 const A = _mm256_set1_ps(A1);
    __m256 const B = _mm256_set1_ps(B1);
    __m256 const C = _mm256_set1_ps(C1);
    __m256 const D = _mm256_set1_ps(D1);
    __m256 const E = _mm256_set1_ps(E1);
    __m256 const F = _mm256_set1_ps(F1);
    __m256 const G = _mm256_set1_ps(G1);
#endif
#if PRECISION == 2
    __m256 const A = _mm256_set1_ps(A2);
    __m256 const B = _mm256_set1_ps(B2);
    __m256 const C = _mm256_set1_ps(C2);
    __m256 const D = _mm256_set1_ps(D2);
    __m256 const E = _mm256_set1_ps(E2);
    __m256 const F = _mm256_set1_ps(F2);
    __m256 const G = _mm256_set1_ps(G2);
    __m256 const H = _mm256_set1_ps(H2);
#endif

    __m256 const PARTITION_CONST = _mm256_set1_ps(PARTITION_CONST_F);
    __m256 const TWO_TO_M126     = _mm256_set1_ps(TWO_TO_M126_F);
    __m256 const LN2             = _mm256_set1_ps(LN2_F);

    __m256i const N_INF      = _mm256_set1_epi32(0xff800000);
    __m256i const P_INF      = _mm256_set1_epi32(0x7f800000);
    __m256i const NINF2NAN   = _mm256_set1_epi32(CANONICAL_NAN_I ^ 0xff800000);
    __m256i const bit_mask2  = _mm256_set1_epi32(0x807fffff);
    __m256i const offset     = _mm256_set1_epi32(0x3f000000);
    __m256i const exp_offset = _mm256_set1_epi32(126);
    __m256  const ZERO       = _mm256_set1_ps(0.0f);
    __m256  const ONE        = _mm256_set1_ps(1.0f);

    __m256 e = (__m256)_mm256_srli_epi32((__m256i)a, 23);
           e = (__m256)_mm256_sub_epi32((__m256i)e, exp_offset);
           e = _mm256_cvtepi32_ps((__m256i)e);

    __m256i im = _mm256_and_si256(bit_mask2, (__m256i)a);
    __m256   m = (__m256)_mm256_add_epi32(im, offset);

    __m256 cmp = (__m256)_mm256_cmpgt_epi32((__m256i)PARTITION_CONST, (__m256i)m);

    __m256 fixe = _mm256_and_ps(cmp, LN2);
    e = _mm256_fmsub_ps(e, LN2, fixe);

    __m256i fixm = _mm256_and_si256((__m256i)cmp, _mm256_set1_epi32(0x00800000));
    m = (__m256)_mm256_add_epi32((__m256i)m, fixm);
    m = _mm256_sub_ps(m, ONE);

    __m256 t =                A;
    t = _mm256_fmadd_ps(t, m, B);
    t = _mm256_fmadd_ps(t, m, C);
    t = _mm256_fmadd_ps(t, m, D);
    t = _mm256_fmadd_ps(t, m, E);
#if PRECISION >= 1
    t = _mm256_fmadd_ps(t, m, F);
    t = _mm256_fmadd_ps(t, m, G);
#endif
#if PRECISION >= 2
    t = _mm256_fmadd_ps(t, m, H);
#endif
    t = _mm256_fmadd_ps(t, m, e);

    __m256 mask0, mask1;
    mask0 = _mm256_cmp_ps(a, TWO_TO_M126, _CMP_NGE_UQ);
    mask1 = (__m256)_mm256_cmpeq_epi32((__m256i)a, P_INF);

    if (__builtin_expect(_mm256_movemask_ps(_mm256_or_ps(mask0, mask1)) ,0))
    {
        // [0.0, FLT_MIN) u nan -> -inf
        __m256 spec = _mm256_and_ps(mask0, (__m256)N_INF);

        // (-oo, 0.0) -> nan
        __m256 neg = _mm256_cmp_ps(a, ZERO, _CMP_LT_OQ);
        neg = _mm256_and_ps(neg, (__m256)NINF2NAN);

        // nan -> nan, inf -> inf
        __m256 unord = _mm256_cmp_ps(a, (__m256)P_INF, _CMP_NLT_UQ);
        unord = _mm256_and_ps(unord, a);

        spec = _mm256_xor_ps(spec, neg);
        spec = _mm256_add_ps(spec, unord);
        t = _mm256_add_ps(t, spec);
   }

    return t;
}

