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
#include "mth_avx512helper.h"
#endif
#include "rslog_defs.h"

extern "C" __m512 FCN_AVX512(__rvs_log_fma3)(__m512);

#ifndef PRECISION
#define PRECISION 0
#endif

__m512 FCN_AVX512(__rvs_log_fma3)(__m512 a) {


#if PRECISION == 0
    __m512 const A = _mm512_set1_ps(A0);
    __m512 const B = _mm512_set1_ps(B0);
    __m512 const C = _mm512_set1_ps(C0);
    __m512 const D = _mm512_set1_ps(D0);
    __m512 const E = _mm512_set1_ps(E0);
#endif
#if PRECISION == 1
    __m512 const A = _mm512_set1_ps(A1);
    __m512 const B = _mm512_set1_ps(B1);
    __m512 const C = _mm512_set1_ps(C1);
    __m512 const D = _mm512_set1_ps(D1);
    __m512 const E = _mm512_set1_ps(E1);
    __m512 const F = _mm512_set1_ps(F1);
    __m512 const G = _mm512_set1_ps(G1);
#endif
#if PRECISION == 2
    __m512 const A = _mm512_set1_ps(A2);
    __m512 const B = _mm512_set1_ps(B2);
    __m512 const C = _mm512_set1_ps(C2);
    __m512 const D = _mm512_set1_ps(D2);
    __m512 const E = _mm512_set1_ps(E2);
    __m512 const F = _mm512_set1_ps(F2);
    __m512 const G = _mm512_set1_ps(G2);
    __m512 const H = _mm512_set1_ps(H2);
#endif

    __m512 const PARTITION_CONST = _mm512_set1_ps(PARTITION_CONST_F);
    __m512 const TWO_TO_M126     = _mm512_set1_ps(TWO_TO_M126_F);
    __m512 const LN2             = _mm512_set1_ps(LN2_F);

    __m512i const N_INF      = _mm512_set1_epi32(0xff800000);
    __m512i const P_INF      = _mm512_set1_epi32(0x7f800000);
    __m512i const NINF2NAN   = _mm512_set1_epi32(CANONICAL_NAN_I ^ 0xff800000);
    __m512i const bit_mask2  = _mm512_set1_epi32(0x807fffff);
    __m512i const offset     = _mm512_set1_epi32(0x3f000000);
    __m512i const exp_offset = _mm512_set1_epi32(126);
    __m512  const ZERO       = _mm512_set1_ps(0.0f);
    __m512  const ONE        = _mm512_set1_ps(1.0f);

    __m512 e = (__m512)_mm512_srli_epi32((__m512i)a, 23);
           e = (__m512)_mm512_sub_epi32((__m512i)e, exp_offset);
           e = _mm512_cvtepi32_ps((__m512i)e);

    __m512i im = _mm512_and_si512(bit_mask2, (__m512i)a);
    __m512   m = (__m512)_mm512_add_epi32(im, offset);

    __m512 cmp = (__m512)_MM512_CMPGT_EPI32((__m512i)PARTITION_CONST, (__m512i)m);

    __m512 fixe = _MM512_AND_PS(cmp, LN2);
    e = _mm512_fmsub_ps(e, LN2, fixe);

    __m512i fixm = _mm512_and_si512((__m512i)cmp, _mm512_set1_epi32(0x00800000));
    m = (__m512)_mm512_add_epi32((__m512i)m, fixm);
    m = _mm512_sub_ps(m, ONE);

    __m512 t =                A;
    t = _mm512_fmadd_ps(t, m, B);
    t = _mm512_fmadd_ps(t, m, C);
    t = _mm512_fmadd_ps(t, m, D);
    t = _mm512_fmadd_ps(t, m, E);
#if PRECISION >= 1
    t = _mm512_fmadd_ps(t, m, F);
    t = _mm512_fmadd_ps(t, m, G);
#endif
#if PRECISION >= 2
    t = _mm512_fmadd_ps(t, m, H);
#endif
    t = _mm512_fmadd_ps(t, m, e);

    __m512 mask0, mask1;
    mask0 = _MM512_CMP_PS(a, TWO_TO_M126, _CMP_NGE_UQ);
    mask1 = (__m512)_MM512_CMPEQ_EPI32((__m512i)a, P_INF);

    if (__builtin_expect(_MM512_MOVEMASK_PS(_MM512_OR_PS(mask0, mask1)) ,0))
    {
        // [0.0, FLT_MIN) u nan -> -inf
        __m512 spec = _MM512_AND_PS(mask0, (__m512)N_INF);

        // (-oo, 0.0) -> nan
        __m512 neg = _MM512_CMP_PS(a, ZERO, _CMP_LT_OQ);
        neg = _MM512_AND_PS(neg, (__m512)NINF2NAN);

        // nan -> nan, inf -> inf
        __m512 unord = _MM512_CMP_PS(a, (__m512)P_INF, _CMP_NLT_UQ);
        unord = _MM512_AND_PS(unord, a);

        spec = _MM512_XOR_PS(spec, neg);
        spec = _mm512_add_ps(spec, unord);
        t = _mm512_add_ps(t, spec);
   }

    return t;
}

