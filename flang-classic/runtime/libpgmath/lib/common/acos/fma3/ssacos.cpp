
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
#include <math.h>
#include "acos_defs.h"

extern "C" float __fss_acos_fma3(float);

static int __float_as_int(float const a) {
    return *(int*)&a;
}

static float __int_as_float(int const a) {
    return *(float*)&a;
}

float __fss_acos_fma3(float const a)
{
    __m128i const ZERO          = _mm_set1_epi32(0);
    __m128  const PI            = _mm_set1_ps(PI_F);

    // p0, p1 coefficients
    __m128 const A = _mm_setr_ps(A0_F, A1_F, 0.0f, 0.0f);
    __m128 const B = _mm_setr_ps(B0_F, B1_F, 0.0f, 0.0f);
    __m128 const C = _mm_setr_ps(C0_F, C1_F, 0.0f, 0.0f);
    __m128 const D = _mm_setr_ps(D0_F, D1_F, 0.0f, 0.0f);
    __m128 const E = _mm_setr_ps(E0_F, E1_F, 0.0f, 0.0f);
    __m128 const F = _mm_setr_ps(F0_F, F1_F, 0.0f, 0.0f);

    __m128 _x2_x, _a, _a3, p, p0, p1, _sq, _c;
    float x, sq, res;
    x = __int_as_float(ABS_MASK_I & __float_as_int(a));
    _x2_x = _mm_setr_ps(a * a, x, 0.0f, 0.0f);
    _a = _mm_set1_ps(a);
    _a3 = _mm_mul_ps(_x2_x, _a);
    _c = _mm_sub_ps(F, _a);

    p = _mm_fmadd_ps(A, _x2_x, B);
    p = _mm_fmadd_ps(p, _x2_x, C);
    p = _mm_fmadd_ps(p, _x2_x, D);
    p = _mm_fmadd_ps(p, _x2_x, E);
    p0 = _mm_fmadd_ps(p, _a3, _c);
    res = __int_as_float(_mm_extract_ps(p0, 0));

    if (__float_as_int(x) > __float_as_int(THRESHOLD_F))
    {
        sq = 1.0f - x;
	/*
	 * There seems to be a concensus that setting errno is important
	 * for fastmath intrinsics.
	 * Disable using Intel hardware instruction sqrt.
	 */
	sq = sqrtf(sq);
        _sq = _mm_setr_ps(0.0f, sq, 0.0f, 0.0f);
        p1 = _mm_fmadd_ps(p, _x2_x, F);

#if defined(__clang__) && defined(TARGET_ARM64)
        __m128 pi_mask = (__m128)((long double)_mm_cmpgt_epi32(ZERO, (__m128i)((long double)_a)));
#else
        __m128 pi_mask = (__m128)_mm_cmpgt_epi32(ZERO, (__m128i)_a);
#endif
        pi_mask = _mm_and_ps(pi_mask, PI);
        p1 = _mm_fmsub_ps(_sq, p1, pi_mask);

        res = __int_as_float(_mm_extract_ps(p1, 1));

        int sign;
        sign = SGN_MASK_I & __float_as_int(a);

        int fix;
        fix = (a > 1.0f) << 31;
        fix ^= sign;
        res = __int_as_float(__float_as_int(res) ^ fix);
    }

    return res;
}
