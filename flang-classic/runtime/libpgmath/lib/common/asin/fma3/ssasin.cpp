
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
#include "asin_defs.h"

//#define INTEL_NAN

extern "C" float __fss_asin_fma3(float);

static int __float_as_int(float const a) {
    return *(int*)&a;
}

static float __int_as_float(int const a) {
    return *(float*)&a;
}


float __fss_asin_fma3(float const a) {
    __m128 const PIO2 = _mm_set1_ps(PIO2_F);

    // p0, p1 coefficients
    __m128 const A = _mm_setr_ps(A0_F, A1_F, 0.0f, 0.0f);
    __m128 const B = _mm_setr_ps(B0_F, B1_F, 0.0f, 0.0f);
    __m128 const C = _mm_setr_ps(C0_F, C1_F, 0.0f, 0.0f);
    __m128 const D = _mm_setr_ps(D0_F, D1_F, 0.0f, 0.0f);
    __m128 const E = _mm_setr_ps(E0_F, E1_F, 0.0f, 0.0f);
    __m128 const F = _mm_setr_ps(0.0f, F1_F, 0.0f, 0.0f);
    __m128 const G = _mm_setr_ps(0.0f, G1_F, 0.0f, 0.0f);

    __m128 _x2_x, _x, _x3, p, p0, p1, _sq;

    float x, sq, res;
    x = __int_as_float(ABS_MASK_I & __float_as_int(a));
    _x2_x = _mm_setr_ps(a * a, x, 0.0f, 0.0f);
    _x = _mm_set1_ps(x);
    _x3 = _mm_mul_ps(_x2_x, _x);

    // p0, p1 evaluation
    p = _mm_fmadd_ps(A, _x2_x, B);
    p = _mm_fmadd_ps(p, _x2_x, C);
    p = _mm_fmadd_ps(p, _x2_x, D);
    p = _mm_fmadd_ps(p, _x2_x, E);

    p0 = _mm_fmadd_ps(p, _x3, _x);
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
        p1 = _mm_fmadd_ps(p1, _x2_x, G);
        p1 = _mm_fmadd_ps(_sq, p1, PIO2);
        res = __int_as_float(_mm_extract_ps(p1, 1));
    }

#ifndef INTEL_NAN // GCC NAN:
    int sign, fix;

    sign = SGN_MASK_I & __float_as_int(a);
    fix = (a > 1.0f) << 31;
    fix ^= sign;
    res = __int_as_float(__float_as_int(res) ^ fix);
#else // INTEL NAN:
    int sign;

    sign = SGN_MASK_I & __float_as_int(a);
    res = __int_as_float(__float_as_int(res) | sign);
#endif

    return res;
}

