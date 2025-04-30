
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
#include "rslog_defs.h"

#define FMAF __builtin_fmaf
#define FMA __builtin_fma

extern "C" float __rss_log_fma3(float);

#ifndef PRECISION
#define PRECISION 1
#endif

#if PRECISION <= 1
float __rss_log_fma3(float a_input)
{
union U {
   float f;
   unsigned u;
};

struct float4 {
   float a, b, c, d;
};

    const static float4* C4 = (const float4*)C_F;

    U CANONICAL_NAN, N_INF, P_INF;
    CANONICAL_NAN.u = CANONICAL_NAN_I;
    N_INF.u = NEG_INF;
    P_INF.u = POS_INF;

    unsigned const bit_mask2 = BIT_MASK2;
    unsigned const offset = OFFSET;
    unsigned exp_offset = EXP_OFFSET;

    U input_union;
    input_union.f = a_input;

    if (__builtin_expect(!(TWO_TO_M126_F <= a_input), 0))
    {
        if (a_input != a_input)
        {
            return a_input + a_input;
        }

        return a_input < 0.0f ? CANONICAL_NAN.f : N_INF.f;
    }

    U mantisa_union;
    mantisa_union.u = input_union.u & bit_mask2;
    int idx = mantisa_union.u >> 17;
    int pred = idx >= 30;

    float4 coeff = C4[idx];

    if (__builtin_expect(a_input == P_INF.f, 0))
    {
        return a_input;
    }

    mantisa_union.u += offset;
    float m = mantisa_union.f;
    int e_int = (input_union.u >> 23) - exp_offset;
    float e = (float)e_int;

    e = pred ? e : e - 1.0f;
    m = pred ? m - 1.0f : FMAF(m, 2.0f, -1.0f);

    float exp = FMAF(e, LN2_F, coeff.d);
    float t = FMAF(coeff.a, m, coeff.b);
    t = FMAF(t, m, coeff.c);
    t = FMAF(t, m, exp);

    return t;
}
#endif

#if PRECISION >= 2
float __rss_log_fma3(float a_input)
{
union U {
   float f;
   unsigned u;
};

struct double4 {
   double a, b, c, d;
};

    const static double4* C4 = (const double4*)C_D;

    U CANONICAL_NAN, N_INF, P_INF;
    CANONICAL_NAN.u = CANONICAL_NAN_I;
    N_INF.u = NEG_INF;
    P_INF.u = POS_INF;

    unsigned const bit_mask2 = BIT_MASK2;
    unsigned long long const offset = OFFSET;
    unsigned const exp_offset = EXP_OFFSET;

    U input_union;
    input_union.f = a_input;

    if (__builtin_expect(!(TWO_TO_M126_F <= a_input), 0))
    {
        if (a_input != a_input)
        {
            return a_input;
        }

        return a_input < 0.0f ? CANONICAL_NAN.f : N_INF.f;
    }

    U mantisa_union;
    mantisa_union.u = input_union.u & bit_mask2;
    int idx = mantisa_union.u >> 17;
    int pred = mantisa_union.u >> 22;

    double4 coeff = C4[idx];

    if (__builtin_expect(a_input == P_INF.f, 0))
    {
        return a_input;
    }

    mantisa_union.u += offset;
    double m = (double)mantisa_union.f;
    int e_int = (input_union.u>>23) - exp_offset;
    double e = (double)e_int;

    e = pred ? e : e - 1.0f;
    m = pred ? m - 1.0f : FMAF(m, 2.0f, -1.0f);

    double t;
    t = FMA(coeff.a, m, coeff.b);
    t = FMA(t, m, coeff.c);
    t = FMA(t, m, coeff.d);
    t = FMA(t, m, e * LN2_D);

    return (float)t;

}
#endif

