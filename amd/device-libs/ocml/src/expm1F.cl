/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

PUREATTR INLINEATTR float
MATH_MANGLE(expm1)(float x)
{
    USE_TABLE(float2, p_tbl, M32_EXP_EP);

    if (MATH_MANGLE(fabs)(x) < 0x1.ep-2f) {
        return MATH_MAD(x*x,
                   MATH_MAD(x,
                       MATH_MAD(x,
                           MATH_MAD(x,
                               MATH_MAD(x,
                                   MATH_MAD(x, 0x1.a01a02p-13f, 0x1.6c16c2p-10f),
                                   0x1.111112p-7f),
                               0x1.555556p-5f),
                           0x1.555556p-3f),
                       0.5f),
                   x);
    } else {
        const float X_MAX = 0x1.62e42ep+6f;  // 128*log2 : 88.722839111673
        const float X_MIN = -0x1.9d1da0p+6f; // -149*log2 : -103.27892990343184

        const float R_64_BY_LOG2 = 0x1.715476p+6f;     // 64/log2 : 92.332482616893657
        const float R_LOG2_BY_64_LD = 0x1.620000p-7f;  // log2/64 lead: 0.0108032227
        const float R_LOG2_BY_64_TL = 0x1.c85fdep-16f; // log2/64 tail: 0.0000272020388

        int n = (int)(x * R_64_BY_LOG2);
        float fn = (float)n;

        int j = n & 0x3f;
        int m = n >> 6;

        float r = MATH_MAD(fn, -R_LOG2_BY_64_TL, MATH_MAD(fn, -R_LOG2_BY_64_LD, x));

        // Truncated Taylor series
        float z2 = MATH_MAD(r*r, MATH_MAD(r, MATH_MAD(r, 0x1.555556p-5f,  0x1.555556p-3f), 0.5f), r);

        float m2 = AS_FLOAT((m + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
        float2 tv = p_tbl[j];
        float two_to_jby64_h = tv.s0 * m2;
        float two_to_jby64_t = tv.s1 * m2;
        float two_to_jby64 = two_to_jby64_h + two_to_jby64_t;

        z2 = MATH_MAD(z2, two_to_jby64, two_to_jby64_t) + (two_to_jby64_h - 1.0f);

        z2 = x < X_MIN | m < -24 ? -1.0f : z2;

        if (!FINITE_ONLY_OPT()) {
            z2 = x > X_MAX ? AS_FLOAT(PINFBITPATT_SP32) : z2;
            z2 = MATH_MANGLE(isnan)(x) ? x : z2;
        }

        return z2;
    }
}

