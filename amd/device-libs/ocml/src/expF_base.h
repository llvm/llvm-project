/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

//    Algorithm:
// 
//    e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
// 
//    x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//    n = 64*m + j,   0 <= j < 64
// 
//    e^x = 2^((64*m + j + f)/64)
//        = (2^m) * (2^(j/64)) * 2^(f/64)
//        = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
// 
//    f = x*(64/ln(2)) - n
//    r = f*(ln(2)/64) = x - n*(ln(2)/64)
// 
//    e^x = (2^m) * (2^(j/64)) * e^r
// 
//    (2^(j/64)) is precomputed
// 
//    e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//    e^r = 1 + q
// 
//    q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
// 
//    e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) ) 

PUREATTR INLINEATTR float
#if defined COMPILING_EXP2
MATH_MANGLE(exp2)(float x)
#elif defined COMPILING_EXP10
MATH_MANGLE(exp10)(float x)
#else
MATH_MANGLE(exp)(float x)
#endif
{
    if (AMD_OPT()) {
        if (DAZ_OPT()) {
            if (FAST_RELAXED_OPT()) {
#if defined COMPILING_EXP2
                return BUILTIN_EXP2_F32(x);
#elif defined COMPILING_EXP10
                return BUILTIN_EXP2_F32(x * 0x1.a92000p+1f) * BUILTIN_EXP2_F32(x * 0x1.4f0978p-11f);
#else
                return BUILTIN_EXP2_F32(x * 0x1.715476p+0f);
#endif
            } else {
#if defined COMPILING_EXP2
                return BUILTIN_EXP2_F32(x);
#else
                float ph, pl;

                if (HAVE_FAST_FMA32()) {
#if defined COMPILING_EXP
                    const float c = 0x1.715476p+0f;
                    const float cc = 0x1.4ae0bep-26f; // c+cc are 49 bits
#else
                    const float c = 0x1.a934f0p+1f;
                    const float cc = 0x1.2f346ep-24f;
#endif
                    ph = x * c;
                    pl = BUILTIN_FMA_F32(x, cc, BUILTIN_FMA_F32(x, c, -ph));
                } else {
#if defined COMPILING_EXP
                    const float ch = 0x1.714000p+0f;
                    const float cl = 0x1.47652ap-12f; // ch + cl are 36 bits
#else
                    const float ch = 0x1.a92000p+1f;
                    const float cl = 0x1.4f0978p-11f;
#endif
                    float xh = AS_FLOAT(AS_INT(x) & 0xfffff000);
                    float xl = x - xh;
                    ph = xh * ch;
                    pl = MATH_MAD(xh, cl, MATH_MAD(xl, ch, xl*cl));
                }

                float r = BUILTIN_EXP2_F32(pl) * BUILTIN_EXP2_F32(ph);

#if defined COMPILING_EXP
                r = x < -0x1.5d58a0p+6f ? 0.0f : r;
                r = x > 0x1.62e430p+6f ? AS_FLOAT(PINFBITPATT_SP32) : r;
#else
                r = x < -0x1.2f7030p+5f ? 0.0f : r;
                r = x > 0x1.344136p+5f ? AS_FLOAT(PINFBITPATT_SP32): r;
#endif
                return r;
#endif
            }
        } else {
            if (FAST_RELAXED_OPT()) {
#if defined COMPILING_EXP2
                bool s = x < -0x1.f80000p+6f;
                return BUILTIN_EXP2_F32(x + (s ? 0x1.0p+6f : 0.0f)) * (s ? 0x1.0p-64f : 1.0f);
#elif defined COMPILING_EXP10
                bool s = x < -0x1.2f7030p+5f;
                x += s ? 0x1.0p+5f : 0.0f;
                return BUILTIN_EXP2_F32(x * 0x1.a92000p+1f) *
                       BUILTIN_EXP2_F32(x * 0x1.4f0978p-11f) *
                       (s ? 0x1.9f623ep-107f : 1.0f);
#else
                bool s = x < -0x1.5d58a0p+6f;
                return BUILTIN_EXP2_F32((x + (s ? 0x1.0p+6f : 0.0f)) * 0x1.715476p+0f) * (s ? 0x1.969d48p-93f : 1.0f);
#endif
            } else {
#if defined COMPILING_EXP2
                bool s = x < -0x1.f80000p+6f;
                return BUILTIN_EXP2_F32(x + (s ? 0x1.0p+6f : 0.0f)) * (s ? 0x1.0p-64f : 1.0f);
#else
                float ph, pl;

#if defined COMPILING_EXP
                bool s = x < -0x1.5d58a0p+6f;
                x += s ? 0x1.0p+6f : 0.0f;
#else
                bool s = x < -0x1.2f7030p+5f;
                x += s ? 0x1.0p+5f : 0.0f;
#endif

                if (HAVE_FAST_FMA32()) {
#if defined COMPILING_EXP
                    const float c = 0x1.715476p+0f;
                    const float cc = 0x1.4ae0bep-26f; // c+cc are 49 bits
#else
                    const float c = 0x1.a934f0p+1f;
                    const float cc = 0x1.2f346ep-24f;
#endif
                    ph = x * c;
                    pl = BUILTIN_FMA_F32(x, cc, BUILTIN_FMA_F32(x, c, -ph));
                } else {
#if defined COMPILING_EXP
                    const float ch = 0x1.714000p+0f;
                    const float cl = 0x1.47652ap-12f; // ch + cl are 36 bits
#else
                    const float ch = 0x1.a92000p+1f;
                    const float cl = 0x1.4f0978p-11f;
#endif
                    float xh = AS_FLOAT(AS_INT(x) & 0xfffff000);
                    float xl = x - xh;
                    ph = xh * ch;
                    pl = MATH_MAD(xh, cl, MATH_MAD(xl, ch, xl*cl));
                }

                float r = BUILTIN_EXP2_F32(pl) * BUILTIN_EXP2_F32(ph);

#if defined COMPILING_EXP
                r *= s ? 0x1.969d48p-93f : 1.0f;
                r = x < -0x1.9d1da0p+6f ? 0.0f : r;
                r = x > 0x1.62e430p+6f ? AS_FLOAT(PINFBITPATT_SP32) : r;
#else
                r *= s ? 0x1.9f623ep-107f : 1.0f;
                r = x < -0x1.66d3e8p+5f ? 0.0f : r;
                r = x > 0x1.344136p+5f ? AS_FLOAT(PINFBITPATT_SP32): r;
#endif
                return r;
#endif
            }
        }
    } else {
#if defined COMPILING_EXP2 || defined COMPILING_EXP
        // Use a table free approach for exp and exp2

        // Reduce x
        const float ln2hi = 0x1.62e300p-1f;
        const float ln2lo = 0x1.2fefa2p-17f;
        const float invln2 = 0x1.715476p+0f;

#if defined COMPILING_EXP
        float fp  = BUILTIN_TRUNC_F32(MATH_MAD(x, invln2, BUILTIN_COPYSIGN_F32(0.5f, x)));
        int p = (int)fp;
        float hi = MATH_MAD(fp, -ln2hi, x);
        float lo = -fp*ln2lo;
#else
        float fp = BUILTIN_RINT_F32(x);
        int p = (int)fp;
        float fx = x - fp;
        float hi = fx * ln2hi;
        float lo = fx * ln2lo;
#endif

        // Evaluate poly
        float t = hi + lo;   
        float tt  = t*t;
        float v = MATH_MAD(tt,
                           -MATH_MAD(tt,
                                     MATH_MAD(tt,
                                              MATH_MAD(tt,
                                                       MATH_MAD(tt, 0x1.637698p-25f, -0x1.bbd41cp-20f),
                                                       0x1.1566aap-14f),
                                              -0x1.6c16c2p-9f),
                                     0x1.555556p-3f),
                           t); 

        float y = 1.0f - (((-lo) - MATH_FAST_DIV(t * v,  2.0f - v)) - hi);

        // Scale by 2^p
        float r;

        if (AMD_OPT()) {
            r = BUILTIN_FLDEXP_F32(y, p);
        } else if (DAZ_OPT()) {
            r = AS_FLOAT(AS_INT(y) + (p << 23));
        } else {
            int p2 = p >> 1;
            float sc1 = AS_FLOAT((p2 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
            float sc2 = AS_FLOAT(((p - p2) + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
            r = (y * sc1) * sc2;
        }

        if (DAZ_OPT()) {
#if defined COMPILING_EXP
            const float llim = -0x1.5d589ep+6f;
#else
            const float llim = -126.0f;
#endif
            r = x < llim ? 0.0f : r;
        } else {
#if defined COMPILING_EXP
            const float llim = -0x1.9d1da0p+6f;
#else
            const float llim = -149.0f;
#endif
            r = x < llim ? 0.0f : r;
        }

        if (!FINITE_ONLY_OPT()) {
#if defined COMPILING_EXP
            const float ulim =  0x1.62e430p+6f; // ln(largest_normal) = 88.72283905206835305366
#else
            const float ulim = 128.0f;
#endif
            r = x < ulim ? r : AS_FLOAT(PINFBITPATT_SP32);
            r = MATH_MANGLE(isnan)(x) ? x : r;
        }
        return r;
#else
        // This code is only used currently for exp10
        // but it handles exp and exp2 as well
        USE_TABLE(float, p_tbl, M32_EXP);

#if defined COMPILING_EXP2
        const float X_MAX =  0x1.fffffep+6f; // 128
        const float X_MIN = -0x1.2a0000p+7f; // -149
#elif defined COMPILING_EXP10
        const float X_MAX =  0x1.344134p+5f; // 128*log2/log10 : 38.53183944498959 
        const float X_MIN = -0x1.66d3e8p+5f; // -149*log2/log10 : -44.8534693539332
#else
        const float X_MAX =  0x1.62e42ep+6f; // 128*log2 : 88.722839111673
        const float X_MIN = -0x1.9d1da0p+6f; // -149*log2 : -103.27892990343184
#endif

#if defined COMPILING_EXP2
        const float R_64 = 0x1.000000p+6f; // 2^6 
        const float R_1_BY_64 = 0x1.000000p-6f; // 2^-6
        const float R_LN2 = 0x1.62e430p-1f; // 0.6931471805599453 
#elif defined COMPILING_EXP10
        const float R_64_BY_LOG10_2 = 0x1.a934f0p+7f; // 64*log10/log2 : 212.6033980727912
        const float R_LOG10_2_BY_64_LD = 0x1.340000p-8f; // log2/(64 * log10) lead : 0.004699707
        const float R_LOG10_2_BY_64_TL = 0x1.04d426p-18f; // log2/(64 * log10) tail : 0.00000388665057
        const float R_LN10 = 0x1.26bb1cp+1f;
#else
        const float R_64_BY_LOG2 = 0x1.715476p+6f; // 64/log2 : 92.332482616893657
        const float R_LOG2_BY_64_LD = 0x1.620000p-7f; /* log2/64 lead: 0.0108032227 */
        const float R_LOG2_BY_64_TL = 0x1.c85fdep-16f; /* log2/64 tail: 0.0000272020388 */
#endif

#if defined COMPILING_EXP2
        int n = (int)(x * R_64);
#elif defined COMPILING_EXP10
        int n = (int)(x * R_64_BY_LOG10_2);
#else
        int n = (int)(x * R_64_BY_LOG2);
#endif

        float fn = (float)n;
        int j = n & 0x3f;
        int m = n >> 6;
        float r;

#if defined COMPILING_EXP2
        r = R_LN2 * MATH_MAD(-R_1_BY_64, fn, x);
#elif defined COMPILING_EXP10
        r = R_LN10 * MATH_MAD(fn, -R_LOG10_2_BY_64_TL, MATH_MAD(fn, -R_LOG10_2_BY_64_LD, x));
#else
        r = MATH_MAD(fn, -R_LOG2_BY_64_TL, MATH_MAD(fn, -R_LOG2_BY_64_LD, x));
#endif

        // Truncated Taylor series for e^r
        float z2 = MATH_MAD(MATH_MAD(MATH_MAD(r, 0x1.555556p-5f, 0x1.555556p-3f), r, 0x1.000000p-1f), r*r, r);

        float two_to_jby64 = p_tbl[j];
        z2 = MATH_MAD(two_to_jby64, z2, two_to_jby64);

        float z2s = z2 * AS_FLOAT(0x1 << (m + 149));
        float z2n = AS_FLOAT(AS_INT(z2) + (m <<EXPSHIFTBITS_SP32));
        z2 = m <= -126 ? z2s : z2n;

        if (!FINITE_ONLY_OPT()) {
            z2 = MATH_MANGLE(isnan)(x) ? x : z2;
            z2 = x > X_MAX ? AS_FLOAT(PINFBITPATT_SP32) : z2;
        }

        z2 = x < X_MIN ? 0.0f : z2;
        return z2;
#endif
    }
}

