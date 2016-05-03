
#include "mathF.h"

/*
   Algorithm:

   Based on:
   Ping-Tak Peter Tang
   "Table-driven implementation of the logarithm function in IEEE
   floating-point arithmetic"
   ACM Transactions on Mathematical Software (TOMS)
   Volume 16, Issue 4 (December 1990)


   x very close to 1.0 is handled differently, for x everywhere else
   a brief explanation is given below

   x = (2^m)*A
   x = (2^m)*(G+g) with (1 <= G < 2) and (g <= 2^(-8))
   x = (2^m)*2*(G/2+g/2)
   x = (2^m)*2*(F+f) with (0.5 <= F < 1) and (f <= 2^(-9))

   Y = (2^(-1))*(2^(-m))*(2^m)*A
   Now, range of Y is: 0.5 <= Y < 1

   F = 0x80 + (first 7 mantissa bits) + (8th mantissa bit)
   Now, range of F is: 128 <= F <= 256 
   F = F / 256 
   Now, range of F is: 0.5 <= F <= 1

   f = -(Y-F), with (f <= 2^(-9))

   log(x) = m*log(2) + log(2) + log(F-f)
   log(x) = m*log(2) + log(2) + log(F) + log(1-(f/F))
   log(x) = m*log(2) + log(2*F) + log(1-r)

   r = (f/F), with (r <= 2^(-8))
   r = f*(1/F) with (1/F) precomputed to avoid division

   log(x) = m*log(2) + log(G) - poly

   log(G) is precomputed
   poly = (r + (r^2)/2 + (r^3)/3 + (r^4)/4) + (r^5)/5))

   log(2) and log(G) need to be maintained in extra precision
   to avoid losing precision in the calculations


   For x close to 1.0, we employ the following technique to
   ensure faster convergence.

   log(x) = log((1+s)/(1-s)) = 2*s + (2/3)*s^3 + (2/5)*s^5 + (2/7)*s^7
   x = ((1+s)/(1-s)) 
   x = 1 + r
   s = r/(2+r)

*/

INLINEATTR PUREATTR float
#if defined COMPILING_LOG2
MATH_MANGLE(log2)(float x)
#elif defined COMPILING_LOG10
MATH_MANGLE(log10)(float x)
#else
MATH_MANGLE(log)(float x)
#endif
{
    if (AMD_OPT()) {
        if (DAZ_OPT()) {
            if (FAST_RELAXED_OPT()) {
#if defined COMPILING_LOG2
                return BUILTIN_LOG2_F32(x);
#elif defined COMPILING_LOG10
                return BUILTIN_LOG2_F32(x) * 0x1.344136p-2f;
#else
                return BUILTIN_LOG2_F32(x) * 0x1.62e430p-1f;
#endif
            } else {
#if defined COMPILING_LOG2
                return BUILTIN_LOG2_F32(x);
#else
                float y = BUILTIN_LOG2_F32(x);
                float r;

                if (HAVE_FAST_FMA32()) {
#if defined COMPILING_LOG10
                    const float c = 0x1.344134p-2f;
                    const float cc = 0x1.09f79ep-26f; // c+cc are ln(2)/ln(10) to more than 49 bits
#else
                    const float c = 0x1.62e42ep-1f;
                    const float cc = 0x1.efa39ep-25f; // c + cc is ln(2) to more than 49 bits
#endif
	            r = y * c;
	            r = r + BUILTIN_FMA_F32(y, cc, BUILTIN_FMA_F32(y, c, -r));
                } else {
#if defined COMPILING_LOG10
                    const float ch = 0x1.344000p-2f;
                    const float ct = 0x1.3509f6p-18f; // ch+ct is ln(2)/ln(10) to more than 36 bits
#else
                    const float ch = 0x1.62e000p-1f;
                    const float ct = 0x1.0bfbe8p-15f; // ch + ct is ln(2) to more than 36 bits
#endif
                    float yh = as_float(as_uint(y) & 0xfffff000);
                    float yt = y - yh;
	            r = MATH_MAD(yh, ch, MATH_MAD(yt, ch, MATH_MAD(yh, ct, yt*ct)));
                }

                r = BUILTIN_CLASS_F32(y, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) != 0 ? y : r;
                return r;
#endif
            }
        } else {
            // not DAZ
            if (FAST_RELAXED_OPT()) {
                bool s = BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB);
                x *= s ? 0x1.0p+32f : 1.0f;
#if defined COMPILING_LOG2
                return BUILTIN_LOG2_F32(x) - (s ? 32.0f : 0.0f);
#elif defined COMPILING_LOG10
                return MATH_MAD(BUILTIN_LOG2_F32(x), 0x1.344136p-2f, s ? -0x1.344136p+3f : 0.0f);
#else
                return MATH_MAD(BUILTIN_LOG2_F32(x), 0x1.62e430p-1f, s ? -0x1.62e430p+4f : 0.0f);
#endif
            } else {
                bool s = BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB);
                x *= s ? 0x1.0p+32f : 1.0f;
#if defined COMPILING_LOG2
                return BUILTIN_LOG2_F32(x) - (s ? 32.0f : 0.0f);
#else
                float y = BUILTIN_LOG2_F32(x);
                float r;

                if (HAVE_FAST_FMA32()) {
#if defined COMPILING_LOG10
                    const float c = 0x1.344134p-2f;
                    const float cc = 0x1.09f79ep-26f; // c+cc are ln(2)/ln(10) to more than 49 bits
#else
                    const float c = 0x1.62e42ep-1f;
                    const float cc = 0x1.efa39ep-25f; // c + cc is ln(2) to more than 49 bits
#endif
	            r = y * c;
	            r = r + BUILTIN_FMA_F32(y, cc, BUILTIN_FMA_F32(y, c, -r));
                } else {
#if defined COMPILING_LOG10
                    const float ch = 0x1.344000p-2f;
                    const float ct = 0x1.3509f6p-18f; // ch+ct is ln(2)/ln(10) to more than 36 bits
#else
                    const float ch = 0x1.62e000p-1f;
                    const float ct = 0x1.0bfbe8p-15f; // ch + ct is ln(2) to more than 36 bits
#endif
                    float yh = as_float(as_uint(y) & 0xfffff000);
                    float yt = y - yh;
	            r = MATH_MAD(yh, ch, MATH_MAD(yt, ch, MATH_MAD(yh, ct, yt*ct)));
                }

                r = BUILTIN_CLASS_F32(y, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) != 0 ? y : r;

#if defined COMPILING_LOG10
                r = r - (s ? 0x1.344136p+3f : 0.0f);
#else
                r = r - (s ? 0x1.62e430p+4f : 0.0f);
#endif

                // r = BUILTIN_CLASS_F32(y, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) != 0 ? y : r;
                return r;
#endif
            }
        }
    } else {
        USE_TABLE(float, p_inv, M32_LOG_INV);

#if defined COMPILING_LOG2
        USE_TABLE(float2, p_log, M32_LOG2);
        const float LOG2E = 0x1.715476p+0f;      // 1.4426950408889634
        const float LOG2E_HEAD = 0x1.700000p+0f; // 1.4375
        const float LOG2E_TAIL = 0x1.547652p-8f; // 0.00519504072
#elif defined COMPILING_LOG10
        USE_TABLE(float2, p_log, M32_LOG10);
        const float LOG10E = 0x1.bcb7b2p-2f;        // 0.43429448190325182
        const float LOG10E_HEAD = 0x1.bc0000p-2f;   // 0.43359375
        const float LOG10E_TAIL = 0x1.6f62a4p-11f;  // 0.0007007319
        const float LOG10_2_HEAD = 0x1.340000p-2f;  // 0.30078125
        const float LOG10_2_TAIL = 0x1.04d426p-12f; // 0.000248745637
#else
        USE_TABLE(float2, p_log, M32_LOGE);
        const float LOG2_HEAD = 0x1.62e000p-1f;  // 0.693115234
        const float LOG2_TAIL = 0x1.0bfbe8p-15f; // 0.0000319461833
#endif

        float z;

        float r = x - 1.0f;
        if (BUILTIN_ABS_F32(r) >= 0x1.0p-4f) {
            int m;
            uint xi;
            if (!DAZ_OPT()) {
                float xs = x * 0x1.0p+30f;
                bool c = x < 0x1.0p-96f;
                xi = as_uint(c ? xs : x);
                m = (int)(xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32 - (c ? 30 : 0);
            } else {
                x = BUILTIN_CANONICALIZE_F32(x);
                xi = as_uint(x);
                m = (int)(xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
            }

            float mf = (float)m;
            uint indx = (xi & 0x007f0000) + ((xi & 0x00008000) << 1);

            // F - Y
            float f = as_float(0x3f000000 | indx) - as_float(0x3f000000 | (xi & MANTBITS_SP32));

            indx = indx >> 16;
            r = f * p_inv[indx];

            // 1/3,  1/2
            float poly = MATH_MAD(MATH_MAD(r, 0x1.555556p-2f, 0.5f), r*r, r);

            float2 tv = p_log[indx];
            float z1, z2;

#if defined COMPILING_LOG2
            z1 = tv.s0 + mf;
            z2 = MATH_MAD(poly, -LOG2E, tv.s1);
#elif defined COMPILING_LOG10
            z1 = MATH_MAD(mf, LOG10_2_HEAD, tv.s0);
            z2 = MATH_MAD(poly, -LOG10E, mf*LOG10_2_TAIL) + tv.s1;
#else
            z1 = MATH_MAD(mf, LOG2_HEAD, tv.s0);
            z2 = MATH_MAD(mf, LOG2_TAIL, -poly) + tv.s1;
#endif

            z = z1 + z2;
        } else {
            // Calculations for |x-1| < 2^-4
            float u2 = MATH_FAST_DIV(r, 2.0f + r);
            float corr = u2 * r;
            float u = u2 + u2;
            float v = u * u;
            float z1, z2;

            // 2/(5 * 2^5), 2/(3 * 2^3)
            z2 = MATH_MAD(u, MATH_MAD(v, 0x1.99999ap-7f, 0x1.555556p-4f)*v, -corr);

#if defined COMPILING_LOG2
            z1 = as_float(as_int(r) & 0xffff0000);
            z2 = z2 + (r - z1);
            z = MATH_MAD(z1, LOG2E_HEAD, MATH_MAD(z2, LOG2E_HEAD, MATH_MAD(z1, LOG2E_TAIL, z2*LOG2E_TAIL)));
#elif defined COMPILING_LOG10
            z1 = as_float(as_int(r) & 0xffff0000);
            z2 = z2 + (r - z1);
            z = MATH_MAD(z1, LOG10E_HEAD, MATH_MAD(z2, LOG10E_HEAD, MATH_MAD(z1, LOG10E_TAIL, z2*LOG10E_TAIL)));
#else
            z = z2 + r;
#endif
        }

        // Corner cases
        if (!FINITE_ONLY_OPT()) {
            uint xi = as_uint(x);
            uint ax = xi & EXSIGNBIT_SP32;
            z = ax >= PINFBITPATT_SP32 ? x : z;
            z = xi != ax ? as_float(QNANBITPATT_SP32) : z;
            z = ax == 0 ? as_float(NINFBITPATT_SP32) : z;
        }

        return z;
    }
}

