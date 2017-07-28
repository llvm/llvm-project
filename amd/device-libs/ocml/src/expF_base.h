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

PUREATTR float
#if defined COMPILING_EXP2
MATH_MANGLE(exp2)(float x)
#elif defined COMPILING_EXP10
MATH_MANGLE(exp10)(float x)
#else
MATH_MANGLE(exp)(float x)
#endif
{
    if (DAZ_OPT()) {
        if (UNSAFE_MATH_OPT()) {
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
        if (UNSAFE_MATH_OPT()) {
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
}

