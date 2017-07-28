/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
#if defined COMPILING_LOG2
MATH_MANGLE(log2)(float x)
#elif defined COMPILING_LOG10
MATH_MANGLE(log10)(float x)
#else
MATH_MANGLE(log)(float x)
#endif
{
    if (DAZ_OPT()) {
        if (UNSAFE_MATH_OPT()) {
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
                float yh = AS_FLOAT(AS_UINT(y) & 0xfffff000);
                float yt = y - yh;
	            r = MATH_MAD(yh, ch, MATH_MAD(yt, ch, MATH_MAD(yh, ct, yt*ct)));
            }

            r = BUILTIN_CLASS_F32(y, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) != 0 ? y : r;
            return r;
#endif
        }
    } else {
        // not DAZ
        if (UNSAFE_MATH_OPT()) {
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
                float yh = AS_FLOAT(AS_UINT(y) & 0xfffff000);
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
}

