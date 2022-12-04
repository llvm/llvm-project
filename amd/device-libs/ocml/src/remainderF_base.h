/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

// The arguments must only be variable names
#define FULL_MUL(A, B, CHI, CLO) \
    do { \
        float __ha = AS_FLOAT(AS_UINT(A) & 0xfffff000U); \
        float __ta = A - __ha; \
        float __hb = AS_FLOAT(AS_UINT(B) & 0xfffff000U); \
        float __tb = B - __hb; \
        CHI = A * B; \
        CLO = MATH_MAD(__ta, __tb, MATH_MAD(__ta, __hb, MATH_MAD(__ha, __tb, MATH_MAD(__ha, __hb, -CHI)))); \
    } while (0)

CONSTATTR static float
fnma(float a, float b, float c)
{
    float d;
    if (HAVE_FAST_FMA32()) {
        d = BUILTIN_FMA_F32(-a, b, c);
    } else {
        float h, t;
        FULL_MUL(a, b, h, t);
        d = c - h;
        d = (((c - d) - h) - t) + d;
    }
    return d;
}

#if defined(COMPILING_FMOD)
CONSTATTR float
MATH_MANGLE(fmod)(float x, float y)
#elif defined(COMPILING_REMQUO)
__ocml_remquo_f32_result
MATH_MANGLE(remquo2)(float x, float y)
#else
CONSTATTR float
MATH_MANGLE(remainder)(float x, float y)
#endif
{
    // How many bits of the quotient per iteration
    const int bits = 12;
    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);

    float ret;
#if defined(COMPILING_REMQUO)
    int q7;
#endif

    if (ax > ay) {
        int ex, ey;

        ex = BUILTIN_FREXP_EXP_F32(ax) - 1;
        ax = BUILTIN_FLDEXP_F32(BUILTIN_FREXP_MANT_F32(ax), bits);
        ey = BUILTIN_FREXP_EXP_F32(ay) - 1;
        ay = BUILTIN_FLDEXP_F32(BUILTIN_FREXP_MANT_F32(ay), 1);

        int nb = ex - ey;
        float ayinv = MATH_FAST_RCP(ay);

#if !defined(COMPILING_FMOD)
        int qacc = 0;
#endif

        while (nb > bits) {
            float q = BUILTIN_RINT_F32(ax * ayinv);
            ax = fnma(q, ay, ax);
            int clt = ax < 0.0f;
            float axp = ax + ay;
            ax = clt ? axp : ax;
#if defined(COMPILING_REMQUO)
            int iq = (int)q;
            iq -= clt;
            qacc = (qacc << bits) | iq;
#endif
            ax = BUILTIN_FLDEXP_F32(ax, bits);
            nb -= bits;
        }

        ax = BUILTIN_FLDEXP_F32(ax, nb - bits + 1);

        // Final iteration
        {
            float q = BUILTIN_RINT_F32(ax * ayinv);
            ax = fnma(q, ay, ax);
            int clt = ax < 0.0f;
            float axp = ax + ay;
            ax = clt ? axp : ax;
#if !defined(COMPILING_FMOD)
            int iq = (int)q;
            iq -= clt;
#if defined(COMPILING_REMQUO)
            qacc = (qacc << (nb+1)) | iq;
#else
            qacc = iq;
#endif
#endif
        }

#if !defined(COMPILING_FMOD)
        // Adjust ax so that it is the range (-y/2, y/2]
        // We need to choose the even integer when x/y is midway between two integers
        int aq = (2.0f*ax > ay) | ((qacc & 0x1) & (2.0f*ax == ay));
        ax = ax - (aq ? ay : 0.0f);
#if defined(COMPILING_REMQUO)
        qacc += aq;
        int qneg = (AS_INT(x) ^ AS_INT(y)) >> 31;
        q7 = ((qacc & 0x7f) ^ qneg) - qneg;
#endif
#endif

        ax = BUILTIN_FLDEXP_F32(ax, ey);
        ret = AS_FLOAT((AS_INT(x) & SIGNBIT_SP32) ^ AS_INT(ax));
    } else {
        ret = x;
#if defined(COMPILING_REMQUO)
        q7 = 0;
#endif

#if !defined(COMPILING_FMOD)
        bool c = (ay < 0x1.0p+127f & 2.0f*ax > ay) | (ax > 0.5f*ay);

        int qsgn = 1 + (((AS_INT(x) ^ AS_INT(y)) >> 31) << 1);
        float t = MATH_MAD(y, -(float)qsgn, x);
        ret = c ? t : (DAZ_OPT() ? BUILTIN_CANONICALIZE_F32(x) : x);
#if defined(COMPILING_REMQUO)
        q7 = c ? qsgn : q7;
#endif
#endif

        ret = ax == ay ? BUILTIN_COPYSIGN_F32(0.0f, x) : ret;
#if defined(COMPILING_REMQUO)
        q7 = ax == ay ? qsgn : q7;
#endif
    }

    if (!FINITE_ONLY_OPT()) {
        ret = y == 0.0f ? QNAN_F32 : ret;
#if defined(COMPILING_REMQUO)
        q7 = y == 0.0f ? 0 : q7;
#endif

        bool c = !BUILTIN_ISNAN_F32(y) && BUILTIN_ISFINITE_F32(x);
        ret = c ? ret : QNAN_F32;
#if defined(COMPILING_REMQUO)
        q7 = c ? q7 : 0;
#endif
    }

#if defined(COMPILING_REMQUO)
    __ocml_remquo_f32_result result = { ret, q7 };
    return result;
#else
    return ret;
#endif

}

#if defined(COMPILING_REMQUO)
float
MATH_MANGLE(remquo)(float x, float y, __private int *q7p) {
    __ocml_remquo_f32_result result = MATH_MANGLE(remquo2)(x, y);
    *q7p = result.quo;
    return result.rem;
}
#endif
