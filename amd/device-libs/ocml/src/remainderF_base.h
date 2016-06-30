
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

CONSTATTR static inline float
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
float
MATH_MANGLE(remquo)(float x, float y, __private int *q7p)
#else
CONSTATTR float
MATH_MANGLE(remainder)(float x, float y)
#endif
{
    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
        y = BUILTIN_CANONICALIZE_F32(y);
    }

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

        if (AMD_OPT()) {
            ex = BUILTIN_FREXP_EXP_F32(ax) - 1;
            ax = BUILTIN_FLDEXP_F32(BUILTIN_FREXP_MANT_F32(ax), bits);
            ey = BUILTIN_FREXP_EXP_F32(ay) - 1;
            ay = BUILTIN_FLDEXP_F32(BUILTIN_FREXP_MANT_F32(ay), 1);
        } else {
            if (!DAZ_OPT()) {
                int exs = -118 - (int)MATH_CLZI(AS_INT(ax));
                float axs = AS_FLOAT(((EXPBIAS_SP32+bits-1) << EXPSHIFTBITS_SP32) |
                                     ((AS_INT(ax) << (-126 - exs)) & MANTBITS_SP32));
                ex = (AS_INT(ax) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
                ax = AS_FLOAT(((EXPBIAS_SP32+bits-1) << EXPSHIFTBITS_SP32) | (AS_INT(ax) & MANTBITS_SP32));
                ax = ex == -EXPBIAS_SP32 ? axs : ax;
                ex = ex == -EXPBIAS_SP32 ? exs : ex;
            } else {
                ex = (AS_INT(ax) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
                ax = AS_FLOAT(((EXPBIAS_SP32+bits-1) << EXPSHIFTBITS_SP32) | (AS_INT(ax) & MANTBITS_SP32));
            }
            ax = x == 0.0f ? 0.0f : ax;
            ex = x == 0.0f ? 0 : ex;

            if (!DAZ_OPT()) {
                int eys = -118 - (int)MATH_CLZI(AS_INT(ay));
                float ays = AS_FLOAT((EXPBIAS_SP32 << EXPSHIFTBITS_SP32) | (AS_INT(ay) << (-126 - eys)));
                ey = (AS_INT(ay) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
                ay = AS_FLOAT((EXPBIAS_SP32 << EXPSHIFTBITS_SP32) | (AS_INT(ay) & MANTBITS_SP32));
                ay = ey == -EXPBIAS_SP32 ? ays : ay;
                ey = ey == -EXPBIAS_SP32 ? eys : ey;
            } else {
                ey = (AS_INT(ay) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
                ay = AS_FLOAT((EXPBIAS_SP32 << EXPSHIFTBITS_SP32) | (AS_INT(ay) & MANTBITS_SP32));
            }
            ey = y == 0.0f ? ex : ey;
        }

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
            if (AMD_OPT()) {
                ax = BUILTIN_FLDEXP_F32(ax, bits); 
            } else {
                ax *= AS_FLOAT((EXPBIAS_SP32 + bits) << EXPSHIFTBITS_SP32);
            }
            nb -= bits;
        }

        if (AMD_OPT()) {
            ax = BUILTIN_FLDEXP_F32(ax, nb - bits + 1);
        } else {
            ax *= AS_FLOAT((EXPBIAS_SP32 + nb - bits + 1) << EXPSHIFTBITS_SP32);
        }

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

        if (AMD_OPT()) {
            ax = BUILTIN_FLDEXP_F32(ax, ey);
        } else {
            int ey2 = ey >> 1;
            float xsc1 = AS_FLOAT((EXPBIAS_SP32 + ey2) << EXPSHIFTBITS_SP32);
            float xsc2 = AS_FLOAT((EXPBIAS_SP32 + (ey - ey2)) << EXPSHIFTBITS_SP32);
            ax = (ax * xsc1) * xsc2;
        }

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
        ret = c ? t : ret;
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
        ret = y == 0.0f ? AS_FLOAT(QNANBITPATT_SP32) : ret;
#if defined(COMPILING_REMQUO)
        q7 = y == 0.0f ? 0 : q7;
#endif

        bool c = BUILTIN_CLASS_F32(y, CLASS_QNAN|CLASS_SNAN) |
                 BUILTIN_CLASS_F32(x, CLASS_NINF|CLASS_PINF|CLASS_SNAN|CLASS_QNAN);
        ret = c ? AS_FLOAT(QNANBITPATT_SP32) : ret;
#if defined(COMPILING_REMQUO)
        q7 = c ? 0 : q7;
#endif
    }

#if defined(COMPILING_REMQUO)
    *q7p = q7;
#endif
    return ret;
}

