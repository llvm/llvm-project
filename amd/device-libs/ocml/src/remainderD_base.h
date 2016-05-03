
#include "mathD.h"

#define FULL_MUL(A, B, CHI, CLO) \
    do { \
        double __ha = as_double(as_ulong(A) & 0xfffffffff8000000UL); \
        double __ta = A - __ha; \
        double __hb = as_double(as_ulong(B) & 0xfffffffff8000000UL); \
        double __tb = B - __hb; \
        CHI = A * B; \
        CLO = MATH_MAD(__ta, __tb, MATH_MAD(__ta, __hb, MATH_MAD(__ha, __tb, MATH_MAD(__ha, __hb, -CHI)))); \
    } while (0)

static inline double
fnma(double a, double b, double c)
{
    double d;
    if (HAVE_FAST_FMA64()) {
        d = BUILTIN_FMA_F64(-a, b, c);
    } else {
        double h, t;
        FULL_MUL(a, b, h, t);
        d = c - h;
        d = (((c - d) - h) - t) + d;
    }
    return d;
}

#if defined(COMPILING_FMOD)
CONSTATTR double
MATH_MANGLE(fmod)(double x, double y)
#elif defined(COMPILING_REMQUO)
double
MATH_MANGLE(remquo)(double x, double y, __private int *q7p)
#else
CONSTATTR double
MATH_MANGLE(remainder)(double x, double y)
#endif
{
    // How many bits of the quotient per iteration
    const int bits = 26;

    double ax = BUILTIN_ABS_F64(x);
    double ay = BUILTIN_ABS_F64(y);
    double ret;
#if defined(COMPILING_REMQUO)
    int q7;
#endif

    if (ax > ay) {
        int ex, ey;

        if (AMD_OPT()) {
            ex = BUILTIN_FREXP_EXP_F64(ax) - 1;
            ax = BUILTIN_FLDEXP_F64(BUILTIN_FREXP_MANT_F64(ax), bits);
            ey = BUILTIN_FREXP_EXP_F64(ay) - 1;
            ay = BUILTIN_FLDEXP_F64(BUILTIN_FREXP_MANT_F64(ay), 1);
        } else {
            ex = (as_int2(ax).hi >> 20) - EXPBIAS_DP64;
            int exs = -1011 - (int)MATH_CLZL(as_ulong(ax));
            double axs = as_double(((ulong)(EXPBIAS_DP64+bits-1) << EXPSHIFTBITS_DP64) |
                               ((as_ulong(ax) << (-1022 - exs)) & MANTBITS_DP64));
            ax = as_double(((ulong)(EXPBIAS_DP64+bits-1) << EXPSHIFTBITS_DP64) | (as_ulong(ax) & MANTBITS_DP64));
            ax = ex == -EXPBIAS_DP64 ? axs : ax;
            ex = ex == -EXPBIAS_DP64 ? exs : ex;
            ax = x == 0.0 ? 0.0 : ax;
            ex = x == 0.0 ? 0 : ex;

            ey = (as_int2(ay).hi >> 20) - EXPBIAS_DP64;
            int eys = -1011 - (int)MATH_CLZL(as_ulong(ay));
            double ays = as_double(((ulong)EXPBIAS_DP64 << EXPSHIFTBITS_DP64) | (as_ulong(ay) << (-1022 - eys)));
            ay = as_double(((ulong)EXPBIAS_DP64 << EXPSHIFTBITS_DP64) | (as_ulong(ay) & MANTBITS_DP64));
            ay = ey == -EXPBIAS_DP64 ? ays : ay;
            ey = ey == -EXPBIAS_DP64 ? eys : ey;
            ey = y == 0.0 ? ex : ey;
        }

        int nb = ex - ey;
        double ayinv = MATH_RCP(ay);

#if !defined(COMPILING_FMOD)
        int qacc = 0;
#endif

        while (nb > bits) {
            double q = BUILTIN_RINT_F64(ax * ayinv);
            ax = fnma(q, ay, ax);
            int clt = ax < 0.0;
            double axp = ax + ay;
            ax = clt ? axp : ax;
#if defined(COMPILING_REMQUO)
            int iq = (int)q;
            iq -= clt;
            qacc = (qacc << bits) | iq;
#endif
            if (AMD_OPT()) {
                ax = BUILTIN_FLDEXP_F64(ax, bits); 
            } else {
                ax *= as_double((ulong)(EXPBIAS_DP64 + bits) << EXPSHIFTBITS_DP64);
            }
            nb -= bits;
        }

        if (AMD_OPT()) {
            ax = BUILTIN_FLDEXP_F64(ax, nb - bits + 1);
        } else {
            ax *= as_double((ulong)(EXPBIAS_DP64 + nb - bits + 1) << EXPSHIFTBITS_DP64);
        }

        // Final iteration
        {
            double q = BUILTIN_RINT_F64(ax * ayinv);
            ax = fnma(q, ay, ax);
            int clt = ax < 0.0;
            double axp = ax + ay;
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
        int aq = (2.0*ax > ay) | ((qacc & 0x1) & (2.0f*ax == ay));
        ax = ax - (aq ? ay : 0.0f);
#if defined(COMPILING_REMQUO)
        qacc += aq;
        int qneg = (as_int2(x).hi ^ as_int2(y).hi) >> 31;
        q7 = ((qacc & 0x7f) ^ qneg) - qneg;
#endif
#endif

        if (AMD_OPT()) {
            ax = BUILTIN_FLDEXP_F64(ax, ey);
        } else {
            int ey2 = ey >> 1;
            double xsc1 = as_double((ulong)(EXPBIAS_DP64 + ey2) << EXPSHIFTBITS_DP64);
            double xsc2 = as_double((ulong)(EXPBIAS_DP64 + (ey - ey2)) << EXPSHIFTBITS_DP64);
            ax = (ax * xsc1) * xsc2;
        }

        ret =  as_double((as_ulong(x) & SIGNBIT_DP64) ^ as_ulong(ax));
    } else {
        ret = x;
#if defined(COMPILING_REMQUO)
        q7 = 0;
#endif

#if !defined(COMPILING_FMOD)
        int c = (ay < 0x1.0p+1023 & 2.0*ax > ay) | (ax > 0.5*ay);

        int qsgn = 1 + (((as_int2(x).hi ^ as_int2(y).hi) >> 31) << 1);
        double t = MATH_MAD(y, -(double)qsgn, x);
        ret = c ? t : ret;
#if defined(COMPILING_REMQUO)
        q7 = c ? qsgn : q7;
#endif
#endif
        ret = ax == ay ? BUILTIN_COPYSIGN_F64(0.0, x) : ret;
#if defined(COMPILING_REMQUO)
        q7 = ax == ay ? qsgn : q7;
#endif
    }

    if (!FINITE_ONLY_OPT()) {
        ret = y == 0.0 ? as_double(QNANBITPATT_DP64) : ret;
#if defined(COMPILING_REMQUO)
        q7 = y == 0.0 ? 0 : q7;
#endif

        bool c = BUILTIN_CLASS_F64(y, CLASS_QNAN|CLASS_SNAN) |
                 BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN);
        ret = c ? as_double(QNANBITPATT_DP64) : ret;
#if defined(COMPILING_REMQUO)
        q7 = c ? 0 : q7;
#endif
    }

#if defined(COMPILING_REMQUO)
    *q7p = q7;
#endif
    return ret;
}

