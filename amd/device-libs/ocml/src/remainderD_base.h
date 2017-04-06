/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR static double
fnma(double a, double b, double c)
{
    return BUILTIN_FMA_F64(-a, b, c);
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

        ex = BUILTIN_FREXP_EXP_F64(ax) - 1;
        ax = BUILTIN_FLDEXP_F64(BUILTIN_FREXP_MANT_F64(ax), bits);
        ey = BUILTIN_FREXP_EXP_F64(ay) - 1;
        ay = BUILTIN_FLDEXP_F64(BUILTIN_FREXP_MANT_F64(ay), 1);

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
            ax = BUILTIN_FLDEXP_F64(ax, bits); 
            nb -= bits;
        }

        ax = BUILTIN_FLDEXP_F64(ax, nb - bits + 1);

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
        int qneg = (AS_INT2(x).hi ^ AS_INT2(y).hi) >> 31;
        q7 = ((qacc & 0x7f) ^ qneg) - qneg;
#endif
#endif

        ax = BUILTIN_FLDEXP_F64(ax, ey);
        ret =  AS_DOUBLE((AS_ULONG(x) & SIGNBIT_DP64) ^ AS_ULONG(ax));
    } else {
        ret = x;
#if defined(COMPILING_REMQUO)
        q7 = 0;
#endif

#if !defined(COMPILING_FMOD)
        int c = (ay < 0x1.0p+1023 & 2.0*ax > ay) | (ax > 0.5*ay);

        int qsgn = 1 + (((AS_INT2(x).hi ^ AS_INT2(y).hi) >> 31) << 1);
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
        ret = y == 0.0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
#if defined(COMPILING_REMQUO)
        q7 = y == 0.0 ? 0 : q7;
#endif

        bool c = BUILTIN_CLASS_F64(y, CLASS_QNAN|CLASS_SNAN) |
                 BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN);
        ret = c ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
#if defined(COMPILING_REMQUO)
        q7 = c ? 0 : q7;
#endif
    }

#if defined(COMPILING_REMQUO)
    *q7p = q7;
#endif
    return ret;
}

