/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

INLINEATTR CONSTATTR double
MATH_PRIVATE(tanred2)(double x, double xx, int sel)
{
    const double piby4_lead = 0x1.921fb54442d18p-1;
    const double piby4_tail = 0x1.1a62633145c06p-55;

    // In order to maintain relative precision transform using the identity:
    // tan(pi/4-x) = (1-tan(x))/(1+tan(x)) for arguments close to pi/4.
    // Similarly use tan(x-pi/4) = (tan(x)-1)/(tan(x)+1) close to -pi/4.

    bool ca = x >  0.68;
    bool cb = x < -0.68;
    double transform = ca ?  1.0 : 0.0;
    transform = cb ? -1.0 : transform;

    double tx = MATH_MAD(-transform, x, piby4_lead) + MATH_MAD(-transform, xx, piby4_tail);

    bool c = ca | cb;
    x = c ? tx : x;
    xx = c ? 0.0 : xx;

    // Core Remez [2,3] approximation to tan(x+xx) on the interval [0,0.68].
    double t1 = x;
    double r = MATH_MAD(x*xx, 2.0, x*x);

    double a = MATH_MAD(r,
                        MATH_MAD(r, 0x1.d5daf289c385ap-13, -0x1.77c24c7569abbp-6),
                        0x1.7d50f6638564ap-2);

    double b = MATH_MAD(r,
                        MATH_MAD(r,
                                 MATH_MAD(r, -0x1.e7517ef6d98f8p-13, 0x1.ab0f4f80a0acfp-6),
                                 -0x1.08046499eb90fp-1),
                        0x1.1dfcb8caa40b8p+0);

    double t2 = MATH_MAD(MATH_FAST_DIV(a, b), x*r, xx);

    double tp = t1 + t2;
    double ret;

    if (c) {
        if (sel)
            ret = transform * (MATH_FAST_DIV(2.0*tp, tp - 1.0) - 1.0);
        else
            ret = transform * (1.0 - MATH_FAST_DIV(2.0*tp, 1.0 + tp));
    } else {
        if (sel) {
            // Compute -1.0/(t1 + t2) accurately
            double z1 = AS_DOUBLE(AS_LONG(tp) & 0xffffffff00000000L);
            double z2 = t2 - (z1 - t1);
            double trec = -MATH_FAST_RCP(tp);
            double trec_top = AS_DOUBLE(AS_LONG(trec) & 0xffffffff00000000L);
            ret = MATH_MAD(MATH_MAD(trec_top, z2, MATH_MAD(trec_top, z1, 1.0)), trec, trec_top);
        } else {
            ret = tp;
        }
    }

    return ret;
}

