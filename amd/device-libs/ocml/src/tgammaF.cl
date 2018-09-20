/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(tgamma)(float x)
{
    const float pi = 0x1.921fb6p+1f;
    const float sqrt2pi = 0x1.40d932p+1f;
    const float sqrtpiby2 = 0x1.40d932p+0f;
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax > 0.0125f) {
        // For x < 3, push to larger value using gamma(x) = gamma(x+1) / x
        float d = 1.0f;
        if (x < 1.0f) {
            d = MATH_MAD((ax + 3.0f), ax, 2.0f) * ax;
            ax = ax + 3.0f;
        } else if (ax < 2.0f) {
            d = MATH_MAD(ax, ax, ax);
            ax = ax + 2.0f;
        } else if (ax < 3.0f) {
            d = ax;
            ax = ax + 1.0f;
        }

        // x^x e^-x (1 + poly(1/x)) sqrt(twopi / x) / d
        // Split x^x into a product since it overflows faster than gamma(x)
        float t1 = MATH_MANGLE(powr)(ax, MATH_MAD(ax, 0.5f, -0.25f));
        float t2 = MATH_MANGLE(exp)(-ax);
        float xr = MATH_FAST_RCP(ax);
        float pt = xr*MATH_MAD(xr, MATH_MAD(xr, -139.0f/51840.0f, 1.0f/288.0f) , 1.0f/12.0f);
        if (x > 0.0f) {
            float p = sqrt2pi*t2*t1*t1 * MATH_FAST_RCP(d);
            ret =  MATH_MAD(p, pt, p);
            ret = x >  0x1.18521ep+5f ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        } else {
            float s = MATH_MANGLE(sinpi)(x);
            float p = s*x*t2*t1*t1;
            ret = MATH_DIV(-sqrtpiby2*d,  MATH_MAD(p, pt, p));
            ret = x < -42.0f ? 0.0f : ret;
            ret = BUILTIN_FRACTION_F32(x) == 0.0f ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        }
    } else {
        float p =  MATH_MAD(ax,
                       MATH_MAD(ax,
                           MATH_MAD(ax, 0.95758557809281868459f, -0.90729132749086121523f),
                           0.98905552641429454945f),
                       -0.57721566471808262829f);
        if (BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR))
            ret = MATH_RCP(ax) + p;
        else
            ret = MATH_DIV(pi, MATH_MAD(ax, p, 1.0f) * MATH_MANGLE(sinpi)(x));
    }
    return ret;
}

