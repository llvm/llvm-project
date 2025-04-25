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
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 16.0f) {
        float n, d;
        float y = x;
        if (x > 0.0f) {
            n = 1.0f;
            while (y > 2.5f) {
                n = MATH_MAD(n, y, -n);
                y = y - 1.0f;
                n = MATH_MAD(n, y, -n);
                y = y - 1.0f;
            }
            if (y > 1.5f) {
                n = MATH_MAD(n, y, -n);
                y = y - 1.0f;
            }
            if (x >= 0.5f)
                y = y - 1.0f;
            d = x < 0.5f ? x : 1.0f;
        } else {
            d = x;
            while (y < -1.5f) {
                d = MATH_MAD(d, y, d);
                y = y + 1.0f;
                d = MATH_MAD(d, y, d);
                y = y + 1.0f;
            }
            if (y < -0.5f) {
                d = MATH_MAD(d, y, d);
                y = y + 1.0f;
            }
            n = 1.0f;
        }
        float qt = MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y,
                   MATH_MAD(y, MATH_MAD(y,
                       0x1.d5a56ep-8f, -0x1.4dcb00p-7f), -0x1.59c03ap-5f), 0x1.55405ap-3f),
                       -0x1.5810f2p-5f), -0x1.4fcfd6p-1f), 0x1.2788ccp-1f);
        ret = MATH_DIV(n, MATH_MAD(d, y*qt, d));
        ret = x == 0.0f ? BUILTIN_COPYSIGN_F32(PINF_F32, x) : ret;
        ret = x < 0.0f && BUILTIN_FRACTION_F32(x) == 0.0f ? QNAN_F32 : ret;
    } else {
        const float sqrt2pi = 0x1.40d932p+1f;
        const float sqrtpiby2 = 0x1.40d932p+0f;

        float t1 = MATH_MANGLE(powr)(ax, MATH_MAD(ax, 0.5f, -0.25f));
        float t2 = MATH_MANGLE(exp)(-ax);
        float xr = MATH_FAST_RCP(ax);
        float p = MATH_MAD(xr, MATH_MAD(xr, 0x1.96d7e4p-9f, 0x1.556652p-4f), 0x1.fffff8p-1f);
        if (x > 0.0f) {
            float g = sqrt2pi*t2*t1*t1*p;
            ret = x >  0x1.18521ep+5f ? PINF_F32 : g;
        } else {
            float s = -x * MATH_MANGLE(sinpi)(x);
            if (x > -30.0f)
                ret = MATH_DIV(sqrtpiby2, s*t2*t1*t1*p);
            else if (x > -41.0f)
                ret = MATH_DIV(MATH_DIV(sqrtpiby2, t2*t1*p), s*t1);
            else
                ret = BUILTIN_COPYSIGN_F32(0.0f, s);
            ret = BUILTIN_FRACTION_F32(x) == 0.0f || BUILTIN_ISNAN_F32(x) ? QNAN_F32 : ret;
        }
    }

    return ret;
}

