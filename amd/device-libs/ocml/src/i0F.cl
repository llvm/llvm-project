/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

float
MATH_MANGLE(i0)(float x)
{
    x = BUILTIN_ABS_F32(x);

    float ret;

    if (x < 8.0f) {
        float t = 0.25f * x * x;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, 
                  0x1.38d760p-43f, 0x1.7fd5c6p-38f), 0x1.66ffc8p-31f), 0x1.4ecb6ep-25f),
                  0x1.033c70p-19f), 0x1.233bb2p-14f), 0x1.c71db2p-10f), 0x1.c71c5ep-6f),
                  0x1.000000p-2f), 0x1.000000p+0f);
        ret = MATH_MAD(t, ret, 1.0f);
    } else {
        float t = MATH_FAST_RCP(x);
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, 
                  0x1.c49916p-2f, -0x1.110f5ep-5f), 0x1.2a130ap-5f), 0x1.c68702p-6f),
                  0x1.9890aep-5f), 0x1.988450p-2f);
        float xs = x - 88.0f;
        float e1 = MATH_MANGLE(exp)(x > 88.0f ? xs : x);
        float e2 = x > 88.0f ? 0x1.f1056ep+126f : 1.0f;
        ret = e1 * BUILTIN_RSQRT_F32(x) * ret * e2;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
    }

    return ret;
}

