/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

float
MATH_MANGLE(i1)(float x)
{
    float a = BUILTIN_ABS_F32(x);

    float ret;

    if (a < 8.0f) {
        a *= 0.5f;
        float t = a * a;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  0x1.882dd2p-40f, 0x1.af97f6p-35f), 0x1.66a3eap-28f), 0x1.251b32p-22f),
                  0x1.84cbb6p-17f), 0x1.6c0d4ap-12f), 0x1.c71d3ap-8f), 0x1.555550p-4f),
                  0x1.000000p-1f);
        ret = MATH_MAD(t, a*ret, a);
    } else {
        float t = MATH_FAST_RCP(a);
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, 
                  -0x1.06de32p-1f, 0x1.043b22p-5f), -0x1.925276p-5f), -0x1.7c15c8p-5f),
                  -0x1.3266ccp-3f), 0x1.988456p-2f);

        float as = a - 88.0f;
        float e1 = MATH_MANGLE(exp)(a > 88.0f ? as : a);
        float e2 = a > 88.0f ? 0x1.f1056ep+126f : 1.0f;
        ret = e1 * BUILTIN_AMDGPU_RSQRT_F32(a) * ret * e2;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F32(a, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? a : ret;
    }

    return BUILTIN_COPYSIGN_F32(ret, x);
}

