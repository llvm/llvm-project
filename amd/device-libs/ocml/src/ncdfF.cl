/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

PUREATTR float
MATH_MANGLE(ncdf)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 1.0f) {
        float t = x*x;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, 
                  0x1.20379ep-21f, -0x1.3727aep-17f), 0x1.e3af2ep-14f), -0x1.373d8cp-10f),
                  0x1.46d034p-7f), -0x1.105838p-4f), 0x1.988454p-2f);
        ret = MATH_MAD(x, ret, 0.5f);
    } else {
        if (ax < 2.0f) {
            float t = ax - 2.0f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.5fe456p-13f, 0x1.0af2dep-13f), -0x1.6b7222p-10f), 0x1.2217eap-9f),
                      0x1.25dd12p-8f), -0x1.ba588ep-6f), 0x1.ba4a9ep-5f), -0x1.ba4b4ap-5f),
                      0x1.74bcf8p-6f);
        } else {
            float t = MATH_FAST_DIV(ax - 4.0f, ax + 4.0f);
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.490cd2p-10f, -0x1.3013c0p-10f), -0x1.5657e6p-7f), 0x1.6b5024p-5f),
                      -0x1.8634a0p-4f), 0x1.08017ap-3f), -0x1.8ac59ep-4f), -0x1.78fbc6p-6f),
                      0x1.b30b52p-1f);

            float x2h, x2l;
            if (HAVE_FAST_FMA32()) {
                x2h = ax * ax;
                x2l = BUILTIN_FMA_F32(ax, ax, -x2h);
            } else {
                float xh = AS_FLOAT(AS_UINT(ax) & 0xfffff000U);
                float xl = ax - xh;
                x2h = xh*xh;
                x2l = (ax + xh)*xl;
            }

            ret = MATH_FAST_DIV(ret, MATH_MAD(ax, 2.0f, 1.0f)) * MATH_MANGLE(exp)(-0.5f*x2l) * MATH_MANGLE(exp)(-0.5f*x2h);
            ret = BUILTIN_CLASS_F32(ax, CLASS_PINF) ? 0.0f : ret;
        }

        float retp = 1.0f - ret;
        ret = x > 0.0f ? retp : ret;
    }

    return ret;
}

