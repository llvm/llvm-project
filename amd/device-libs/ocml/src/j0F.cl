/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern float MATH_PRIVATE(cosb)(float, int, float);
extern CONSTATTR float MATH_PRIVATE(bp0)(float);
extern CONSTATTR float MATH_PRIVATE(ba0)(float);

float
MATH_MANGLE(j0)(float x)
{
    x = BUILTIN_ABS_F32(x);

    const float b0 = 1.65625f;
    const float b1 = 3.125f;
    const float b2 = 4.6875f;
    const float b3 = 6.265625f;
    const float b4 = 7.84375f;
    const float b5 = 9.421875f;
    const float b6 = 10.984375f;
    const float b7 = 12.578125f;

    float ret;

    if (x <= b7) {
        // Ty to maintain relative accuracy here

        USE_TABLE(float, p, M32_J0);
        float ch, cl;

        if (x <= b3) {
            if (x <= b0) {
                ch = 0x0.000000p+0f;
                cl = 0x0.000000p+0f;
            } else if (x <= b1) {
                ch = 0x1.33d152p+1f;
                cl = 0x1.d2e368p-24f;
                p += 1*9;
            } else if (x <= b2) {
                ch = 0x1.ea7558p+1f;
                cl = -0x1.4a121ep-24f;
                p += 2*9;
            } else {
                ch = 0x1.6148f6p+2f;
                cl = -0x1.34f46ep-24f;
                p += 3*9;
            }
        } else {
            if (x <= b4) {
                ch = 0x1.c0ff60p+2f;
                cl = -0x1.8971b6p-23f;
                p += 4*9;
            } else if (x <= b5) {
                ch = 0x1.14eb56p+3f;
                cl = 0x1.999bdap-22f;
                p += 5*9;
            } else if (x <= b6) {
                ch = 0x1.458d0ep+3f;
                cl = -0x1.e8407ap-22f;
                p += 6*9;
            } else {
                ch = 0x1.795440p+3f;
                cl = 0x1.04e56cp-26f;
                p += 7*9;
            }
        }

        x = x - ch - cl;
        ret = MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              p[8],  p[7]), p[6]), p[5]), p[4]),
              p[3]), p[2]), p[1]), p[0]);
    } else {
        float r = MATH_RCP(x);
        float r2 = r*r;
        float p = MATH_PRIVATE(bp0)(r2) * r;
        ret = 0x1.988454p-1f * BUILTIN_RSQRT_F32(x) * MATH_PRIVATE(ba0)(r2) * MATH_PRIVATE(cosb)(x, 0, p);
        ret = BUILTIN_CLASS_F32(x, CLASS_PINF) ? 0.0f : ret;
    }

    return ret;
}

