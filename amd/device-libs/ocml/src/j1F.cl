/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern float MATH_PRIVATE(cosb)(float, int, float);
extern CONSTATTR float MATH_PRIVATE(bp1)(float);
extern CONSTATTR float MATH_PRIVATE(ba1)(float);

float
MATH_MANGLE(j1)(float x)
{
    const float b0 =  1.09375f;
    const float b1 =  2.84375f;
    const float b2 =  4.578125f;
    const float b3 =  6.171875f;
    const float b4 =  7.78125f;
    const float b5 =  9.359375f;
    const float b6 = 10.953125f;
    const float b7 = 12.515625f;

    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax <= b7) {
        // Ty to maintain relative accuracy here

        USE_TABLE(float, p, M32_J1);
        float ch, cl;

        if (ax <= b3) {
            if (ax <= b0) {
                ch = 0.0f;
                cl = 0.0f;
            } else if (ax <= b1) {
                ch = 0x1.d757d2p+0f;
                cl = -0x1.375c60p-32f;
                p += 1*9;
            } else if (ax <= b2) {
                ch = 0x1.ea7558p+1f;
                cl = -0x1.4a121ep-24f;
                p += 2*9;
            } else {
                ch = 0x1.55365cp+2f;
                cl = -0x1.fe6dccp-25f;
                p += 3*9;
            }
        } else {
            if (ax <= b4) {
                ch = 0x1.c0ff60p+2f;
                cl = -0x1.8971b6p-23f;
                p += 4*9;
            } else if (ax <= b5) {
                ch = 0x1.112980p+3f;
                cl = 0x1.e17114p-22f;
                p += 5*9;
            } else if (ax <= b6) {
                ch = 0x1.458d0ep+3f;
                cl = -0x1.e8407ap-22f;
                p += 6*9;
            } else {
                ch = 0x1.769798p+3f;
                cl = -0x1.a04694p-23f;
                p += 7*9;
            }
        }

        ax = ax - ch - cl;
        ret = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
              MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
              p[8],  p[7]), p[6]), p[5]), p[4]),
              p[3]), p[2]), p[1]), p[0]);
    } else {
        float r = MATH_RCP(ax);
        float r2 = r*r;
        float p = MATH_PRIVATE(bp1)(r2) * r;
        ret = 0x1.988454p-1f * BUILTIN_AMDGPU_RSQRT_F32(ax) * MATH_PRIVATE(ba1)(r2) * MATH_PRIVATE(cosb)(ax, 1, p);
        ret = ax == PINF_F32 ? 0.0f : ret;
    }

    if (x < 0.0f)
        ret = -ret;

    return ret;
}

