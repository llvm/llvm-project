/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern float MATH_PRIVATE(sinb)(float, int, float);
extern CONSTATTR float MATH_PRIVATE(bp1)(float);
extern CONSTATTR float MATH_PRIVATE(ba1)(float);

CONSTATTR float
MATH_MANGLE(y1)(float x)
{
    const float b0 = 0.5f;
    const float b1 = 0.625f;
    const float b2 = 0.75f;
    const float b3 = 0.9375f;
    const float b4 = 1.21875f;
    const float b5 = 1.53125f;
    const float b6 = 1.84375f;
    const float b7 = 2.078125f;
    const float b8 = 2.3125f;
    const float b9 = 2.734375f;
    const float b10 = 3.15625f;
    const float b11 = 4.203125f;
    const float b12 = 4.6875f;
    const float b13 = 6.1875f;
    const float b14 = 7.76953125f;
    const float b15 = 9.359375f;
    const float b16 = 10.9375f;
    const float b17 = 12.5625f;

    float ret;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(float, p, M32_Y1);
        float ch, cl;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0f;
                    cl = 0.0f;
                    p += 0*9;
                } else if (x < b1) {
                    ch = 0x1.0p-1f;
                    cl = 0.0f;
                    p += 1*9;
                } else if (x < b2) {
                    ch = 0x1.4p-1f;
                    cl = 0.0f;
                    p += 2*9;
                } else if (x < b3) {
                    ch = 0x1.8p-1f;
                    cl = 0.0f;
                    p += 3*9;
                } else {
                    ch = 0x1.ep-1f;
                    cl = 0.0f;
                    p += 4*9;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.38p+0f;
                    cl = 0.0f;
                    p += 5*9;
                } else if (x < b6) {
                    ch = 0x1.88p+0f;
                    cl = 0.0f;
                    p += 6*9;
                } else if (x < b7) {
                    ch = 0x1.d8p+0f;
                    cl = 0.0f;
                    p += 7*9;
                } else {
                    ch = 0x1.193beep+1f;
                    cl = -0x1.6401b8p-24f;
                    p += 8*9;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.28p+1f;
                    cl = 0.0f;
                    p += 9*9;
                } else if (x < b10) {
                    ch = 0x1.5ep+1f;
                    cl = 0.0f;
                    p += 10*9;
                } else if (x < b11) {
                    ch = 0x1.d76d4ap+1f;
                    cl = 0x1.ff742ep-24f;
                    p += 11*9;
                } else if (x < b12) {
                    ch = 0x1.0dp+2f;
                    cl = 0.0f;
                    p += 12*9;
                } else {
                    ch = 0x1.5b7fe4p+2f;
                    cl = 0x1.d0f606p-23f;
                    p += 13*9;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.bc418ap+2f;
                    cl = -0x1.f4ef56p-23f;
                    p += 14*9;
                } else if (x < b15) {
                    ch = 0x1.13127ap+3f;
                    cl = 0x1.cc2d36p-22f;
                    p += 15*9;
                } else if (x < b16) {
                    ch = 0x1.43f2eep+3f;
                    cl = 0x1.47a32p-23f;
                    p += 16*9;
                } else {
                    ch = 0x1.77f914p+3f;
                    cl = -0x1.caf37ep-23f;
                    p += 17*9;
                }
            }
        }

        float x2 = x*x;
        float xs = x - ch - cl;
        float t = x < b0 ? x2 : xs;

        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              p[8],  p[7]), p[6]), p[5]), p[4]),
              p[3]), p[2]), p[1]), p[0]);

        if (x < b0) {
            const float twobypi = 0x1.45f306p-1f;
            if (x < 0x1.0p-20f)
                ret = MATH_DIV(-twobypi, BUILTIN_ABS_F32(x));
            else
                ret = MATH_MAD(ret, x, twobypi*(MATH_MANGLE(j1)(x) * MATH_MANGLE(log)(x) - MATH_RCP(x)));
            ret = x < 0.0f ? QNAN_F32 : ret;
        }
    } else {
        float r = MATH_RCP(x);
        float r2 = r*r;
        float p = MATH_PRIVATE(bp1)(r2) * r;
        ret = 0x1.988454p-1f * BUILTIN_AMDGPU_RSQRT_F32(x) * MATH_PRIVATE(ba1)(r2) * MATH_PRIVATE(sinb)(x, 1, p);
        ret = x == PINF_F32 ? 0.0f : ret;
    }

    return ret;
}

