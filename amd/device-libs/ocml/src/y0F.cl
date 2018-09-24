/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern float MATH_PRIVATE(sinb)(float, int, float);
extern CONSTATTR float MATH_PRIVATE(bp0)(float);
extern CONSTATTR float MATH_PRIVATE(ba0)(float);

CONSTATTR float
MATH_MANGLE(y0)(float x)
{
    const float b0  = 0.3125f;
    const float b1  = 0.4375f;
    const float b2  = 0.5625f;
    const float b3  = 0.6875f;
    const float b4  = 0.8125f;
    const float b5  = 1.0f;
    const float b6  = 1.25f;
    const float b7  = 1.625f;
    const float b8  = 2.0f;
    const float b9  = 2.53125f;
    const float b10 = 3.0f;
    const float b11 = 3.484375f;
    const float b12 = 4.703125f;
    const float b13 = 6.265625f;
    const float b14 = 7.84375f;
    const float b15 = 9.421875f;
    const float b16 = 10.984375f;
    const float b17 = 12.546875f;

    float ret;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(float, p, M32_Y0);
        float ch, cl;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0f;
                    cl = 0.0f;
                } else if (x < b1) {
                    ch = 0x1.4p-2f;
                    cl = 0.0f;
                    p += 1*9;
                } else if (x < b2) {
                    ch = 0x1.cp-2f;
                    cl = 0.0f;
                    p += 2*9;
                } else if (x < b3) {
                    ch = 0x1.2p-1f;
                    cl = 0.0f;
                    p += 3*9;
                } else {
                    ch = 0x1.6p-1f;
                    cl = 0.0f;
                    p += 4*9;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.c982ecp-1f;
                    cl = -0x1.cafa06p-27f;
                    p += 5*9;
                } else if (x < b6) {
                    ch = 0x1.p+0f;
                    cl = 0.0f;
                    p += 6*9;
                } else if (x < b7) {
                    ch = 0x1.4p+0f;
                    cl = 0.0f;
                    p += 7*9;
                } else {
                    ch = 0x1.ap+0f;
                    cl = 0.0f;
                    p += 8*9;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.193beep+1f;
                    cl = -0x1.6401b8p-24f;
                    p += 9*9;
                } else if (x < b10) {
                    ch = 0x1.44p+1f;
                    cl = 0.0f;
                    p += 10*9;
                } else if (x < b11) {
                    ch = 0x1.8p+1f;
                    cl = 0.0f;
                    p += 11*9;
                } else if (x < b12) {
                    ch = 0x1.fa9534p+1f;
                    cl = 0x1.b30ad4p-24f;
                    p += 12*9;
                } else {
                    ch = 0x1.5b7fe4p+2f;
                    cl = 0x1.d0f606p-23f;
                    p += 13*9;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.c581dcp+2f;
                    cl = 0x1.39c84p-24f;
                    p += 14*9;
                } else if (x < b15) {
                    ch = 0x1.13127ap+3f;
                    cl = 0x1.cc2d36p-22f;
                    p += 15*9;
                } else if (x < b16) {
                    ch = 0x1.471d74p+3f;
                    cl = -0x1.4b7056p-22f;
                    p += 16*9;
                } else {
                    ch = 0x1.77f914p+3f;
                    cl = -0x1.caf37ep-23f;
                    p += 17*9;
                }
            }
        }

        ret = 0.0f;
        if (x < b0) {
            ret = 0x1.45f306p-1f * MATH_MANGLE(j0)(x) * MATH_MANGLE(log)(x);
            x = x*x;
        }

        x = x - ch - cl;
        ret += MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
               MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
               p[8],  p[7]), p[6]), p[5]), p[4]),
               p[3]), p[2]), p[1]), p[0]);
    } else {
        float r = MATH_RCP(x);
        float r2 = r*r;
        float p = MATH_PRIVATE(bp0)(r2) * r;
        ret = 0x1.988454p-1f * BUILTIN_RSQRT_F32(x) * MATH_PRIVATE(ba0)(r2) * MATH_PRIVATE(sinb)(x, 0, p);
        ret = BUILTIN_CLASS_F32(x, CLASS_PINF) ? 0.0f : ret;
    }

    return ret;
}

