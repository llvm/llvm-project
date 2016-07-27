/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

PUREATTR float
MATH_PRIVATE(sinhcosh)(float y, int which)
{
    // Tabulated values of sinh(i) and cosh(i) for i = 0,...,36.
    USE_TABLE(float2, p_tbl, M32_SINHCOSH);

    int ind = (int)y;
    ind = (uint)ind > 36U ? 0 : ind;

    float dy = y - ind;
    float dy2 = dy * dy;

    float sdy = MATH_MAD(dy2,
                    MATH_MAD(dy2,
                        MATH_MAD(dy2,
                            MATH_MAD(dy2,
                                MATH_MAD(dy2,
                                    MATH_MAD(dy2, 0x1.b4125ap-41f, 0x1.611cb2p-33f),
                                    0x1.ae6460p-26f),
                                0x1.71de3ap-19f),
                            0x1.a01a02p-13f),
                        0x1.111112p-7f),
                    0x1.555556p-3f);
    sdy = MATH_MAD(sdy, dy*dy2, dy);

    float cdy = MATH_MAD(dy2,
                    MATH_MAD(dy2,
                        MATH_MAD(dy2,
                            MATH_MAD(dy2,
                                MATH_MAD(dy2,
                                    MATH_MAD(dy2, 0x1.9984b8p-37f, 0x1.1ee564p-29f),
                                    0x1.27e506p-22f),
                                0x1.a01a02p-16f),
                            0x1.6c16c2p-10f),
                        0x1.555556p-5f),
                    0x1.000000p-1f);
    cdy = MATH_MAD(cdy, dy2, 1.0f);

    float2 tv = p_tbl[ind];

    float zc = MATH_MAD(tv.s0, sdy, tv.s1 * cdy);
    float zs = MATH_MAD(tv.s1, sdy, tv.s0 * cdy);

    return which == 1 ? zc : zs;
}
