/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

PUREATTR float
MATH_MANGLE(erf)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 2.2f) {
        USE_TABLE(float, c, M32_ERF);
        float t = ax * ax;
        int k = (int)t;
        c += k * 11;
        t = BUILTIN_FRACTION_F32(t);

        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              c[0], c[1]), c[2]), c[3]), c[4]), c[5]),
                    c[6]), c[7]), c[8]), c[9]), c[10]) * ax;
    } else if (ax < 3.92f) {
        int k = (int)ax;
        float t = BUILTIN_FRACTION_F32(ax);

        float c0, c1, c2, c3;
        if (k == 2) {
            c0 =  0x1.0b3bdep-7;
            c1 = -0x1.cb8014p-7;
            c2 = -0x1.946a96p-3;
            c3 =  0x1.6e23d8p-1;
        } else { // k == 3
            c0 =  0x1.b13bf6p-8;
            c1 =  0x1.2a08fcp-7;
            c2 = -0x1.9ce0d8p-3;
            c3 =  0x1.05fd26p-1;
        }

        float y = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, c0, c1), c2), c3);
        y *= y;
        y *= y;
        y *= y;
        ret = MATH_MAD(-y, y, 1.0f);
    } else {
        ret = 1.0f;
    }

    ret = BUILTIN_COPYSIGN_F32(ret, x);
    ret = BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN) ? x : ret;
    return ret;
}

