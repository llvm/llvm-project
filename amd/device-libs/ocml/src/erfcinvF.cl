/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

PUREATTR float
MATH_MANGLE(erfcinv)(float y)
{
    float ret;

    if (y > 0.0625f) {
        ret = MATH_MANGLE(erfinv)(1.0f - y);
    } else {
        float t = MATH_RCP(MATH_SQRT(-MATH_MANGLE(log)(y)));

        float a = MATH_MAD(t,
                      MATH_MAD(t,
                          MATH_MAD(t,
                              MATH_MAD(t, -0x1.50c6bep-3f, 0x1.5c704cp-1f),
                              -0x1.20c9f2p+0f),
                          0x1.61c6bcp-1f),
                      0x1.61f9eap+0);

        float b = MATH_MAD(t, MATH_MAD(t, 1.0f, 0x1.629e50p+0f), 0x1.3d7dacp-3f);

        ret = MATH_DIV(MATH_DIV(0x1.3d8948p-3f, t) + a,  b); 
    }

    if (!FINITE_ONLY_OPT()) {
        ret = (y < 0.0f) | (y > 2.0f) ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = y == 0.0 ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = y == 2.0 ? AS_FLOAT(NINFBITPATT_SP32) : ret;
    }

    return ret;
}

