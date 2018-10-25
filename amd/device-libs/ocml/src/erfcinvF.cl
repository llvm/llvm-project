/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(erfcinv)(float y)
{
    float ret;

    if (y > 0.625f) {
        ret = MATH_MANGLE(erfinv)(1.0f - y);
    } else if (y > 0x1.0p-10f) {
        float t = -MATH_MANGLE(log)(y * (2.0f - y)) - 3.125f;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t,
                  0x1.7ee662p-31f, -0x1.3f5a80p-28f), -0x1.b638f0p-26f), 0x1.c9ccc6p-22f),
                  -0x1.72f8aep-20f), -0x1.d21aa6p-17f), 0x1.87aebcp-13f), -0x1.8455d4p-11f),
                  -0x1.8b6ca4p-8f), 0x1.ebd80cp-3f), 0x1.a755e8p+0f);
        ret = MATH_MAD(-y, ret, ret);
    } else {
        float s = MATH_FAST_SQRT(-MATH_MANGLE(log)(y));
        float t = MATH_FAST_RCP(s);

        if (y > 0x1.0p-42f) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t,
                      -0x1.57221ep+0f, 0x1.7f6144p+1f), -0x1.98dd40p+1f), 0x1.2c9066p+1f),
                      -0x1.3a07eap+0f), -0x1.ba546cp-5f), 0x1.004e66p+0f);
        } else {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t,
                      -0x1.649c6ap+4f, 0x1.8fa8fap+4f), -0x1.a112d8p+3f), 0x1.309d98p+2f),
                      -0x1.919488p+0f), -0x1.c084ecp-6f), 0x1.00143ep+0f);
        }
        ret = s * ret;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = (y < 0.0f) | (y > 2.0f) ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = y == 0.0f ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = y == 2.0f ? AS_FLOAT(NINFBITPATT_SP32) : ret;
    }

    return ret;
}

