
#include "mathF.h"

PUREATTR float
MATH_MANGLE(erfcx)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 1.0f) {
        ret = MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x,
                  -0x1.77d64p-11f, 0x1.269372p-9f),
                  -0x1.c27dd4p-9f), 0x1.d3d3c4p-8f),
                  -0x1.35d6cap-6f), 0x1.5bb082p-5f),
                  -0x1.60e46ep-4f), 0x1.54d3e4p-3f),
                  -0x1.340edap-2f), 0x1.00049ap-1f),
                  -0x1.81286p-1f), 0x1.ffffcap-1f),
                  -0x1.20dd7p+0f), 0x1.0p+0);
    } else if (ax < 32.0f) {
        double t = MATH_DIV(ax - 4.0f, ax + 4.0f);
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t,
                  0.00416076401f, -0.0167250745f),
                  0.0378070959f), -0.0661972834f),
                  0.0935599947f), -0.101052745f),
                  0.0681148962f), 0.0153801711f),
                  -0.139621619f), 1.23299511f);

        ret = MATH_DIV(ret, MATH_MAD(ax, 2.0f, 1.0f));
    } else {
        const float one_over_sqrtpi = 0x1.20dd76p-1f;
        double z = MATH_RCP(x * x);
        ret =  MATH_DIV(one_over_sqrtpi, x) * MATH_MAD(z, MATH_MAD(z, 0.375f, -0.5f), 1.0f);
    }

    if (x <= -1.0f) {
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

        ret = MATH_MANGLE(exp)(x2h) * MATH_MANGLE(exp)(x2l) * 2.0f - ret;
        ret = x < -10.0f ? AS_FLOAT(PINFBITPATT_SP32) : ret;
    }

    return ret;
}

