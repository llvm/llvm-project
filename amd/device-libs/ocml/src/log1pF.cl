
#include "mathF.h"

PUREATTR float
MATH_MANGLE(log1p)(float x)
{
    USE_TABLE(float2, p_log, M32_LOGE);
    USE_TABLE(float, p_inv, M32_LOG_INV);

    float w = x;
    uint ux = AS_UINT(x);
    uint ax = ux & EXSIGNBIT_SP32;

    // |x| < 2^-4
    float u2 = MATH_FAST_DIV(x, 2.0f + x);
    float u = u2 + u2;
    float v = u * u;
    // 2/(5 * 2^5), 2/(3 * 2^3)
    float zsmall = MATH_MAD(-u2, x, MATH_MAD(v, 0x1.99999ap-7f, 0x1.555556p-4f) * v * u) + x;

    // |x| >= 2^-4
    x = x + 1.0f;
    ux = AS_UINT(x);

    int m = (int)((ux >> EXPSHIFTBITS_SP32) & 0xff) - EXPBIAS_SP32;
    float mf = (float)m;
    uint indx = (ux & 0x007f0000) + ((ux & 0x00008000) << 1);
    float F = AS_FLOAT(indx | 0x3f000000);

    // x > 2^24
    float fg24 = F - AS_FLOAT(0x3f000000 | (ux & MANTBITS_SP32));

    // x <= 2^24
    uint xhi = ux & 0xffff8000;
    float xh = AS_FLOAT(xhi);
    float xt = (1.0f - xh) + w;
    uint xnm = ((~(xhi & 0x7f800000)) - 0x00800000) & 0x7f800000;
    xt = xt * AS_FLOAT(xnm) * 0.5f;
    float fl24 = F - AS_FLOAT(0x3f000000 | (xhi & MANTBITS_SP32)) - xt;

    float f = mf > 24.0f ? fg24 : fl24;

    indx = indx >> 16;
    float r = f * p_inv[indx];

    // 1/3, 1/2
    float poly = MATH_MAD(MATH_MAD(r, 0x1.555556p-2f, 0x1.0p-1f), r*r, r);

    const float LOG2_HEAD = 0x1.62e000p-1f;   // 0.693115234
    const float LOG2_TAIL = 0x1.0bfbe8p-15f;  // 0.0000319461833

    float2 tv = p_log[indx];
    float z1 = MATH_MAD(mf, LOG2_HEAD, tv.s0);
    float z2 = MATH_MAD(mf, LOG2_TAIL, -poly) + tv.s1;
    float z = z1 + z2;

    z = ax < 0x3d800000U ? zsmall : z;

    // Edge cases
    if (!FINITE_ONLY_OPT()) {
        z = ax >= PINFBITPATT_SP32 ? w : z;
        z = w  < -1.0f ? AS_FLOAT(QNANBITPATT_SP32) : z;
        z = w == -1.0f ? AS_FLOAT(NINFBITPATT_SP32) : z;
    }

    return z;
}

