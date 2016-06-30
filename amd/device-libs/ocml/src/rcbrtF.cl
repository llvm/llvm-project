
#include "mathF.h"

// Algorithm: see cbrt

PUREATTR INLINEATTR float
MATH_MANGLE(rcbrt)(float x)
{
    USE_TABLE(float2, p_rcbrt, M32_RCBRT);
    USE_TABLE(float, p_log_inv, M32_LOG_INV);

    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
    }

    uint xi = AS_UINT(x);
    uint axi = xi & EXSIGNBIT_SP32;
    uint xsign = axi ^ xi;
    xi = axi;

    int m = (int)(xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;

    if (!DAZ_OPT()) {
        // Treat subnormals
        uint xis = AS_UINT(AS_FLOAT(xi | 0x3f800000) - 1.0f);
        int ms = (xis >> EXPSHIFTBITS_SP32) - 253;
        int c = m == -127;
        xi = c ? xis : xi;
        m = c ? ms : m;
    }

    int m3 = m / 3;
    int rem = m - m3*3;
    float mf = AS_FLOAT((EXPBIAS_SP32 - m3) << EXPSHIFTBITS_SP32);

    uint indx = (xi & 0x007f0000) + ((xi & 0x00008000) << 1);
    float f = AS_FLOAT((xi & MANTBITS_SP32) | 0x3f000000) - AS_FLOAT(indx | 0x3f000000);

    indx >>= 16;
    float r = f * p_log_inv[indx];
    float poly = MATH_MAD(MATH_MAD(r, -0x1.61f9aep-3f, 0x1.c71c72p-3f), r*r, r * -0x1.555556p-2f);

    // This could also be done with a 5-element table
    float remH = 0x1.964000p+0f;
    float remT = 0x1.fea53ep-12f;

    remH = rem == -1 ? 0x1.428000p+0f : remH;
    remT = rem == -1 ? 0x1.45f31ap-13f : remT;

    remH = rem ==  0 ? 0x1.000000p+0f : remH;
    remT = rem ==  0 ? 0x0.000000p+0f  : remT;

    remH = rem ==  1 ? 0x1.964000p-1f : remH;
    remT = rem ==  1 ? 0x1.fea53ep-13f : remT;

    remH = rem ==  2 ? 0x1.428000p-1f : remH;
    remT = rem ==  2 ? 0x1.45f31ap-14f : remT;

    float2 tv = p_rcbrt[indx];
    float rcbrtH = tv.s0;
    float rcbrtT = tv.s1;

    float bH = rcbrtH * remH;
    float bT = MATH_MAD(rcbrtH, remT, MATH_MAD(rcbrtT, remH, rcbrtT*remT));

    float z = MATH_MAD(poly, bH, MATH_MAD(poly, bT, bT)) + bH;
    z *= mf;

    if (!FINITE_ONLY_OPT()) {
        z = axi == 0 ? AS_FLOAT(PINFBITPATT_SP32) : z;
        z = axi == PINFBITPATT_SP32 ? 0.0f : z;
        z = axi > PINFBITPATT_SP32 ? axi : z;
    }
    
    z = AS_FLOAT(AS_UINT(z) | xsign);

    return z;
}

