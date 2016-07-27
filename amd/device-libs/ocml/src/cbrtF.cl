/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

// Algorithm:
// 
// x = (2^m)*A
// x = (2^m)*(G+g) with (1 <= G < 2) and (g <= 2^(-8))
// x = (2^m)*2*(G/2+g/2)
// x = (2^m)*2*(F+f) with (0.5 <= F < 1) and (f <= 2^(-9))
// 
// Y = (2^(-1))*(2^(-m))*(2^m)*A
// Now, range of Y is: 0.5 <= Y < 1
// 
// F = 0x100 + (first 7 mantissa bits) + (8th mantissa bit)
// Now, range of F is: 128 <= F <= 256
// F = F / 256
// Now, range of F is: 0.5 <= F <= 1
// 
// f = (Y-F), with (f <= 2^(-9))
// 
// cbrt(x) = cbrt(2^m) * cbrt(2) * cbrt(F+f)
// cbrt(x) = cbrt(2^m) * cbrt(2) * cbrt(F) + cbrt(1+(f/F))
// cbrt(x) = cbrt(2^m) * cbrt(2*F) * cbrt(1+r)
// 
// r = (f/F), with (r <= 2^(-8))
// r = f*(1/F) with (1/F) precomputed to avoid division
// 
// cbrt(x) = cbrt(2^m) * cbrt(G) * (1+poly)
// 
// poly = c1*r + c2*(r^2) + c3*(r^3) + c4*(r^4) + c5*(r^5) + c6*(r^6)

PUREATTR INLINEATTR float
MATH_MANGLE(cbrt)(float x)
{
    if (AMD_OPT()) {
        if (DAZ_OPT()) {
            x = BUILTIN_CANONICALIZE_F32(x);
        }
        float ax = BUILTIN_ABS_F32(x);
        if (!DAZ_OPT()) {
            ax = BUILTIN_FLDEXP_F32(ax, BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB) ? 24 : 0);
        }
        float z = BUILTIN_EXP2_F32(0x1.555556p-2f * BUILTIN_LOG2_F32(ax));
        z = MATH_MAD(MATH_MAD(MATH_FAST_RCP(z*z), -ax, z), -0x1.555556p-2f, z);
        if (!DAZ_OPT()) {
            z = BUILTIN_FLDEXP_F32(z, BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB) ? -8 : 0);
        }
        z = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN|CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER) ? x : z;
        return BUILTIN_COPYSIGN_F32(z, x);
    } else {
        USE_TABLE(float2, p_cbrt, M32_CBRT);
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
        float mf = AS_FLOAT((m3 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);

        uint indx = (xi & 0x007f0000) + ((xi & 0x00008000) << 1);
        float f = AS_FLOAT((xi & MANTBITS_SP32) | 0x3f000000) - AS_FLOAT(indx | 0x3f000000);

        indx >>= 16;
        float r = f * p_log_inv[indx];
        float poly = MATH_MAD(MATH_MAD(r, 0x1.f9add4p-5f, -0x1.c71c72p-4f), r*r, r * 0x1.555556p-2f);

        // This could also be done with a 5-element table
        float remH = 0x1.428000p-1f;
        float remT = 0x1.45f31ap-14f;

        remH = rem == -1 ? 0x1.964000p-1f : remH;
        remT = rem == -1 ? 0x1.fea53ep-13f : remT;

        remH = rem ==  0 ? 0x1.000000p+0f : remH;
        remT = rem ==  0 ? 0x0.000000p+0f  : remT;

        remH = rem ==  1 ? 0x1.428000p+0f : remH;
        remT = rem ==  1 ? 0x1.45f31ap-13f : remT;

        remH = rem ==  2 ? 0x1.964000p+0f : remH;
        remT = rem ==  2 ? 0x1.fea53ep-12f : remT;

        float2 tv = p_cbrt[indx];
        float cbrtH = tv.s0;
        float cbrtT = tv.s1;

        float bH = cbrtH * remH;
        float bT = MATH_MAD(cbrtH, remT, MATH_MAD(cbrtT, remH, cbrtT*remT));

        float z = MATH_MAD(poly, bH, MATH_MAD(poly, bT, bT)) + bH;
        z *= mf;
        z = AS_FLOAT(AS_UINT(z) | xsign);

        z = axi >= EXPBITS_SP32 | axi == 0 ? x : z;

        return z;
    }
}

