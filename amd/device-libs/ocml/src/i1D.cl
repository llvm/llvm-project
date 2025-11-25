/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

double
MATH_MANGLE(i1)(double x)
{
    double a = BUILTIN_ABS_F64(x);

    double ret;

    if (a < 8.0) {
        a *= 0.5;
        double t = a * a;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  0x1.fc892c836e80ap-93, 0x1.432352d94a857p-85), 0x1.588ae4f7b7a4ap-77), 0x1.15e96e9231b49p-69),
                  0x1.8bdcb5f2184d1p-62), 0x1.e26237a1e02fep-55), 0x1.f176aca1a831fp-48), 0x1.ab81e97c83e75p-41),
                  0x1.2c9758e3649ffp-34), 0x1.522a43f5ed306p-28), 0x1.27e4fb778d591p-22), 0x1.845c8a0ce4edap-17),
                  0x1.6c16c16c16c26p-12), 0x1.c71c71c71c71cp-8), 0x1.5555555555555p-4), 0x1.0000000000000p-1);
        ret = MATH_MAD(t, a*ret, a);
    } else {
        double t = MATH_RCP(a);
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, 
                  -0x1.c9d8d43214423p+49, 0x1.5c072e12fb4bap+50), -0x1.e26cff438b6f6p+49), 0x1.952224c61a221p+48),
                  -0x1.cdc7c873cf435p+46), 0x1.7b1e32a15fb86p+44), -0x1.d07dbd6696f1cp+41), 0x1.b227934f2ced2p+38),
                  -0x1.39f23e6685444p+35), 0x1.6229383f6f890p+31), -0x1.38bf1ceeee865p+27), 0x1.b01a348b749b8p+22),
                  -0x1.d0e043ef0916ap+17), 0x1.81b06f82cfbacp+12), -0x1.ea879b2a6508bp+6), 0x1.85cffc8d54f52p+0),
                  -0x1.09f107ee0f7e2p-3), -0x1.d61631539fb0dp-5), -0x1.4f1e01d904ebap-5), -0x1.7efc0ced79c58p-5),
                  -0x1.32633e6e0f07ap-3), 0x1.9884533d43674p-2);

        double xs = x - 709.0;
        double e1 = MATH_MANGLE(exp)(x > 709.0 ? xs : x);
        double e2 = x > 709.0 ? 0x1.d422d2be5dc9bp+1022 : 1.0;
        ret = e1 * MATH_MANGLE(rsqrt)(x) * ret * e2;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(a, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? a : ret;
    }

    return BUILTIN_COPYSIGN_F64(ret, x);
}

