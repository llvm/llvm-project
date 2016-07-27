/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// Much of this implementation is adapted from Sun libm
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


#include "mathF.h"

PUREATTR float
MATH_MANGLE(erfc)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 0.84375f) {
        float z = x * x;
        float r, s;

        r = MATH_MAD(z,
                MATH_MAD(z,
                    MATH_MAD(z,
                        MATH_MAD(z, -2.37630166566502e-05f, -5.77027029648944e-03f),
                        -2.84817495755985e-02f),
                    -3.25042107247001e-01f),
                1.28379167095513e-01f);

        s = MATH_MAD(z,
                MATH_MAD(z,
                    MATH_MAD(z,
                        MATH_MAD(z,
                            MATH_MAD(z, -3.96022827877537e-06f, 1.32494738004322e-04f),
                            5.08130628187577e-03f),
                        6.50222499887673e-02f),
                    3.97917223959155e-01f),
                1.0f);

        // 1 - MATH_MAD(x, r/s, x) is better for x < 0.25
        ret = 0.5f - MATH_MAD(x, MATH_FAST_DIV(r, s), x-0.5f);
    } else if (ax < 1.25f) { // next most common
        float s = ax - 1.0f;
        float P, Q;

        P = MATH_MAD(s,
                MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s, -2.16637559486879e-03f, 3.54783043256182e-02f),
                                -1.10894694282397e-01f),
                            3.18346619901162e-01f),
                        -3.72207876035701e-01f),
                    4.14856118683748e-01f),
                -2.36211856075266e-03f);

        Q = MATH_MAD(s,
                MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s, 1.19844998467991e-02f, 1.363708391202905e-02),
                                1.261712198087616e-01f),
                            7.182865441419627e-02f),
                        5.403979177021710e-01f),
                    1.064208804008442e-01f),
                1.0f);

        float pbyq = MATH_FAST_DIV(P, Q);
        const float erx = 8.45062911510468e-01f;
        float retn = pbyq + erx + 1.0f;
        float retp = 1.0f - erx - pbyq;
        ret = x < 0.0f ? retn : retp;
    } else if (ax < 9.194549560546785f) {
        float s = MATH_FAST_RCP(x*x);
        float R, S;

        if (ax < 2.8571414947509766f) {  // |x| < 1/.35
            R = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s, -9.81432934416915f, -8.12874355063066e+1f),
                                        -1.84605092906711e+2f),
                                    -1.62396669462573e+2f),
                                -6.23753324503260e+1f),
                            -1.05586262253233e+1f),
                        -6.93858572707182e-1f),
                    -9.86494403484715e-3f);

            S = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s,
                                            MATH_MAD(s, -6.04244152148581e-02f, 6.57024977031928f),
                                            1.08635005541779e+02f),
                                        4.29008140027568e+02f),
                                    6.45387271733268e+02f),
                                4.34565877475229e+02f),
                            1.37657754143519e+02f),
                        1.96512716674393e+01f),
                    1.0f);
        } else { // |x| >= 1/.35 ~ 2.857143
            R = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s, -4.83519191608651e+2f, -1.02509513161108e+3f),
                                    -6.37566443368390e+2f),
                                -1.60636384855822e+2f),
                            -1.77579549177548e+1f),
                        -7.99283237680523e-1f),
                    -9.86494292470010e-3);

            S = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s, -2.24409524465858e+1f, 4.74528541206955e+2f),
                                        2.55305040643316e+03f),
                                    3.19985821950860e+03f),
                                1.53672958608444e+03f),
                            3.25792512996574e+02f),
                        3.03380607434825e+01f),
                    1.0f);
        }

        float z = AS_FLOAT(AS_UINT(x) & 0xffffe000);
        float r = MATH_MANGLE(exp)(MATH_MAD(-z, z, -0.5625f)) *
                  MATH_MANGLE(exp)(MATH_MAD(z-x, z+x, MATH_FAST_DIV(R, S)));
        r *= MATH_FAST_RCP(ax);
        float retn = 2.0f - r;
        ret = x < 0.0f ? retn : r;
    } else if (ax < 10.054545402526855f) { // subnormal or zero return
        if (!DAZ_OPT()) {
            float s = MATH_FAST_RCP(x*x);
            float R, S;

            R = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s, -4.83519191608651e+2f, -1.02509513161108e+3f),
                                    -6.37566443368390e+2f),
                                -1.60636384855822e+2f),
                            -1.77579549177548e+1f),
                        -7.99283237680523e-1f),
                    -9.86494292470010e-3);

            S = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s, -2.24409524465858e+1f, 4.74528541206955e+2f),
                                        2.55305040643316e+03f),
                                    3.19985821950860e+03f),
                                1.53672958608444e+03f),
                            3.25792512996574e+02f),
                        3.03380607434825e+01f),
                    1.0f);

            // Factoring out the 9.1875 keeps everything normal until
            // the last factor which is e^(-9.1875^2)
            float y = x - 9.1875f;
            float r = MATH_MANGLE(exp)(MATH_MAD(-18.375f, y, -0.5625)) * MATH_MANGLE(exp)(MATH_MAD(-y, y, R/S)) / x;
            r *= 0x1.2a8fd8p-122f;
            ret = x < 0.0f ? 2.0f : r;
        } else {
            ret = x < 0.0f ? 2.0f : 0.0f;
        }
    } else {
        ret = x < 0.0f ? 2.0f : 0.0f;
        ret = BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN) ? x : ret;
    }

    return ret;
}

