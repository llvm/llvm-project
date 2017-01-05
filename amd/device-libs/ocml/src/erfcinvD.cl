/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(erfcinv)(double y)
{
    double ret;

    if (y > 0.625) {
        ret = MATH_MANGLE(erfinv)(1.0 - y);
    } else if (y > 0x1.0p-10) {
        double t = -MATH_MANGLE(log)(y * (2.0 - y)) - 3.125;

        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, 
                  0x1.1267a785a1166p-69, -0x1.a6581051dd484p-63), 0x1.2b2956fc047a4p-60), 0x1.ad835aed5cc07p-57),
                  -0x1.25e0612eae68fp-53), 0x1.a0cab63f02a91p-57), 0x1.d9227af501adbp-48), -0x1.6c3ad559a9b4ep-45),
                  -0x1.6cafa36036318p-44), 0x1.72879641e158fp-39), -0x1.c89d755f7fff8p-37), -0x1.dc51171ddae3ap-35),
                  0x1.20f512744ae65p-30), -0x1.1a9e5f4bcfcd8p-28), -0x1.f36ce926b83e8p-26), 0x1.c6b4f6c7cfa1ep-22),
                  -0x1.6e8a53e0c2026p-20), -0x1.d1d1f7bf4570bp-17), 0x1.879c2a20cc3e2p-13), -0x1.8457694844d14p-11),
                  -0x1.8b6c33114edadp-8), 0x1.ebd80d9b13e14p-3), 0x1.a755e7c99ae86p+0);
        ret = BUILTIN_FMA_F64(-y, ret, ret);
    } else {
        double s = MATH_SQRT(-MATH_MANGLE(log)(y));
        double t = MATH_RCP(s);

        if (y > 0x1.0p-19) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.8b3cfc98a5212p+4, -0x1.907bcdab54a4ep+6), 0x1.7659cf8216d7dp+7), -0x1.ac222777f664dp+7),
                      0x1.4f2f8e33151acp+7), -0x1.7d7d1eb301c4cp+6), 0x1.48e630c1c77e7p+5), -0x1.c63e7d0e327f6p+3),
                      0x1.225b286aeb0dfp+2), -0x1.82a4acc22b05dp+0), -0x1.0a88271680e57p-5), 0x1.001f6acebb122p+0);
        } else if (y > 0x1.0p-40) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.0fdcb40bf066dp+9, -0x1.870ddeaa832dbp+10), 0x1.035c39e0428c4p+11), -0x1.a4d3c54a3ec14p+10),
                      0x1.d382aee6efae8p+9), -0x1.79f9e26565bc1p+8), 0x1.d00e058ce9abap+6), -0x1.c7d1e01821eb3p+4),
                      0x1.9d930ba7a3111p+2), -0x1.af47941dd2baap+0), -0x1.787ecc823998bp-6), 0x1.000fae5fb73e3p+0);
        } else if (y > 0x1.0p-82) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.c9e5b8e31c18ep+13, -0x1.c866153b1bce6p+14), 0x1.a386b3b4fb25cp+14), -0x1.d7bf378e7b5fbp+13),
                      0x1.6b416de0a7a75p+12), -0x1.9757c1cf44e90p+10), 0x1.5b56ededbaa8cp+8), -0x1.da79924b4d155p+5),
                      0x1.2ba25315d612bp+3), -0x1.de5808fbd786dp+0), -0x1.04e014b9fc507p-6), 0x1.000788df1c89fp+0);
        } else if (y > 0x1.0p-200) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.ff518aae00301p+18, -0x1.5781ef98c6aa9p+19), 0x1.a9511b21c7715p+18), -0x1.41d8f1455b21ep+17),
                      0x1.4d4a3d4025a4cp+15), -0x1.f640fe7077996p+12), 0x1.1faf674f42181p+10), -0x1.080c5cd81d791p+7),
                      0x1.c0ae370098ef4p+3), -0x1.08ebd67dc005ap+1), -0x1.5cf3329e72289p-7), 0x1.00035e75f27e2p+0);
        } else if (y > 0x1.0p-400) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.d554f00bf9d81p+20, 0x1.8456711ff3627p+20), -0x1.26c90acc5daafp+19), 0x1.106501cdef815p+17),
                      -0x1.57a4c95601c04p+14), 0x1.3ca627cbaede6p+11), -0x1.c716e091922fbp+7), 0x1.292f8f6e8bc75p+4),
                      -0x1.1b469c212bd5fp+1), -0x1.04977fb6d0462p-7), 0x1.0001dc9f52f8ap+0);
        } else if (y > 0x1.0p-900) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.21913925f3a73p+25, 0x1.4aa2fba282b9bp+24), -0x1.5a2a3f9742896p+22), 0x1.b8ee3895772e8p+19),
                      -0x1.7f2ce0b036be4p+16), 0x1.e62ab1bcbb738p+12), -0x1.e0ed2965d2a06p+8), 0x1.b0c16705263e5p+4),
                      -0x1.334f9a732ecc7p+1), -0x1.65f60412f9578p-8), 0x1.0000e0bda43b5p+0);
        } else {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.e3d70f1fdc7bep+11, 0x1.28d9acd5b9596p+10), -0x1.554c1ce591414p+7), 0x1.15b1e5a1fe7f5p+4),
                      -0x1.1aa8e6f616c69p+1), -0x1.f6803b3b4d6ccp-8), 0x1.00019ac5bed2ap+0);
        }
        ret = s * ret;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = (y < 0.0) | (y > 2.0) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = y == 0.0 ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = y == 2.0 ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
    }

    return ret;
}

