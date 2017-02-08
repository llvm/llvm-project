/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(ncdf)(double x)
{
    double ret;

    if (x > -0x1.5956b87528a49p-1) {
        if (x < 1.0) {
            double t = x * x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, 
                      -0x1.8cb754014e0b3p-34, 0x1.320d075b1fdefp-29), -0x1.61ab7dd43f8c3p-25), 0x1.6584e2ae1c515p-21),
                      -0x1.3ce8d5eca373fp-17), 0x1.e42b0c16331c9p-14), -0x1.37403f689501bp-10), 0x1.46d0429761749p-7),
                      -0x1.1058377e2ce69p-4), 0x1.9884533d43650p-2);
            ret = MATH_MAD(x, ret, 0.5);
        } else if (x < 2.5) {
            double t = x - 1.0;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.060edab4a19d2p-29, -0x1.53a0eb739ccefp-25), 0x1.4c8f542ea757fp-22), -0x1.1c15387d5063ap-20),
                      0x1.fadb9735a0803p-22), 0x1.a2bae693176d3p-18), -0x1.cd9e9b6a563dbp-21), -0x1.73fccf7f7f32cp-14),
                      0x1.f8d0e4a86cde5p-14), 0x1.92ac8d4045877p-11), -0x1.084ad98cd25bfp-9), -0x1.084c041e359abp-8),
                      0x1.4a5ee6ad39afcp-6), -0x1.c16ac04dad985p-35), -0x1.ef8e58e30ef67p-4), 0x1.ef8e58e331308p-3),
                      0x1.aec4bd120d37ep-1);
        } else if (x < 4.0) {
            double t = x - 2.5;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      -0x1.5f0f31da8eb78p-33, -0x1.51820cdbd28e7p-32), 0x1.af16a4a50d960p-26), -0x1.b5b829c3676fep-23),
                      0x1.6a839ce113434p-21), -0x1.efa0b32917d76p-24), -0x1.c2eaad7a58467p-18), 0x1.2c1fa77adea62p-16),
                      0x1.c789d533e599bp-16), -0x1.13874be6da82dp-12), 0x1.0d3cf7e102cccp-11), 0x1.5d67fa3a182e7p-11),
                      -0x1.84e50141ef284p-8), 0x1.f6924953c9cbbp-7), -0x1.66fac6add3b42p-6), 0x1.1f2f0557f4ab9p-6),
                      0x1.fcd21635036c6p-1);
        } else if (x < 8.2109375) {
            double t = x - 4.0;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      -0x1.49dae5934aa9ep-37, 0x1.a0a9b27e4276cp-33), -0x1.40ae395c9950bp-32), -0x1.6d7df112c9529p-26),
                      0x1.f76261921be9dp-25), 0x1.a70ffb3533144p-19), -0x1.9e462dbfa92d9p-16), -0x1.5db0c27784edap-13),
                      0x1.3c5a964f22d79p-9), 0x1.5cadd35757947p-9), -0x1.1b11634e869afp-3), 0x1.0bf46d4a7c1dap-1);
            ret = ret * ret;
            ret = ret * ret;
            ret = ret * ret;
            ret = MATH_MAD(-ret, ret, 1.0);
        } else {
            ret = 1.0;
        }
    } else {
        if (x > -1.5) {
            double t = -1.5 - x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.87f6d8bacfe4dp-24, -0x1.48dcea6d816e1p-23), 0x1.a32c40a47a30ep-20), 0x1.bd22f42e45845p-21),
                      -0x1.40839ec0fb6a8p-16), 0x1.a659159d48d42p-16), 0x1.6f322a8af7fa6p-13), -0x1.2466b5cb3347ep-11),
                      -0x1.58d37df0dc6c4p-11), 0x1.809d8fed7b759p-8), -0x1.8de0c7fed2ce4p-8), -0x1.ba1633b5691dfp-6),
                      0x1.8de0c823b3adcp-4), -0x1.0940856d21e73p-3), 0x1.11a46d89647efp-4);
        } else if (x > -2.25) {
            double t = -2.25 - x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      0x1.34778becb8778p-25, -0x1.48b485e383089p-24), -0x1.bd48bc73889cap-21), 0x1.b73b6859639c8p-20),
                      0x1.3582af30190aap-18), -0x1.1ac5d5e34ec1bp-15), 0x1.0cc99e25a5373p-15), 0x1.14835909e7060p-12),
                      -0x1.03e8ee71d051cp-10), 0x1.e44553637b8cap-12), 0x1.9234723301c22p-8), -0x1.601939c453937p-6),
                      0x1.24833bce57500p-5), -0x1.0402dfd3dc1adp-5), 0x1.90924f21d3612p-7);
        } else if (x > -2.75) {
            double t = -2.75 - x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, 
                      0x1.b9337a6a3734cp-24, -0x1.6590be46da1cep-23), -0x1.267a1aba29190p-20), 0x1.5254da7def6c3p-18),
                      -0x1.502fd581f8723p-19), -0x1.9d5f911317093p-15), 0x1.7a91271378f92p-13), -0x1.f4331ea1149bdp-14),
                      -0x1.2654aaf562b70p-10), 0x1.378ebd4d4cb5bp-8), -0x1.45e9ccb8cbc85p-7), 0x1.99b83490879c6p-7),
                      -0x1.29fa54c6341e5p-7), 0x1.86904349ec803p-9);
        } else if (x > -38.46875) {
            double t = MATH_RCP(x * x);

            if (x > -4.0)
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, 
                          0x1.088bebb0c7bfcp+25, -0x1.964e1d51045b9p+25), 0x1.255cf223ca4ddp+25), -0x1.093e30bdaaf0ap+24),
                          0x1.51dabf56ccafap+22), -0x1.440d8ce218330p+20), 0x1.eaab175120c83p+17), -0x1.31cd405f6ece6p+15),
                          0x1.4949b45c18bffp+12), -0x1.476ca2d47ed6dp+9), 0x1.4b5c83b73de92p+6), -0x1.86317d1686e59p+3),
                          0x1.3fab4df0327b3p+1), -0x1.fffc093fa2eedp-1), -0x1.3f9112da61104p-8);
            else
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                          0x1.668af6ed742f7p+59, -0x1.e8a3ea3ebba9fp+58), 0x1.39149210574c4p+57), -0x1.f6e7aed1dc814p+54),
                          0x1.1d2c1545c3a31p+52), -0x1.e8eb69ce384f2p+48), 0x1.4c8445a6d688bp+45), -0x1.7638c79bb1508p+41),
                          0x1.6c05288dd5cfbp+37), -0x1.41fe50b8d5f0fp+33), 0x1.12af999e7acfap+29), -0x1.e02f34f68433ep+24),
                          0x1.c4864e8ef2105p+20), -0x1.dc7852ceec4e8p+16), 0x1.1f83f2164bb6fp+13), -0x1.9819642b134dbp+9),
                          0x1.60fffe9105243p+6), -0x1.8aaaaaa42b3fdp+3), 0x1.3ffffffff70fdp+1), -0x1.fffffffffff98p-1),
                          -0x1.3f8e4325f5a57p-8);

            double xh = AS_DOUBLE(AS_LONG(x) & 0xffffffff00000000L);
            ret = MATH_DIV(MATH_MANGLE(exp)(MATH_MAD(x - xh,  -0.5*(x + xh), ret)), -x) *
                  MATH_MANGLE(exp)(MATH_MAD(xh, -0.5*xh, -0.9140625));
        } else {
            ret = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? x : 0.0;
        }
    }

    return ret;
}

