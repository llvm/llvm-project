/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(erf)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 1.0) {
        double t = ax * ax;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  -0x1.abae491c443a9p-31, 0x1.d71b0f1b10a64p-27), -0x1.5c0726f04dcfbp-23), 0x1.b97fd3d992938p-20),
                  -0x1.f4ca4d6f3e30fp-17), 0x1.f9a2baa8fedd2p-14), -0x1.c02db03dd71d4p-11), 0x1.565bccf92b2f9p-8),
                  -0x1.b82ce311fa93ep-6), 0x1.ce2f21a040d16p-4), -0x1.812746b0379bdp-2), 0x1.20dd750429b6dp+0);
        ret = ax * ret;
    } else if (ax < 1.75) {
        double t = ax - 1.0;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, 
                  -0x1.6bcf230661065p-17, 0x1.9753b107bf6d4p-14), -0x1.412a8fbbf6a41p-12), 0x1.7e91ed5c78d47p-13),
                  0x1.311624dd4ecf6p-10), -0x1.e690fc0cfdf11p-10), -0x1.39b63f9de527cp-8), 0x1.f0ab1e1d99aabp-7),
                  0x1.2e3eae226d671p-8), -0x1.1b613dcdcae03p-4), 0x1.1b614a07894f2p-4), 0x1.1b614b157fb65p-3),
                  -0x1.a911f0970f238p-2), 0x1.a911f096fbf25p-2), 0x1.af767a741088ap-1);
    } else if (ax < 2.5) {
        double t = ax - 1.75;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, 
                  0x1.1f15b50f14138p-18, -0x1.045a3b7a40cf2p-15), 0x1.56639c276284fp-14), -0x1.dafc588ff3ae7p-17),
                  -0x1.cbee01a4e2823p-12), 0x1.d416a49b45130p-11), 0x1.7eeb945b26b23p-11), -0x1.8d11b6edee21fp-8),
                  0x1.25b37e45fe07cp-7), 0x1.b22258ef6e0b9p-8), -0x1.8a0da54b6ec66p-5), 0x1.7148c3d5d1ad0p-4),
                  -0x1.7a4a8a2bdfe71p-4), 0x1.b05530322115ap-5), 0x1.f92d077f8d56dp-1);
    } else if (ax < 4.0) {
        double t = ax - 2.5;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  -0x1.708f6d0e65c33p-32, 0x1.dbd0618847c60p-28), -0x1.c3001cf83cd69p-26), -0x1.4dca746dfe625p-22),
                  0x1.a8e79a95d6f67p-20), 0x1.8d8d7711fc864p-16), -0x1.99fe2d9d9b69bp-13), -0x1.b3b1f1e28669cp-12),
                  0x1.01d3d83753fb1p-7), -0x1.e842cf8341e6ap-10), -0x1.a49bb4ab1d7d9p-3), 0x1.3a50e1b16e339p-1);
        ret = ret*ret;
        ret = ret*ret;
        ret = ret*ret;
        ret = MATH_MAD(-ret, ret, 1.0);
    } else if (ax < 5.9375) {
        double t = ax - 4.0;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, 
                  0x1.5b22d2cd54932p-26, -0x1.3e056a1040a29p-24), -0x1.2d8f6bf8af04ap-19), 0x1.4c20d337a4541p-16),
                  0x1.d9d0971c8f96dp-16), -0x1.0a33e01adb0ddp-10), 0x1.63716fb40eab9p-9), 0x1.7d6f6bbcfc7e0p-6),
                  -0x1.5687476feec74p-3), 0x1.4cb2bacd30820p-2);
        ret = ret*ret;
        ret = ret*ret;
        ret = ret*ret;
        ret = MATH_MAD(-ret, ret, 1.0);
    } else {
        ret = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? x : 1.0;
    }

    ret = BUILTIN_COPYSIGN_F64(ret, x);
    return ret;
}

