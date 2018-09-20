/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// Some of this implementation is based on ideas from Sun LLVM
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

CONSTATTR float
MATH_MANGLE(erfc)(float x)
{
    float ret;

    if (x < 0x1.e861fcp-2f) {
        if (x > -1.0f) {
            float t = x * x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t,
                      0x1.496a32p-14f, -0x1.a3f700p-11f), 0x1.5405b2p-8f), -0x1.b7f90ep-6f),
                      0x1.ce2cf8p-4f), -0x1.81273ep-2f), 0x1.20dd74p+0f),
            ret = MATH_MAD(-x, ret, 1.0f);
        } else if (x > -2.0f) {
            float t = -x - 1.0f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                      -0x1.e72c84p-9f, 0x1.fe43a0p-6f), -0x1.6c8eecp-4f), 0x1.3db6cep-4f),
                      0x1.1760e0p-3f), -0x1.a8d6d0p-2f), 0x1.a90f56p-2f), 0x1.d7bb3ep+0f);
        } else if (x > -3.74609375f) {
            float t = -x - 2.0f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t,
                      -0x1.19665ap-13f, -0x1.d8e18ap-14f), 0x1.13b7c0p-7f), -0x1.cf36a8p-7f),
                      -0x1.9460fap-3f), 0x1.6e23c8p-1f);
            ret = ret*ret;
            ret = ret*ret;
            ret = ret*ret;
            ret = MATH_MAD(-ret, ret, 2.0f);
        } else {
            return 2.0f;
        }
    } else {
        if (x < 1.0f) {
            float t = x - 0.75f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t,
                      0x1.b3ca9ap-6f, 0x1.a27606p-5f), -0x1.3489bcp-3f), -0x1.b5b5f0p-6f),
                      0x1.edc50cp-2f), -0x1.492e58p-1f), 0x1.27c6d2p-2f);
        } else if (x < 1.5f) {
            float t = x - 1.25f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t,
                      -0x1.558b4ep-6f, 0x1.7f4316p-5f), 0x1.9362c6p-8f), -0x1.5716acp-3f),
                      0x1.2ebf30p-2f), -0x1.e4653cp-3f), 0x1.3bcd14p-4f);
        } else if (x < 1.75f) {
            float t = x - 1.625f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t, MATH_MAD(t,
                      -0x1.d1cd9cp-7f, 0x1.2d8f6cp-9f), 0x1.9742c6p-5f), -0x1.d66472p-4f),
                      0x1.0bcfcep-3f), -0x1.499d46p-4f), 0x1.612d8ap-6f);
        } else if (x < 10.0234375f) {
            float t = MATH_FAST_RCP(x*x);

            if (x < 2.75f)
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                      MATH_MAD(t, MATH_MAD(t,
                          0x1.ecf46ap-1f, -0x1.d8a006p+0f), 0x1.ab72d8p+0f), -0x1.05ed12p+0f),
                          0x1.2691fep-1f), -0x1.fd0ddcp-2f), -0x1.45b16ep-7f);
            else
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                      MATH_MAD(t, MATH_MAD(t,
                          0x1.107a4cp+4f, -0x1.7fa404p+3f), 0x1.22b8c8p+2f), -0x1.7faf0cp+0f),
                          0x1.3f746ep-1f), -0x1.fffc90p-2f), -0x1.4341a6p-7f);

            float xh = AS_FLOAT(AS_INT(x) & 0xffffe000);
            ret = MATH_FAST_DIV(MATH_MANGLE(exp)(MATH_MAD(xh - x,  xh + x, ret)), x) *
                  MATH_MANGLE(exp)(MATH_MAD(xh, -xh, -0.5625f));
        } else {
            ret = BUILTIN_ISNAN_F32(x) ? x : 0.0f;
        }
    }

    return ret;
}

