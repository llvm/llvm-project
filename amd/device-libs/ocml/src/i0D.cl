/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

double
MATH_MANGLE(i0)(double x)
{
    x = BUILTIN_ABS_F64(x);

    double ret;

    if (x < 8.0) {
        double t = 0.25 * x * x;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  0x1.dd78750ff79b2p-97, 0x1.4394559531e65p-89), 0x1.6f7123f151c79p-81), 0x1.3d9e7c5528048p-73),
                  0x1.e736f323a0cabp-66), 0x1.4196ce3b298c5p-58), 0x1.69caac7bf9255p-51), 0x1.5601878c06ac8p-44),
                  0x1.0b313291f5e48p-37), 0x1.522a43f5dcb54p-31), 0x1.522a43f659634p-25), 0x1.02e85c0898945p-19),
                  0x1.23456789abcf3p-14), 0x1.c71c71c71c71cp-10), 0x1.c71c71c71c71cp-6), 0x1.0000000000000p-2),
                  0x1.0000000000000p+0),
        ret = MATH_MAD(t, ret, 1.0f);
    } else {
        double t = MATH_RCP(x);
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, 
                  0x1.cc967bacb549dp+49, -0x1.5ba7722975981p+50), 0x1.df0f836763276p+49), -0x1.9042a430f3f43p+48),
                  0x1.c630541c4f568p+46), -0x1.7366be5a9784fp+44), 0x1.c5669a48f574ep+41), -0x1.a664cac47f0eap+38),
                  0x1.308250566988cp+35), -0x1.56874c2ddb061p+31), 0x1.2da58968da2aap+27), -0x1.9faaa33f0d6bcp+22),
                  0x1.be0a8f2bc76ddp+17), -0x1.7123c68c3cb02p+12), 0x1.d402150cc72aap+6), -0x1.7a8ae85359520p+0),
                  0x1.bd7e0b6a753cdp-4), 0x1.6d6ce3774506dp-5), 0x1.debdd3d2f7cf9p-6), 0x1.cb94db8d452d5p-6),
                  0x1.9884533daea3dp-5), 0x1.9884533d4362fp-2);
        double xs = x - 709.0;
        double e1 = MATH_MANGLE(exp)(x > 709.0 ? xs : x);
        double e2 = x > 709.0 ? 0x1.d422d2be5dc9bp+1022 : 1.0;
        ret = e1 * MATH_MANGLE(rsqrt)(x) * ret * e2;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
    }

    return ret;
}

