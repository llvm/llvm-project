/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <math.h>
#include <stdint.h>
#include "exp_defs.h"

#define FMAF __builtin_fmaf

extern "C" float __fss_exp_fma3(float);

inline float itf(int a)
{
    return *reinterpret_cast<float*>(&a);
}

inline int fti(float a)
{
    return *reinterpret_cast<int*>(&a);
}

float __fss_exp_fma3(const float a)
{
    // Quick exit if argument is +/-0.0
    const uint32_t a_u32 = *reinterpret_cast<const uint32_t *>(&a);
    if ((a_u32 << 1) == 0) return 1.0;

    if (a != a)
        return a;
    if (a >= EXP_HI)
        return itf(INF);
    if (a <= EXP_LO)
        return 0.0f;
    float t = FMAF(a, L2E, FLT2INT_CVT);
    float tt = t - FLT2INT_CVT;
    float z = FMAF(tt, -LN2_0, a);
          z = FMAF(tt, -LN2_1, z);

    int exp = fti(t);
        exp <<= 23;

    float zz =             EXP_C7;
          zz = FMAF(zz, z, EXP_C6);
          zz = FMAF(zz, z, EXP_C5);
          zz = FMAF(zz, z, EXP_C4);
          zz = FMAF(zz, z, EXP_C3);
          zz = FMAF(zz, z, EXP_C2);
          zz = FMAF(zz, z, EXP_C1);
          zz = FMAF(zz, z, EXP_C0);

    if (a <= EXP_DN) {
        int dnrm = exp > DNRM_THR ? DNRM_THR : exp;
        dnrm = dnrm + DNRM_SHFT;
        exp = exp > DNRM_THR ? exp : DNRM_THR;
        float res = itf(exp + fti(zz));
        res = res * itf(dnrm);
        return res;
    } else {
        return itf(exp + fti(zz));
    }
}
