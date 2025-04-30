/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <math.h>
#include "fslog_defs.h"

#define FMAF __builtin_fmaf

extern "C" float __fss_log_fma3(float);

//inline float itf(int a)
//{
//    return *reinterpret_cast<float*>(&a);
//}

//inline int fti(float a)
//{
//    return *reinterpret_cast<int*>(&a);
//}

float __fss_log_fma3(float a_input)
{

    unsigned const canonical_nan = CANONICAL_NAN;
    unsigned const minus_inf = MINUS_INF;

    unsigned exp_offset = EXP_OFFSET;

    unsigned a_input_as_uint = *(unsigned*)&a_input;

    if (a_input < 0.0f) {
        return *(float*)&canonical_nan;;
    }

    if ( (a_input_as_uint & NAN_INF_MASK) == NAN_INF_MASK ) {
        return a_input + a_input;
    }

    if (a_input == 0.0f) {
        return *(float*)&minus_inf;
    }

    if (a_input < TWO_TO_M126_F) {
        a_input *= TWO_TO_24_F;
        exp_offset += U24;
    }

    unsigned a_as_uint = *(unsigned*)&a_input;
    int e_int = (a_as_uint>>23) - exp_offset;
    float e = (float)e_int;

    unsigned m_as_uint = (a_as_uint & BIT_MASK2) + OFFSET;
    float m = *(float*)&m_as_uint;

    if (m < PARTITION_CONST) {
        m = m + m;
        e = e - 1.0f;
    }
    m = m - 1.0f;

    float LN2_0 =  0x1.62E43p-01;
    
    float exp = e * LN2_0;

    float t = LOG_CA;
    t = FMAF(t, m, LOG_C9);
    t = FMAF(t, m, LOG_C8);
    t = FMAF(t, m, LOG_C7);
    t = FMAF(t, m, LOG_C6);
    t = FMAF(t, m, LOG_C5);
    t = FMAF(t, m, LOG_C4);
    t = FMAF(t, m, LOG_C3);
    t = FMAF(t, m, LOG_C2);
    t = FMAF(t, m, LOG_C1);
    
    float m2 = m * m;
    t = FMAF(t, m2, m);
    t = t + exp;

    return t;
}
