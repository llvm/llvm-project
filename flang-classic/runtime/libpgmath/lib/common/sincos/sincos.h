
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#ifndef COMMON_H_H63T0LSL
#define COMMON_H_H63T0LSL

#include <stdint.h>

#define FMA __builtin_fma

/* Constants for Cody-Waite argument reduction */
#define _1_OVER_PI   3.1830988618379069e-01
#define PI_HI        3.1415926535897931e+00
#define PI_MI        1.2246467991473515e-16
#define PI_LO        1.6956855320737797e-31
#define THRESHOLD    2.1474836480000000e+09

/* Coefficents of approximate sine on [-PI/2,+PI/2] */
#define A_D  2.7216757170572905e-15
#define B_D -7.6430432645975321e-13
#define C_D  1.6058941896620540e-10
#define D_D -2.5052106921502716e-08
#define E_D  2.7557319211743492e-06
#define F_D -1.9841269841213368e-04
#define G_D  8.3333333333332083e-03
#define H_D -1.6666666666666666e-01

/* 1152 bits of 1/PI for Payne-Hanek argument reduction. */
static uint64_t i1opi_f [] = {
    0x35fdafd88fc6ae84ULL,
    0x9e839cfbc5294975ULL,
    0xba93dd63f5f2f8bdULL,
    0xa7a31fb34f2ff516ULL,
    0xb69b3f6793e584dbULL,
    0xf79788c5ad05368fULL,
    0x8ffc4bffef02cc07ULL,
    0x4e422fc5defc941dULL,
    0x9cc8eb1cc1a99cfaULL,
    0x74ce38135a2fbf20ULL,
    0x74411afa975da242ULL,
    0x7f0ef58e5894d39fULL,
    0x0324977504e8c90eULL,
    0xdb92371d2126e970ULL,
    0xff28b1d5ef5de2b0ULL,
    0x6db14acc9e21c820ULL,
    0xfe13abe8fa9a6ee0ULL,
    0x517cc1b727220a94ULL,
    0ULL,
};

/* -fno-strict-aliasing */
static int64_t
double_as_ll(double f)
{
    return *(int64_t*)&f;
}

/* -fno-strict-aliasing */
static double
ll_as_double(int64_t i)
{
    return *(double*)&i;
}


static void
reduction_slowpath(double const a,
S(double *rs, uint64_t *hs) SINCOS_COMMA C(double *rc, uint64_t *hc))
{
    uint64_t result[4];
    uint64_t ia = double_as_ll(a);
    uint64_t s = ia & 0x8000000000000000ULL;
    uint64_t e = ((ia >> 52) & 0x7ff) - 1022;
    int32_t idx = 15 - (e >> 6);
    int32_t q;
    ia = ((ia << 11) | 0x8000000000000000ULL) >> 1;
    e = e & 63;

    __uint128_t acc = 0;
    for (q = idx; q < idx + 4; q++) {
        acc += (__uint128_t)ia * i1opi_f[q];
        result[q - idx] = (uint64_t)acc;
        acc >>= 64;
    }

    uint64_t p = result[3];
    if (e) {
        p         = (p << e)         | (result[2] >> (64 - e));
        result[2] = (result[2] << e) | (result[1] >> (64 - e));
    }

    S(
        {
            uint64_t ps = p;
            uint64_t result2 = result[2];
            uint64_t shi = s | 0x3c20000000000000ULL;
            *hs = ps & 0x8000000000000000ULL;
            if (ps & 0x4000000000000000ULL) {
                ps = ~ps;
                result2 = ~result2;
            }
    
            ps &= 0x7fffffffffffffffULL;
            int lz = __builtin_clzll(ps);
            ps = ps << lz | result2 >> (64 - lz);
            shi -= (uint64_t)lz << 52;
    
            __uint128_t prod = ps * (__uint128_t)0xc90fdaa22168c235ULL;
            uint64_t lhi = prod >> 64;
            *rs = ll_as_double(shi) * lhi;
        }
//        printf("*rs = %f\n", *rs);
    )

    C(
        {
            uint64_t pc = p;
            uint64_t result2 = result[2];
            *hc = pc & 0x8000000000000000ULL;
            uint64_t shi = 0x3c20000000000000ULL;
    
            pc &= 0x7fffffffffffffffULL;
        /* subtract 0.5 */
            pc = (int64_t)pc - 0x4000000000000000LL;
            if ((int64_t)pc < 0) {
                *hc ^= 0x8000000000000000ULL;
                pc = ~pc;
                result2 = ~result2;
            }
    
            int lz = __builtin_clzll(pc);
            pc = pc << lz | result2 >> (64 - lz);
            shi -= (uint64_t)lz << 52;
            __uint128_t prod = pc * (__uint128_t)0xc90fdaa22168c235ULL;
            uint64_t lhi = prod >> 64;
            *rc = ll_as_double(shi) * lhi;
        }
//        printf("*rc = %f\n", *rc);
    )
}


#endif




