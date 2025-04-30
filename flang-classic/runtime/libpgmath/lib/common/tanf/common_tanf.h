
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#ifndef COMMON_H_H63T0LSL
#define COMMON_H_H63T0LSL

#include <stdint.h>

#define FMAF __builtin_fmaf

/* Constants for Cody-Waite argument reduction */
#define _2_OVER_PI_F 6.36619772e-01f
#define PI_2_HI_F    1.57079601e+00f
#define PI_2_MI_F    3.13916473e-07f
#define PI_2_LO_F    5.38561632e-15f
#define THRESHOLD_F  1.00000000e+04f

/* Coefficents of approximate tan on [-PI/4,+PI/4] */
#define A_F 9.42561682e-03f
#define B_F 3.06017953e-03f
#define C_F 2.44512185e-02f
#define D_F 5.34108058e-02f
#define E_F 1.33389056e-01f
#define F_F 3.33331138e-01f

/* 192 bits of 2/PI for Payne-Hanek argument reduction. */
static uint32_t i2opi_f [] = {
    0x3c439041,
    0xdb629599,
    0xf534ddc0,
    0xfc2757d1,
    0x4e441529,
    0xa2f9836e,
};

#define PI_2_M64 1.70306079004327746902e-19

/* -fno-strict-aliasing */
static int32_t
float_as_int(float f)
{
    return *(int32_t*)&f;
}

/* -fno-strict-aliasing */
static float
int_as_float(int32_t i)
{
    return *(float*)&i;
}

typedef struct {
    uint32_t x;
    uint32_t y;
} uint2;

/* -fno-strict-aliasing */
static uint2
umad32wide(uint32_t a, uint32_t b, uint32_t c)
{
    union {
        uint2 ui2;
        uint64_t ull;
    } res;
    res.ull = (uint64_t)a * b + c;
    return res.ui2;
}

#endif
