/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

int   const ABS_MASK_I  = 0x7fffffff;
int   const SGN_MASK_I  = 0x80000000;
float const ONE_F       = 1.0f;
float const THRESHOLD_F = 0.5705f;
float const PIO2_F      = 1.570796327f;

// p0 coefficients
float const A0_F        =  5.175137819e-02f;
float const B0_F        =  1.816697683e-02f;
float const C0_F        =  4.675685871e-02f;
float const D0_F        =  7.484657646e-02f;
float const E0_F        =  1.666701424e-01f;

// p1 coefficients
float const A1_F        = -7.437243476e-04f;
float const B1_F        =  5.207145121e-03f;
float const C1_F        = -1.764218137e-02f;
float const D1_F        =  4.125141352e-02f;
float const E1_F        = -8.533414453e-02f;
float const F1_F        =  2.137603760e-01f;
float const G1_F        = -1.570712566e-00f;

