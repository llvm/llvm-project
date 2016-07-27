/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

// This implementation makes use of large x approximations from
// the Sun library which reqires the following to be included:
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

CONSTATTR INLINEATTR double
MATH_PRIVATE(qzero)(double x)
{
    double z = MATH_RCP(x * x);
    double r = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z, 0x1.212d40e901566p+15, 0x1.14d993e18f46dp+13),
                           0x1.16d6315301825p+9),
                       0x1.789525bb334d6p+3),
                   0x1.2bffffffffe2cp-4) * z;
    double s = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z,
                               MATH_MAD(z,
                                   MATH_MAD(z, -0x1.4fd6d2c9530c5p+18, 0x1.9a66b28de0b3dp+19),
                                   0x1.883da83a52b43p+19),
                               0x1.1665254d38c3fp+17),
                           0x1.fa2584e6b0563p+12),
                       0x1.478d5365b39bcp+7),
                   1.0);
    return MATH_DIV(-0.125 + MATH_DIV(r, s),  x);
}

