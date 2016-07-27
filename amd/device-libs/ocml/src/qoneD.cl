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
MATH_PRIVATE(qone)(double x)
{
    double z = MATH_RCP(x * x);
    double r = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z, -0x1.7a6d065d09c6ap+15, -0x1.724e740f87415p+13),
                           -0x1.7bcd053e4b576p+9),
                       -0x1.04591a26779f7p+4),
                   -0x1.a3ffffffffdf3p-4) * z;

    double s = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z,
                               MATH_MAD(z,
                                   MATH_MAD(z, -0x1.1f9690ea5aa18p+18, 0x1.457d27719ad5cp+19),
                                   0x1.5f65372869c19p+19),
                               0x1.0579ab0b75e98p+17),
                           0x1.e9162d0d88419p+12),
                       0x1.42ca6de5bcde5p+7),
                   1.0);

    return MATH_DIV(0.375 + MATH_DIV(r, s),  x);
}

