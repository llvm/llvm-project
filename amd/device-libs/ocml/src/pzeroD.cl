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
MATH_PRIVATE(pzero)(double x)
{
    double z = MATH_RCP(x * x);
    double r = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z, -0x1.4850b36cc643dp+12, -0x1.36a6ecd4dcafcp+11),
                           -0x1.011027b19e863p+8),
                       -0x1.029d0b44fa779p+3),
                   -0x1.1fffffffffd32p-4) * z;
    double s = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z,
                               MATH_MAD(z, 0x1.741774f2c49dcp+15, 0x1.c810f8f9fa9bdp+16),
                               0x1.3d2bb6eb6b05fp+15),
                           0x1.df37d50596938p+11),
                       0x1.d223307a96751p+6),
                   1.0);
    return 1.0 + MATH_DIV(r, s);
}

