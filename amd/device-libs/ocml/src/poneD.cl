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
MATH_PRIVATE(pone)(double x)
{
    double z = MATH_RCP(x * x);

    double r = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z, 0x1.eea7ac32782ddp+12, 0x1.e457da3a532ccp+11),
                           0x1.9c0d4652ea590p+8),
                       0x1.a7a9d357f7fcep+3),
                   0x1.dfffffffffccep-4) * z;

    double s = MATH_MAD(z,
                   MATH_MAD(z,
                       MATH_MAD(z,
                           MATH_MAD(z,
                               MATH_MAD(z, 0x1.e1511697a0b2dp+14, 0x1.7d42cb28f17bbp+16),
                               0x1.20b8697c5bb7fp+15),
                           0x1.c85dc964d274fp+11),
                       0x1.c8d458e656cacp+6),
                   1.0);

    return 1.0 + MATH_DIV(r, s);
}

