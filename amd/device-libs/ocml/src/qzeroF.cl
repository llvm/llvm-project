
#include "mathF.h"

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

CONSTATTR INLINEATTR float
MATH_PRIVATE(qzero)(float x)
{
    float z = MATH_RCP(x * x);
    float r = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z, 0x1.212d40p+15f, 0x1.14d994p+13f),
                          0x1.16d632p+9f),
                      0x1.789526p+3f),
                  0x1.2c0000p-4f) * z;
    float s = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z,
                              MATH_MAD(z,
                                  MATH_MAD(z, -0x1.4fd6d2p+18f, 0x1.9a66b2p+19f),
                                  0x1.883da8p+19f),
                              0x1.166526p+17f),
                          0x1.fa2584p+12f),
                      0x1.478d54p+7f),
                  1.0f);
    return MATH_DIV(-0.125f + MATH_DIV(r, s),  x);
}

