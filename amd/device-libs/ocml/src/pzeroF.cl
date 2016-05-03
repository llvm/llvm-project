
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
MATH_PRIVATE(pzero)(float x)
{
    float z = MATH_RCP(x * x);
    float r = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z, -0x1.4850b4p+12f, -0x1.36a6ecp+11f),
                          -0x1.011028p+8f),
                      -0x1.029d0cp+3f),
                  -0x1.200000p-4f) * z;
    float s = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z,
                              MATH_MAD(z, 0x1.741774p+15f, 0x1.c810f8p+16f),
                              0x1.3d2bb6p+15f),
                          0x1.df37d6p+11f),
                      0x1.d22330p+6f), 1.0f);
    return 1.0f + MATH_DIV(r, s);
}

