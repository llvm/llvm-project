
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
MATH_PRIVATE(pone)(float x)
{
    float z = MATH_RCP(x * x);

    float r = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z, 0x1.eea7acp+12f, 0x1.e457dap+11f),
                          0x1.9c0d46p+8f),
                      0x1.a7a9d4p+3f),
                  0x1.e00000p-4) * z;

    float s = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z,
                              MATH_MAD(z, 0x1.e15116p+14f, 0x1.7d42ccp+16f),
                              0x1.20b86ap+15f),
                          0x1.c85dcap+11f),
                      0x1.c8d458p+6f),
                  1.0f);

    return 1.0f + MATH_DIV(r, s);
}

