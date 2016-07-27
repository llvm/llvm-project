/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

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
MATH_PRIVATE(qone)(float x)
{
    float z = MATH_RCP(x * x);

    float r = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z, -0x1.7a6d06p+15f, -0x1.724e74p+13f),
                          -0x1.7bcd06p+9f),
                      -0x1.04591ap+4f),
                  -0x1.a40000p-4f) * z;

    float s = MATH_MAD(z,
                  MATH_MAD(z,
                      MATH_MAD(z,
                          MATH_MAD(z,
                              MATH_MAD(z,
                                  MATH_MAD(z, -0x1.1f9690p+18f, 0x1.457d28p+19f),
                                  0x1.5f6538p+19f),
                              0x1.0579acp+17f),
                          0x1.e9162ep+12f),
                      0x1.42ca6ep+7f),
                  1.0f);

    return MATH_DIV(0.375f + MATH_DIV(r, s), x);
}

