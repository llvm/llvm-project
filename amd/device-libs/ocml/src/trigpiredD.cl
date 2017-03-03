/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

INLINEATTR int
MATH_PRIVATE(trigpired)(double x, __private double *r)
{
    double t = 2.0 * BUILTIN_FRACTION_F64(0.5 * x);
    x = x > 1.0 ? t : x;
    t = BUILTIN_RINT_F64(2.0 * x);
    *r = MATH_MAD(t, -0.5, x);
    return (int)t & 0x3;
}

