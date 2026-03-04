/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

CONSTATTR struct redret
MATH_PRIVATE(trigpired)(double x)
{
    double t = 2.0 * BUILTIN_FRACTION_F64(0.5 * x);
    x = x > 1.0 ? t : x;
    t = BUILTIN_RINT_F64(2.0 * x);

    struct redret ret;
    ret.hi = MATH_MAD(t, -0.5, x);
    ret.i = (int)t & 0x3;
    return ret;
}

