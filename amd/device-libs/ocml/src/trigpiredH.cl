/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

REQUIRES_16BIT_INSTS CONSTATTR struct redret
MATH_PRIVATE(trigpired)(half x)
{
    half t = 2.0h * BUILTIN_FRACTION_F16(0.5h * x);
    x = x > 1.0h ? t : x;
    t = BUILTIN_RINT_F16(2.0h * x);

    struct redret ret;
    ret.hi = MATH_MAD(t, -0.5h, x);
    ret.i = (short)t & (short)0x3;
    return ret;
}

