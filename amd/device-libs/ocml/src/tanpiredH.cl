/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

CONSTATTR half
MATH_PRIVATE(tanpired)(half x, short i)
{
    half s = x * x;

    half t = MATH_MAD(s, MATH_MAD(s, 0x1.3d8p+8h, 0x1.fe4p+4h), 0x1.508p+3h);

    t = x * s * t;
    t = MATH_MAD(x, 0x1.92p+1h, t);

    half tr = -MATH_RCP(t);

    return i ? tr : t;
}

