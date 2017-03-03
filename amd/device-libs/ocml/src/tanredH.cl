/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

CONSTATTR INLINEATTR half
MATH_PRIVATE(tanred)(half x, short i)
{
    half s = x * x;

    half t = MATH_MAD(s, MATH_MAD(s, 0x1.794p-4h, 0x1.e3cp-4h), 0x1.57p-2h);
    t = MATH_MAD(x, s*t, x);

    half tr = -MATH_RCP(t);

    return i ? tr : t;
}

