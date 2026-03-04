/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

CONSTATTR struct scret
MATH_PRIVATE(sincosred)(half x)
{
    half t = x * x;
    half s = MATH_MAD(x, t*MATH_MAD(t, 0x1.0bp-7h, -0x1.554p-3h), x);
    half c = MATH_MAD(t, MATH_MAD(t, 0x1.4b4p-5h, -0x1.ffcp-2h), 1.0h);

    struct scret ret;
    ret.c = c;
    ret.s = s;
    return ret;
}

