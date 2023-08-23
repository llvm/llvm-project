/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR short2
MATH_MANGLE2(signbit)(half2 x)
{
    return (short2)
        (AS_SHORT(x.lo) < 0 ? (short)-1 : (short)0,
         AS_SHORT(x.hi) < 0 ? (short)-1 : (short)0);
}

CONSTATTR int
MATH_MANGLE(signbit)(half x)
{
    return AS_SHORT(x) < 0;
}
