/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(scalbn)(half2 x, int2 n)
{
    return BUILTIN_FLDEXP_2F16(x, n);
}

CONSTATTR half
MATH_MANGLE(scalbn)(half x, int n)
{
    return BUILTIN_FLDEXP_F16(x, n);
}

