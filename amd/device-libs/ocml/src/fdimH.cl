/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(fdim)

CONSTATTR half
MATH_MANGLE(fdim)(half x, half y)
{
    return (x <= y && !BUILTIN_ISUNORDERED_F16(x, y)) ? 0.0h : (x - y);
}

