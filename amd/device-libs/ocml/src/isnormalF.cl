/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR int
MATH_MANGLE(isnormal)(float x)
{
    return BUILTIN_ISNORMAL_F32(x);
}
