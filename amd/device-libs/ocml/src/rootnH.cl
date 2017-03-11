/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR INLINEATTR half2
MATH_MANGLE2(rootn)(half2 x, int2 ny)
{
    return (half2)(MATH_MANGLE(rootn)(x.lo, ny.lo), MATH_MANGLE(rootn)(x.hi, ny.hi));
}

#define COMPILING_ROOTN
#include "powH_base.h"

