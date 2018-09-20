/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(pown)(half2 x, int2 ny)
{
    return (half2)(MATH_MANGLE(pown)(x.lo, ny.lo), MATH_MANGLE(pown)(x.hi, ny.hi));
}

#define COMPILING_POWN
#include "powH_base.h"

