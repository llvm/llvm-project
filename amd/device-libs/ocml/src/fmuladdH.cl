/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(fmuladd)(half2 a, half2 b, half2 c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}


CONSTATTR half
MATH_MANGLE(fmuladd)(half a, half b, half c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}

