/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE2(fmuladd)(float2 a, float2 b, float2 c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}

CONSTATTR float
MATH_MANGLE(fmuladd)(float a, float b, float c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}

