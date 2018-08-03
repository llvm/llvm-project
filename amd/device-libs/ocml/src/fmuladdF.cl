/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "mathF.h"

CONSTATTR float
MATH_MANGLE(fmuladd)(float a, float b, float c)
{
    #pragma OPENCL FP_CONTRACT ON
    return a * b + c;
}

