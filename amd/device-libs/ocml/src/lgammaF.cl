/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

INLINEATTR float
MATH_MANGLE(lgamma)(float x)
{
    int s;
    return MATH_MANGLE(lgamma_r)(x, &s);
}

