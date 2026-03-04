/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

double
MATH_MANGLE(fract)(double x, __private double *ip)
{
    *ip = BUILTIN_FLOOR_F64(x);
    return BUILTIN_FRACTION_F64(x);
}

