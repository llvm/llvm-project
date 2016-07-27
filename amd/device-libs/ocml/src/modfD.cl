/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

INLINEATTR double
MATH_MANGLE(modf)(double x, __private double *iptr)
{
    double tx = BUILTIN_TRUNC_F64(x);
    double ret = x - tx;
    ret = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF) ? 0.0 : ret;
    *iptr = tx;
    return BUILTIN_COPYSIGN_F64(ret, x);
}

