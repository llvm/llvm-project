/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(fmin)(double x, double y)
{
    return BUILTIN_MIN_F64(BUILTIN_CANONICALIZE_F64(x), BUILTIN_CANONICALIZE_F64(y));
}

