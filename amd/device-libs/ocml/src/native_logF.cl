/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(native_log)(float x)
{
    return __builtin_logf(x);
}

CONSTATTR float
MATH_MANGLE(native_log2)(float x)
{
    return __builtin_log2f(x);
}

CONSTATTR float
MATH_MANGLE(native_log10)(float x)
{
    return __builtin_log10f(x);
}
