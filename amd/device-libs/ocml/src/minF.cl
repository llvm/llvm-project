/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(min)(float x, float y)
{
    if (AMD_OPT()) {
        return BUILTIN_CMIN_F32(x, y);
    } else {
        return BUILTIN_MIN_F32(x, y);
    }
}

