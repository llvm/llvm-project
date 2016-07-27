/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(isnan)(float x)
{
    return BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN);
}
