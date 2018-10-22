/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(fmin)(float x, float y)
{
    float ret;

    if (DAZ_OPT() & !FINITE_ONLY_OPT()) {
        // XXX revisit this later
        ret = BUILTIN_CMIN_F32(x, y);
    } else {
        if (FINITE_ONLY_OPT()) {
            ret = BUILTIN_MIN_F32(x, y);
        } else {
            ret = BUILTIN_MIN_F32(x, y);
        }
    }

    return ret;
}

