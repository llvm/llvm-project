/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

half2
MATH_MANGLE2(remquo)(half2 x, half2 y, __private int2 *q7p)
{
    int qlo, qhi;
    half2 r;
    r.lo = MATH_MANGLE(remquo)(x.lo, y.lo, &qlo);
    r.hi = MATH_MANGLE(remquo)(x.hi, y.hi, &qhi);
    *q7p = (int2)(qlo, qhi);
    return r;
}

#define COMPILING_REMQUO
#include "remainderH_base.h"

