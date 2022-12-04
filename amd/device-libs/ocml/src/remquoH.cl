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

__ocml_remquo_2f16_result
MATH_MANGLE2(remquo2)(half2 x, half2 y)
{
    __ocml_remquo_f16_result lo = MATH_MANGLE(remquo2)(x.lo, y.lo);
    __ocml_remquo_f16_result hi = MATH_MANGLE(remquo2)(x.hi, y.hi);
    __ocml_remquo_2f16_result result = { (half2)(lo.rem, hi.rem),
                                         (int2)(lo.quo, hi.quo) };
    return result;
}

#define COMPILING_REMQUO
#include "remainderH_base.h"

