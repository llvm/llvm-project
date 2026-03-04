/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

half2
MATH_MANGLE2(lgamma_r)(half2 x, __private int2 *signp)
{
    int slo, shi;
    half2 r;
    r.lo = MATH_MANGLE(lgamma_r)(x.lo, &slo);
    r.hi = MATH_MANGLE(lgamma_r)(x.hi, &shi);
    *signp = (int2)(slo, shi);
    return r;
}

half
MATH_MANGLE(lgamma_r)(half x, __private int *signp)
{
    return (half)MATH_UPMANGLE(lgamma_r)((float)x, signp);
}

