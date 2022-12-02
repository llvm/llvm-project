/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR int
MATH_MANGLE(fpclassify)(float x)
{
    int ret = BUILTIN_ISINF_F32(x) ? FP_INFINITE : FP_NAN;
    ret = BUILTIN_ISZERO_F32(x) ? FP_ZERO : ret;
    ret = BUILTIN_ISSUBNORMAL_F32(x) ? FP_SUBNORMAL : ret;
    ret = BUILTIN_ISNORMAL_F32(x) ? FP_NORMAL : ret;
    return ret;
}

