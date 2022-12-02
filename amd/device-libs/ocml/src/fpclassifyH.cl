/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS CONSTATTR int
MATH_MANGLE(fpclassify)(half x)
{
    int ret = BUILTIN_ISINF_F16(x) ? FP_INFINITE : FP_NAN;
    ret = BUILTIN_ISZERO_F16(x) ? FP_ZERO : ret;
    ret = BUILTIN_ISSUBNORMAL_F16(x) ? FP_SUBNORMAL : ret;
    ret = BUILTIN_ISNORMAL_F16(x) ? FP_NORMAL : ret;
    return ret;
}

