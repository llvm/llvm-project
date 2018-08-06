/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR int
MATH_MANGLE(fpclassify)(double x)
{
    int ret = BUILTIN_ISINF_F64(x) ? FP_INFINITE : FP_NAN;
    ret = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_NZER) ? FP_ZERO : ret;
    ret = BUILTIN_CLASS_F64(x, CLASS_PSUB|CLASS_NSUB) ? FP_SUBNORMAL : ret;
    ret = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_NNOR) ? FP_NORMAL : ret;
    return ret;
}

