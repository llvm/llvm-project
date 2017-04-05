/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR INLINEATTR double
MATH_MANGLE(rsqrt)(double x)
{
    double ret;

    if (AMD_OPT()) {
        double y0 = BUILTIN_RSQRT_F64(x);
        double e = MATH_MAD(-x*y0, y0, 1.0);
        double y1 = MATH_MAD(y0*e, MATH_MAD(e, 0.375, 0.5), y0);
        ret = BUILTIN_CLASS_F64(y0, CLASS_PSUB|CLASS_PNOR) ? y1 : y0;
    } else {
        USE_TABLE(double, p_tbl, M64_RSQRT);
        double y = x * (x < 0x1.0p-1000 ? 0x1.0p+1000 : 1.0);
        int e = (AS_INT2(y).hi >> 20) - EXPBIAS_DP64;
        int i = ((e & 1) << 6) + ((AS_INT2(y).hi >> 14) & 0x3f);
        double r = p_tbl[i] * AS_DOUBLE((long)(EXPBIAS_DP64 - (e >> 1)) << EXPSHIFTBITS_DP64);
        r = r * MATH_MAD(-y*r*0.5, r, 1.5);
        r = r * MATH_MAD(-y*r*0.5, r, 1.5);
        r = r * MATH_MAD(-y*r*0.5, r, 1.5);
        r = r * MATH_MAD(-y*r*0.5, r, 1.5);
        ret = r * (x < 0x1.0p-1000 ? 0x1.0p+500 : 1.0);
        if (!FINITE_ONLY_OPT()) {
            double inf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
            ret = x == 0.0 ? inf : ret;
            ret = BUILTIN_CLASS_F64(x, CLASS_PINF) ? 0.0 : ret;
            ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_NNOR|CLASS_NSUB) ?
                  AS_DOUBLE(QNANBITPATT_DP64) : ret;
        }
    }

    return ret;
}

