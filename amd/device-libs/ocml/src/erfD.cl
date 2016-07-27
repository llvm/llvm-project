/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(erf)(double x)
{
    USE_TABLE(double, c, M64_ERF);
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 2.2) {
        double t = ax * ax;
        int k = (int)t;
        t = BUILTIN_FRACTION_F64(t);
        c += k*13;

        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              c[0] , c[1]), c[2]), c[3]), c[4]),
                     c[5]), c[6]), c[7]), c[8]),
                     c[9]), c[10]), c[11]), c[12]) * ax;
    } else if (ax < 5.925) {
        int k = (int)ax;
        double t = BUILTIN_FRACTION_F64(ax);
        c += 65 + (k-2)*13;

        double y;
        y = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
            MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
            MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
            c[0] , c[1]), c[2]), c[3]), c[4]),
                   c[5]), c[6]), c[7]), c[8]),
                   c[9]), c[10]), c[11]), c[12]);
        y *= y;
        y *= y;
        y *= y;
        ret = MATH_MAD(-y, y, 1.0);
    } else {
        ret = 1.0;
    }

    ret = BUILTIN_COPYSIGN_F64(ret, x);
    ret = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? x : ret;
    return ret;
}

