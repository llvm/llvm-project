/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

INLINEATTR int
MATH_PRIVATE(trigredsmall)(__private double *r, __private double *rr, double x)
{
    const double twobypi = 0x1.45f306dc9c883p-1;
    const double piby2_h = 0x1.921fb54442d18p+0;
    const double piby2_m = 0x1.1a62633145c00p-54;
    const double piby2_t = 0x1.b839a252049c0p-104;

    double dn = BUILTIN_RINT_F64(x * twobypi);
    double xt = BUILTIN_FMA_F64(dn, -piby2_h, x);
    double yh = BUILTIN_FMA_F64(dn, -piby2_m, xt);
    double ph = dn * piby2_m;
    double pt = BUILTIN_FMA_F64(dn, piby2_m, -ph);
    double th = xt - ph;
    double tt = (xt - th) - ph;
    double yt = BUILTIN_FMA_F64(dn, -piby2_t, ((th - yh) + tt) - pt);
    double rh = yh + yt;
    double rt = yt - (rh - yh);

    *r = rh;
    *rr = rt;

    return (int)dn & 0x3;
}

