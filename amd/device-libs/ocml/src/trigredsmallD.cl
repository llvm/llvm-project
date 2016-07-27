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
    int ret;

    if (!HAVE_FAST_FMA64()) {
        const double twobypi =  0x1.45f306dc9c883p-1;
        const double piby2_h =  0x1.921fb54400000p+0;
        const double piby2_ht = 0x1.0b4611a626331p-34;
        const double piby2_m =  0x1.0b4611a600000p-34;
        const double piby2_mt = 0x1.3198a2e037073p-69;
        const double piby2_t =  0x1.3198a2e000000p-69;
        const double piby2_tt = 0x1.b839a252049c1p-104;

        int nc = x * twobypi + 0.5;
        int n;
        n = x > 0x1.921fb54442d17p-1 ? 1 : 0;
        n = x > 0x1.2d97c7f3321d2p+1 ? 2 : n;
        n = x > 0x1.f6a7a2955385ep+1 ? 3 : n;
        n = x > 0x1.5fdbbe9bba775p+2 ? 4 : n;
        n = x > 0x1.c463abeccb2bbp+2 ? nc : n;
        double dn = (double)n;

        // Subtract the multiple from x to get an extra-precision remainder
        double rh = x - dn * piby2_h;
        double rt = dn * piby2_ht;

        double t = rh;
        rt = dn * piby2_m;
        rh = t - rt;
        rt = dn * piby2_mt - ((t - rh) - rt);

        t = rh;
        rt = dn * piby2_t;
        rh = t - rt;
        rt = dn * piby2_tt - ((t - rh) - rt);

        t = rh - rt;
        *r = t;
        *rr = (rh - t) - rt;
        ret = n & 3;
    } else {
        const double shift = 0x1.8p+52;
        const double twobypi = 0x1.45f306dc9c883p-1;
        const double piby2_h = 0x1.921fb54442d18p+0;
        const double piby2_m = 0x1.1a62633145c00p-54;
        const double piby2_t = 0x1.b839a252049c0p-104;

        double dn = BUILTIN_FMA_F64(x, twobypi, shift) - shift;
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
        ret = (int)dn & 0x3;
    }

    return ret;
}

