/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(double x, double y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(double x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(double x, int ny)
#else
MATH_MANGLE(pow)(double x, double y)
#endif
{
    USE_TABLE(double2, p_ln_tbl, M64_POWLOG);
    USE_TABLE(double2, p_Finv_tbl, M64_LOG_F_INV);
    USE_TABLE(double2, p_2to_tbl, M64_EXP_EP);

#if defined(COMPILING_POWN)
    double y = (double) ny;
#elif defined(COMPILING_ROOTN)
    double dny = (double)ny;
    double y = MATH_DIV(1.0,  dny);
#endif

    double ax = BUILTIN_ABS_F64(x);

    // Extended precision y * ln(x)
    double ylnx_h, ylnx_t;
    {
        int e = (AS_INT2(ax).hi >> 20) - EXPBIAS_DP64;
        bool c = e == -1023;
        double tx = AS_DOUBLE(ONEEXPBITS_DP64 | AS_LONG(ax)) - 1.0;
        int es = (AS_INT2(tx).hi >> 20) - (2*EXPBIAS_DP64 - 1);
        e = c ? es : e;

        long mant = AS_LONG(c ? tx : ax) & MANTBITS_DP64;
        int index = AS_INT2(mant).hi;
        index = (index & 0x000ff000) + ((index & 0x00000800) << 1);
        double F = AS_DOUBLE(HALFEXPBITS_DP64 | (long)index << 32);
        double f = F - AS_DOUBLE(HALFEXPBITS_DP64 | mant);
        index >>= 12;

        double2 tv = p_Finv_tbl[index];
        double inv_h = tv.s0;
        double inv_t = tv.s1;
        double f_inv = (inv_h + inv_t) * f;
        double r1 = AS_DOUBLE(AS_LONG(f_inv) & 0xfffffffff8000000L);
        double r2 = MATH_MAD(-F, r1, f) * (inv_h + inv_t);
        double r = r1 + r2;

        double poly = MATH_MAD(r,
                          MATH_MAD(r,
                              MATH_MAD(r,
                                  MATH_MAD(r, 1.0/7.0, 1.0/6.0),
                                  1.0/5.0),
                              1.0/4.0),
                          1.0/3.0);
        poly = poly * r * r * r;

        double hr1r1 = 0.5*r1*r1;
        double poly0h = r1 + hr1r1;
        double poly0t = r1 - poly0h + hr1r1;
	poly = MATH_MAD(r1, r2, MATH_MAD(0.5*r2, r2, poly)) + r2 + poly0t;

        tv = p_ln_tbl[index];
        double lnF_h = tv.s0;
        double lnF_t = tv.s1;

        const double ln2_h = 0x1.62e42e0000000p-1;
        const double ln2_t = 0x1.efa39ef35793cp-25;

        double de = (double)e;
        double tlnx_tt = MATH_MAD(de, ln2_t, lnF_t) - poly;
        double tlnx_t = tlnx_tt - poly0h;
        double tlnx_h = MATH_MAD(de, ln2_h, lnF_h);
        double tlnx_t_h = poly0h;

        double lnx_h = tlnx_t + tlnx_h;
        double lnx_hh = AS_DOUBLE(AS_LONG(lnx_h) & 0xfffffffff8000000L);
        double lnx_t = (tlnx_h - lnx_h + tlnx_t) + (tlnx_tt - (tlnx_t + tlnx_t_h)) + (lnx_h - lnx_hh);
        lnx_h = lnx_hh;

        double y_h = AS_DOUBLE(AS_LONG(y) & 0xfffffffff8000000L);
        double y_t = y - y_h;

#if defined(COMPILING_POWN)
        y_t = (double)(ny - (int)y_h);
#endif

#if defined(COMPILING_ROOTN)
        double dnyh = AS_DOUBLE(AS_LONG(dny) & 0xfffffffffff00000L);
        double dnyt = (double)(ny - (int)dnyh);
        y_t = MATH_DIV(MATH_MAD(-dnyt, y_h, MATH_MAD(-dnyh, y_h, 1.0)), dny);
#endif

        ylnx_t = MATH_MAD(y_t, lnx_h, MATH_MAD(y_h, lnx_t, y_t*lnx_t));
        ylnx_h = MATH_MAD(y_h, lnx_h, ylnx_t);
        ylnx_t = MATH_MAD(y_h, lnx_h, -ylnx_h) + ylnx_t;
    }

    // Now calculate exp of (ylnx_h,ylnx_t)

    double expylnx;
    {
        const double c64byln2 = 0x1.71547652b82fep+6;
        const double ln2by64_h = 0x1.62e42f0000000p-7;
        const double ln2by64_t = -0x1.df473de6af278p-32;

        double dn = BUILTIN_TRUNC_F64(ylnx_h * c64byln2);
        int n = (int)dn;
        int j = n & 0x0000003f;
        int m = n >> 6;

        double2 tv = p_2to_tbl[j];
        double f1 = tv.s0;
        double f2 = tv.s1;
        double f = f1 + f2;

        double r1 = MATH_MAD(dn, -ln2by64_h, ylnx_h);
        double r2 = dn * ln2by64_t;
        double r = (r1 + r2) + ylnx_t;

        double q = MATH_MAD(r,
                       MATH_MAD(r,
                           MATH_MAD(r,
                               MATH_MAD(r, 1.38889490863777199667e-03, 8.33336798434219616221e-03),
                               4.16666666662260795726e-02),
                           1.66666666665260878863e-01),
                       5.00000000000000008883e-01);
        q = MATH_MAD(r*r, q, r);

        expylnx = MATH_MAD(f, q, f2) + f1;

        if (AMD_OPT()) {
            expylnx = BUILTIN_FLDEXP_F64(expylnx, m);
        } else {
            int mh = m >> 1;
            expylnx = expylnx * AS_DOUBLE((long)(mh + 1023) << 52) *
                                AS_DOUBLE((long)(m - mh + 1023) << 52);
        }

        const double max_exp_arg = 0x1.62e42fefa39efp+9;
        const double min_exp_arg = -0x1.74910d52d3051p+9;
        expylnx = ylnx_h > max_exp_arg ? AS_DOUBLE(PINFBITPATT_DP64) : expylnx;
        expylnx = ylnx_h < min_exp_arg ? 0.0 : expylnx;
    }

    // See whether y is an integer.
    // inty = 0 means not an integer.
    // inty = 1 means odd integer.
    // inty = 2 means even integer.

#if defined(COMPILING_POWN) | defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    double ay = BUILTIN_ABS_F64(y);
    int inty;
    {
        double tay = BUILTIN_TRUNC_F64(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F64(tay*0.5) == 0.0);
    }
#endif

    double ret = BUILTIN_COPYSIGN_F64(expylnx, (inty == 1) & (x < 0.0) ? -0.0 : 0.0);

    // Now all the edge cases
#if defined COMPILING_POWR
    bool ax_eq_0 = ax == 0.0;
    bool ax_ne_0 = ax != 0.0;
    bool ax_lt_1 = ax < 1.0;
    bool ax_eq_1 = ax == 1.0;
    bool ax_gt_1 = ax > 1.0;
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool ay_eq_0 = ay == 0.0;
    bool ay_eq_pinf = BUILTIN_CLASS_F64(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F64(ay, CLASS_QNAN|CLASS_SNAN);
    bool y_eq_ninf = BUILTIN_CLASS_F64(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F64(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F64(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F64(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = ax_lt_1 & y_eq_ninf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0 : ret;
        ret = ax_eq_1 & ay_lt_inf ? 1.0 : ret;
        ret = ax_eq_1 & ay_eq_pinf ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0 : ret;
        ret = ax_gt_1 & y_eq_pinf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_lt_pinf & ay_eq_0 ? 1.0 : ret;
        ret = ax_eq_pinf & !y_pos ? 0.0 : ret;
        ret = ax_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_pinf & y_eq_pinf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_pinf & ay_eq_0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_eq_0 & !y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_0 & y_pos ? 0.0 : ret;
        ret = ax_eq_0 & ay_eq_0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_ne_0 & !x_pos ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
	ret = ax_eq_1 ? 1.0 : ret;
	ret = ay_eq_0 ? 1.0 : ret;
	ret = ax_eq_0 & y_pos ? 0.0 : ret;
    }
#elif defined COMPILING_POWN
    bool ax_eq_0 = ax == 0.0;
    bool x_eq_ninf = BUILTIN_CLASS_F64(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        double xinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0 : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0 : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0 : ret;
        ret = x_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
    } else {
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
    }
    ret = ny == 0 ? 1.0 : ret;
#elif defined COMPILING_ROOTN
    bool ax_eq_0 = ax == 0.0;
    bool x_eq_ninf = BUILTIN_CLASS_F64(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 2) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        double xinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0 : ret;
        ret = x_eq_pinf & !y_pos ? 0.0 : ret;
        ret = x_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ny == 0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
    } else {
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
    }
#else
    bool ax_eq_0 = ax == 0.0;
    bool ax_ne_0 = ax != 0.0;
    bool ax_lt_1 = ax < 1.0;
    bool ax_eq_1 = ax == 1.0;
    bool ax_gt_1 = ax > 1.0;
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool x_eq_ninf = BUILTIN_CLASS_F64(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ay_eq_0 = ay == 0.0;
    bool ay_eq_pinf = BUILTIN_CLASS_F64(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F64(ay, CLASS_QNAN|CLASS_SNAN);
    bool y_eq_ninf = BUILTIN_CLASS_F64(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F64(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F64(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F64(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 0) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_lt_1 & y_eq_ninf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0 : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0 : ret;
        ret = ax_gt_1 & y_eq_pinf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        double xinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty != 1) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0 : ret;
        ret = ax_eq_0 & y_eq_ninf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = (x == -1.0) & ay_eq_pinf ? 1.0 : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0 : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0 : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0 : ret;
        ret = x_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0 : ret;
    }
    ret = ay == 0.0 ? 1.0 : ret;
    ret = x == 1.0 ? 1.0 : ret;
#endif

    return ret;
}

