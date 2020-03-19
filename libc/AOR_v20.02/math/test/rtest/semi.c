/*
 * semi.c: test implementations of mathlib seminumerical functions
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include "semi.h"

static void test_rint(uint32 *in, uint32 *out,
                       int isfloor, int isceil) {
    int sign = in[0] & 0x80000000;
    int roundup = (isfloor && sign) || (isceil && !sign);
    uint32 xh, xl, roundword;
    int ex = (in[0] >> 20) & 0x7FF;    /* exponent */
    int i;

    if ((ex > 0x3ff + 52 - 1) ||     /* things this big can't be fractional */
        ((in[0] & 0x7FFFFFFF) == 0 && in[1] == 0)) {   /* zero */
        /* NaN, Inf, a large integer, or zero: just return the input */
        out[0] = in[0];
        out[1] = in[1];
        return;
    }

    /*
     * Special case: ex < 0x3ff, ie our number is in (0,1). Return
     * 1 or 0 according to roundup.
     */
    if (ex < 0x3ff) {
        out[0] = sign | (roundup ? 0x3FF00000 : 0);
        out[1] = 0;
        return;
    }

    /*
     * We're not short of time here, so we'll do this the hideously
     * inefficient way. Shift bit by bit so that the units place is
     * somewhere predictable, round, and shift back again.
     */
    xh = in[0];
    xl = in[1];
    roundword = 0;
    for (i = ex; i < 0x3ff + 52; i++) {
        if (roundword & 1)
            roundword |= 2;            /* preserve sticky bit */
        roundword = (roundword >> 1) | ((xl & 1) << 31);
        xl = (xl >> 1) | ((xh & 1) << 31);
        xh = xh >> 1;
    }
    if (roundword && roundup) {
        xl++;
        xh += (xl==0);
    }
    for (i = ex; i < 0x3ff + 52; i++) {
        xh = (xh << 1) | ((xl >> 31) & 1);
        xl = (xl & 0x7FFFFFFF) << 1;
    }
    out[0] = xh;
    out[1] = xl;
}

char *test_ceil(uint32 *in, uint32 *out) {
    test_rint(in, out, 0, 1);
    return NULL;
}

char *test_floor(uint32 *in, uint32 *out) {
    test_rint(in, out, 1, 0);
    return NULL;
}

static void test_rintf(uint32 *in, uint32 *out,
                       int isfloor, int isceil) {
    int sign = *in & 0x80000000;
    int roundup = (isfloor && sign) || (isceil && !sign);
    uint32 x, roundword;
    int ex = (*in >> 23) & 0xFF;       /* exponent */
    int i;

    if ((ex > 0x7f + 23 - 1) ||      /* things this big can't be fractional */
        (*in & 0x7FFFFFFF) == 0) {     /* zero */
        /* NaN, Inf, a large integer, or zero: just return the input */
        *out = *in;
        return;
    }

    /*
     * Special case: ex < 0x7f, ie our number is in (0,1). Return
     * 1 or 0 according to roundup.
     */
    if (ex < 0x7f) {
        *out = sign | (roundup ? 0x3F800000 : 0);
        return;
    }

    /*
     * We're not short of time here, so we'll do this the hideously
     * inefficient way. Shift bit by bit so that the units place is
     * somewhere predictable, round, and shift back again.
     */
    x = *in;
    roundword = 0;
    for (i = ex; i < 0x7F + 23; i++) {
        if (roundword & 1)
            roundword |= 2;            /* preserve sticky bit */
        roundword = (roundword >> 1) | ((x & 1) << 31);
        x = x >> 1;
    }
    if (roundword && roundup) {
        x++;
    }
    for (i = ex; i < 0x7F + 23; i++) {
        x = x << 1;
    }
    *out = x;
}

char *test_ceilf(uint32 *in, uint32 *out) {
    test_rintf(in, out, 0, 1);
    return NULL;
}

char *test_floorf(uint32 *in, uint32 *out) {
    test_rintf(in, out, 1, 0);
    return NULL;
}

char *test_fmod(uint32 *a, uint32 *b, uint32 *out) {
    int sign;
    int32 aex, bex;
    uint32 am[2], bm[2];

    if (((a[0] & 0x7FFFFFFF) << 1) + !!a[1] > 0xFFE00000 ||
        ((b[0] & 0x7FFFFFFF) << 1) + !!b[1] > 0xFFE00000) {
        /* a or b is NaN: return QNaN, optionally with IVO */
        uint32 an, bn;
        out[0] = 0x7ff80000;
        out[1] = 1;
        an = ((a[0] & 0x7FFFFFFF) << 1) + !!a[1];
        bn = ((b[0] & 0x7FFFFFFF) << 1) + !!b[1];
        if ((an > 0xFFE00000 && an < 0xFFF00000) ||
            (bn > 0xFFE00000 && bn < 0xFFF00000))
            return "i";                /* at least one SNaN: IVO */
        else
            return NULL;               /* no SNaNs, but at least 1 QNaN */
    }
    if ((b[0] & 0x7FFFFFFF) == 0 && b[1] == 0) {   /* b==0: EDOM */
        out[0] = 0x7ff80000;
        out[1] = 1;
        return "EDOM status=i";
    }
    if ((a[0] & 0x7FF00000) == 0x7FF00000) {   /* a==Inf: EDOM */
        out[0] = 0x7ff80000;
        out[1] = 1;
        return "EDOM status=i";
    }
    if ((b[0] & 0x7FF00000) == 0x7FF00000) {   /* b==Inf: return a */
        out[0] = a[0];
        out[1] = a[1];
        return NULL;
    }
    if ((a[0] & 0x7FFFFFFF) == 0 && a[1] == 0) {   /* a==0: return a */
        out[0] = a[0];
        out[1] = a[1];
        return NULL;
    }

    /*
     * OK. That's the special cases cleared out of the way. Now we
     * have finite (though not necessarily normal) a and b.
     */
    sign = a[0] & 0x80000000;          /* we discard sign of b */
    test_frexp(a, am, (uint32 *)&aex);
    test_frexp(b, bm, (uint32 *)&bex);
    am[0] &= 0xFFFFF, am[0] |= 0x100000;
    bm[0] &= 0xFFFFF, bm[0] |= 0x100000;

    while (aex >= bex) {
        if (am[0] > bm[0] || (am[0] == bm[0] && am[1] >= bm[1])) {
            am[1] -= bm[1];
            am[0] = am[0] - bm[0] - (am[1] > ~bm[1]);
        }
        if (aex > bex) {
            am[0] = (am[0] << 1) | ((am[1] & 0x80000000) >> 31);
            am[1] <<= 1;
            aex--;
        } else
            break;
    }

    /*
     * Renormalise final result; this can be cunningly done by
     * passing a denormal to ldexp.
     */
    aex += 0x3fd;
    am[0] |= sign;
    test_ldexp(am, (uint32 *)&aex, out);

    return NULL;                       /* FIXME */
}

char *test_fmodf(uint32 *a, uint32 *b, uint32 *out) {
    int sign;
    int32 aex, bex;
    uint32 am, bm;

    if ((*a & 0x7FFFFFFF) > 0x7F800000 ||
        (*b & 0x7FFFFFFF) > 0x7F800000) {
        /* a or b is NaN: return QNaN, optionally with IVO */
        uint32 an, bn;
        *out = 0x7fc00001;
        an = *a & 0x7FFFFFFF;
        bn = *b & 0x7FFFFFFF;
        if ((an > 0x7f800000 && an < 0x7fc00000) ||
            (bn > 0x7f800000 && bn < 0x7fc00000))
            return "i";                /* at least one SNaN: IVO */
        else
            return NULL;               /* no SNaNs, but at least 1 QNaN */
    }
    if ((*b & 0x7FFFFFFF) == 0) {      /* b==0: EDOM */
        *out = 0x7fc00001;
        return "EDOM status=i";
    }
    if ((*a & 0x7F800000) == 0x7F800000) {   /* a==Inf: EDOM */
        *out = 0x7fc00001;
        return "EDOM status=i";
    }
    if ((*b & 0x7F800000) == 0x7F800000) {   /* b==Inf: return a */
        *out = *a;
        return NULL;
    }
    if ((*a & 0x7FFFFFFF) == 0) {      /* a==0: return a */
        *out = *a;
        return NULL;
    }

    /*
     * OK. That's the special cases cleared out of the way. Now we
     * have finite (though not necessarily normal) a and b.
     */
    sign = a[0] & 0x80000000;          /* we discard sign of b */
    test_frexpf(a, &am, (uint32 *)&aex);
    test_frexpf(b, &bm, (uint32 *)&bex);
    am &= 0x7FFFFF, am |= 0x800000;
    bm &= 0x7FFFFF, bm |= 0x800000;

    while (aex >= bex) {
        if (am >= bm) {
            am -= bm;
        }
        if (aex > bex) {
            am <<= 1;
            aex--;
        } else
            break;
    }

    /*
     * Renormalise final result; this can be cunningly done by
     * passing a denormal to ldexp.
     */
    aex += 0x7d;
    am |= sign;
    test_ldexpf(&am, (uint32 *)&aex, out);

    return NULL;                       /* FIXME */
}

char *test_ldexp(uint32 *x, uint32 *np, uint32 *out) {
    int n = *np;
    int32 n2;
    uint32 y[2];
    int ex = (x[0] >> 20) & 0x7FF;     /* exponent */
    int sign = x[0] & 0x80000000;

    if (ex == 0x7FF) {                 /* inf/NaN; just return x */
        out[0] = x[0];
        out[1] = x[1];
        return NULL;
    }
    if ((x[0] & 0x7FFFFFFF) == 0 && x[1] == 0) {   /* zero: return x */
        out[0] = x[0];
        out[1] = x[1];
        return NULL;
    }

    test_frexp(x, y, (uint32 *)&n2);
    ex = n + n2;
    if (ex > 0x400) {                  /* overflow */
        out[0] = sign | 0x7FF00000;
        out[1] = 0;
        return "overflow";
    }
    /*
     * Underflow. 2^-1074 is 00000000.00000001; so if ex == -1074
     * then we have something [2^-1075,2^-1074). Under round-to-
     * nearest-even, this whole interval rounds up to 2^-1074,
     * except for the bottom endpoint which rounds to even and is
     * an underflow condition.
     *
     * So, ex < -1074 is definite underflow, and ex == -1074 is
     * underflow iff all mantissa bits are zero.
     */
    if (ex < -1074 || (ex == -1074 && (y[0] & 0xFFFFF) == 0 && y[1] == 0)) {
        out[0] = sign;                 /* underflow: correctly signed zero */
        out[1] = 0;
        return "underflow";
    }

    /*
     * No overflow or underflow; should be nice and simple, unless
     * we have to denormalise and round the result.
     */
    if (ex < -1021) {                  /* denormalise and round */
        uint32 roundword;
        y[0] &= 0x000FFFFF;
        y[0] |= 0x00100000;            /* set leading bit */
        roundword = 0;
        while (ex < -1021) {
            if (roundword & 1)
                roundword |= 2;        /* preserve sticky bit */
            roundword = (roundword >> 1) | ((y[1] & 1) << 31);
            y[1] = (y[1] >> 1) | ((y[0] & 1) << 31);
            y[0] = y[0] >> 1;
            ex++;
        }
        if (roundword > 0x80000000 ||  /* round up */
            (roundword == 0x80000000 && (y[1] & 1))) {  /* round up to even */
            y[1]++;
            y[0] += (y[1] == 0);
        }
        out[0] = sign | y[0];
        out[1] = y[1];
        /* Proper ERANGE underflow was handled earlier, but we still
         * expect an IEEE Underflow exception if this partially
         * underflowed result is not exact. */
        if (roundword)
            return "u";
        return NULL;                   /* underflow was handled earlier */
    } else {
        out[0] = y[0] + (ex << 20);
        out[1] = y[1];
        return NULL;
    }
}

char *test_ldexpf(uint32 *x, uint32 *np, uint32 *out) {
    int n = *np;
    int32 n2;
    uint32 y;
    int ex = (*x >> 23) & 0xFF;     /* exponent */
    int sign = *x & 0x80000000;

    if (ex == 0xFF) {                 /* inf/NaN; just return x */
        *out = *x;
        return NULL;
    }
    if ((*x & 0x7FFFFFFF) == 0) {      /* zero: return x */
        *out = *x;
        return NULL;
    }

    test_frexpf(x, &y, (uint32 *)&n2);
    ex = n + n2;
    if (ex > 0x80) {                  /* overflow */
        *out = sign | 0x7F800000;
        return "overflow";
    }
    /*
     * Underflow. 2^-149 is 00000001; so if ex == -149 then we have
     * something [2^-150,2^-149). Under round-to- nearest-even,
     * this whole interval rounds up to 2^-149, except for the
     * bottom endpoint which rounds to even and is an underflow
     * condition.
     *
     * So, ex < -149 is definite underflow, and ex == -149 is
     * underflow iff all mantissa bits are zero.
     */
    if (ex < -149 || (ex == -149 && (y & 0x7FFFFF) == 0)) {
        *out = sign;                 /* underflow: correctly signed zero */
        return "underflow";
    }

    /*
     * No overflow or underflow; should be nice and simple, unless
     * we have to denormalise and round the result.
     */
    if (ex < -125) {                  /* denormalise and round */
        uint32 roundword;
        y &= 0x007FFFFF;
        y |= 0x00800000;               /* set leading bit */
        roundword = 0;
        while (ex < -125) {
            if (roundword & 1)
                roundword |= 2;        /* preserve sticky bit */
            roundword = (roundword >> 1) | ((y & 1) << 31);
            y = y >> 1;
            ex++;
        }
        if (roundword > 0x80000000 ||  /* round up */
            (roundword == 0x80000000 && (y & 1))) {  /* round up to even */
            y++;
        }
        *out = sign | y;
        /* Proper ERANGE underflow was handled earlier, but we still
         * expect an IEEE Underflow exception if this partially
         * underflowed result is not exact. */
        if (roundword)
            return "u";
        return NULL;                   /* underflow was handled earlier */
    } else {
        *out = y + (ex << 23);
        return NULL;
    }
}

char *test_frexp(uint32 *x, uint32 *out, uint32 *nout) {
    int ex = (x[0] >> 20) & 0x7FF;     /* exponent */
    if (ex == 0x7FF) {                 /* inf/NaN; return x/0 */
        out[0] = x[0];
        out[1] = x[1];
        nout[0] = 0;
        return NULL;
    }
    if (ex == 0) {                     /* denormals/zeros */
        int sign;
        uint32 xh, xl;
        if ((x[0] & 0x7FFFFFFF) == 0 && x[1] == 0) {
            /* zero: return x/0 */
            out[0] = x[0];
            out[1] = x[1];
            nout[0] = 0;
            return NULL;
        }
        sign = x[0] & 0x80000000;
        xh = x[0] & 0x7FFFFFFF;
        xl = x[1];
        ex = 1;
        while (!(xh & 0x100000)) {
            ex--;
            xh = (xh << 1) | ((xl >> 31) & 1);
            xl = (xl & 0x7FFFFFFF) << 1;
        }
        out[0] = sign | 0x3FE00000 | (xh & 0xFFFFF);
        out[1] = xl;
        nout[0] = ex - 0x3FE;
        return NULL;
    }
    out[0] = 0x3FE00000 | (x[0] & 0x800FFFFF);
    out[1] = x[1];
    nout[0] = ex - 0x3FE;
    return NULL;                       /* ordinary number; no error */
}

char *test_frexpf(uint32 *x, uint32 *out, uint32 *nout) {
    int ex = (*x >> 23) & 0xFF;        /* exponent */
    if (ex == 0xFF) {                  /* inf/NaN; return x/0 */
        *out = *x;
        nout[0] = 0;
        return NULL;
    }
    if (ex == 0) {                     /* denormals/zeros */
        int sign;
        uint32 xv;
        if ((*x & 0x7FFFFFFF) == 0) {
            /* zero: return x/0 */
            *out = *x;
            nout[0] = 0;
            return NULL;
        }
        sign = *x & 0x80000000;
        xv = *x & 0x7FFFFFFF;
        ex = 1;
        while (!(xv & 0x800000)) {
            ex--;
            xv = xv << 1;
        }
        *out = sign | 0x3F000000 | (xv & 0x7FFFFF);
        nout[0] = ex - 0x7E;
        return NULL;
    }
    *out = 0x3F000000 | (*x & 0x807FFFFF);
    nout[0] = ex - 0x7E;
    return NULL;                       /* ordinary number; no error */
}

char *test_modf(uint32 *x, uint32 *fout, uint32 *iout) {
    int ex = (x[0] >> 20) & 0x7FF;     /* exponent */
    int sign = x[0] & 0x80000000;
    uint32 fh, fl;

    if (((x[0] & 0x7FFFFFFF) | (!!x[1])) > 0x7FF00000) {
        /*
         * NaN input: return the same in _both_ outputs.
         */
        fout[0] = iout[0] = x[0];
        fout[1] = iout[1] = x[1];
        return NULL;
    }

    test_rint(x, iout, 0, 0);
    fh = x[0] - iout[0];
    fl = x[1] - iout[1];
    if (!fh && !fl) {                  /* no fraction part */
        fout[0] = sign;
        fout[1] = 0;
        return NULL;
    }
    if (!(iout[0] & 0x7FFFFFFF) && !iout[1]) {   /* no integer part */
        fout[0] = x[0];
        fout[1] = x[1];
        return NULL;
    }
    while (!(fh & 0x100000)) {
        ex--;
        fh = (fh << 1) | ((fl >> 31) & 1);
        fl = (fl & 0x7FFFFFFF) << 1;
    }
    fout[0] = sign | (ex << 20) | (fh & 0xFFFFF);
    fout[1] = fl;
    return NULL;
}

char *test_modff(uint32 *x, uint32 *fout, uint32 *iout) {
    int ex = (*x >> 23) & 0xFF;        /* exponent */
    int sign = *x & 0x80000000;
    uint32 f;

    if ((*x & 0x7FFFFFFF) > 0x7F800000) {
        /*
         * NaN input: return the same in _both_ outputs.
         */
        *fout = *iout = *x;
        return NULL;
    }

    test_rintf(x, iout, 0, 0);
    f = *x - *iout;
    if (!f) {                          /* no fraction part */
        *fout = sign;
        return NULL;
    }
    if (!(*iout & 0x7FFFFFFF)) {       /* no integer part */
        *fout = *x;
        return NULL;
    }
    while (!(f & 0x800000)) {
        ex--;
        f = f << 1;
    }
    *fout = sign | (ex << 23) | (f & 0x7FFFFF);
    return NULL;
}

char *test_copysign(uint32 *x, uint32 *y, uint32 *out)
{
    int ysign = y[0] & 0x80000000;
    int xhigh = x[0] & 0x7fffffff;

    out[0] = ysign | xhigh;
    out[1] = x[1];

    /* There can be no error */
    return NULL;
}

char *test_copysignf(uint32 *x, uint32 *y, uint32 *out)
{
    int ysign = y[0] & 0x80000000;
    int xhigh = x[0] & 0x7fffffff;

    out[0] = ysign | xhigh;

    /* There can be no error */
    return NULL;
}

char *test_isfinite(uint32 *x, uint32 *out)
{
    int xhigh = x[0];
    /* Being finite means that the exponent is not 0x7ff */
    if ((xhigh & 0x7ff00000) == 0x7ff00000) out[0] = 0;
    else out[0] = 1;
    return NULL;
}

char *test_isfinitef(uint32 *x, uint32 *out)
{
    /* Being finite means that the exponent is not 0xff */
    if ((x[0] & 0x7f800000) == 0x7f800000) out[0] = 0;
    else out[0] = 1;
    return NULL;
}

char *test_isinff(uint32 *x, uint32 *out)
{
    /* Being infinite means that our bottom 30 bits equate to 0x7f800000 */
    if ((x[0] & 0x7fffffff) == 0x7f800000) out[0] = 1;
    else out[0] = 0;
    return NULL;
}

char *test_isinf(uint32 *x, uint32 *out)
{
    int xhigh = x[0];
    int xlow = x[1];
    /* Being infinite means that our fraction is zero and exponent is 0x7ff */
    if (((xhigh & 0x7fffffff) == 0x7ff00000) && (xlow == 0)) out[0] = 1;
    else out[0] = 0;
    return NULL;
}

char *test_isnanf(uint32 *x, uint32 *out)
{
    /* Being NaN means that our exponent is 0xff and non-0 fraction */
    int exponent = x[0] & 0x7f800000;
    int fraction = x[0] & 0x007fffff;
    if ((exponent == 0x7f800000) && (fraction != 0)) out[0] = 1;
    else out[0] = 0;
    return NULL;
}

char *test_isnan(uint32 *x, uint32 *out)
{
    /* Being NaN means that our exponent is 0x7ff and non-0 fraction */
    int exponent = x[0] & 0x7ff00000;
    int fractionhigh = x[0] & 0x000fffff;
    if ((exponent == 0x7ff00000) && ((fractionhigh != 0) || x[1] != 0))
        out[0] = 1;
    else out[0] = 0;
    return NULL;
}

char *test_isnormalf(uint32 *x, uint32 *out)
{
    /* Being normal means exponent is not 0 and is not 0xff */
    int exponent = x[0] & 0x7f800000;
    if (exponent == 0x7f800000) out[0] = 0;
    else if (exponent == 0) out[0] = 0;
    else out[0] = 1;
    return NULL;
}

char *test_isnormal(uint32 *x, uint32 *out)
{
    /* Being normal means exponent is not 0 and is not 0x7ff */
    int exponent = x[0] & 0x7ff00000;
    if (exponent == 0x7ff00000) out[0] = 0;
    else if (exponent == 0) out[0] = 0;
    else out[0] = 1;
    return NULL;
}

char *test_signbitf(uint32 *x, uint32 *out)
{
    /* Sign bit is bit 31 */
    out[0] = (x[0] >> 31) & 1;
    return NULL;
}

char *test_signbit(uint32 *x, uint32 *out)
{
    /* Sign bit is bit 31 */
    out[0] = (x[0] >> 31) & 1;
    return NULL;
}

char *test_fpclassify(uint32 *x, uint32 *out)
{
    int exponent = (x[0] & 0x7ff00000) >> 20;
    int fraction = (x[0] & 0x000fffff) | x[1];

    if ((exponent == 0x00) && (fraction == 0)) out[0] = 0;
    else if ((exponent == 0x00) && (fraction != 0)) out[0] = 4;
    else if ((exponent == 0x7ff) && (fraction == 0)) out[0] = 3;
    else if ((exponent == 0x7ff) && (fraction != 0)) out[0] = 7;
    else out[0] = 5;
    return NULL;
}

char *test_fpclassifyf(uint32 *x, uint32 *out)
{
    int exponent = (x[0] & 0x7f800000) >> 23;
    int fraction = x[0] & 0x007fffff;

    if ((exponent == 0x000) && (fraction == 0)) out[0] = 0;
    else if ((exponent == 0x000) && (fraction != 0)) out[0] = 4;
    else if ((exponent == 0xff) && (fraction == 0)) out[0] = 3;
    else if ((exponent == 0xff) && (fraction != 0)) out[0] = 7;
    else out[0] = 5;
    return NULL;
}

/*
 * Internal function that compares doubles in x & y and returns -3, -2, -1, 0,
 * 1 if they compare to be signaling, unordered, less than, equal or greater
 * than.
 */
static int fpcmp4(uint32 *x, uint32 *y)
{
    int result = 0;

    /*
     * Sort out whether results are ordered or not to begin with
     * NaNs have exponent 0x7ff, and non-zero fraction. Signaling NaNs take
     * higher priority than quiet ones.
     */
    if ((x[0] & 0x7fffffff) >= 0x7ff80000) result = -2;
    else if ((x[0] & 0x7fffffff) > 0x7ff00000) result = -3;
    else if (((x[0] & 0x7fffffff) == 0x7ff00000) && (x[1] != 0)) result = -3;
    if ((y[0] & 0x7fffffff) >= 0x7ff80000 && result != -3) result = -2;
    else if ((y[0] & 0x7fffffff) > 0x7ff00000) result = -3;
    else if (((y[0] & 0x7fffffff) == 0x7ff00000) && (y[1] != 0)) result = -3;
    if (result != 0) return result;

    /*
     * The two forms of zero are equal
     */
    if (((x[0] & 0x7fffffff) == 0) && x[1] == 0 &&
        ((y[0] & 0x7fffffff) == 0) && y[1] == 0)
        return 0;

    /*
     * If x and y have different signs we can tell that they're not equal
     * If x is +ve we have x > y return 1 - otherwise y is +ve return -1
     */
    if ((x[0] >> 31) != (y[0] >> 31))
        return ((x[0] >> 31) == 0) - ((y[0] >> 31) == 0);

    /*
     * Now we have both signs the same, let's do an initial compare of the
     * values.
     *
     * Whoever designed IEEE754's floating point formats is very clever and
     * earns my undying admiration.  Once you remove the sign-bit, the
     * floating point numbers can be ordered using the standard <, ==, >
     * operators will treating the fp-numbers as integers with that bit-
     * pattern.
     */
    if ((x[0] & 0x7fffffff) < (y[0] & 0x7fffffff)) result = -1;
    else if ((x[0] & 0x7fffffff) > (y[0] & 0x7fffffff)) result = 1;
    else if (x[1] < y[1]) result = -1;
    else if (x[1] > y[1]) result = 1;
    else result = 0;

    /*
     * Now we return the result - is x is positive (and therefore so is y) we
     * return the plain result - otherwise we negate it and return.
     */
    if ((x[0] >> 31) == 0) return result;
    else return -result;
}

/*
 * Internal function that compares floats in x & y and returns -3, -2, -1, 0,
 * 1 if they compare to be signaling, unordered, less than, equal or greater
 * than.
 */
static int fpcmp4f(uint32 *x, uint32 *y)
{
    int result = 0;

    /*
     * Sort out whether results are ordered or not to begin with
     * NaNs have exponent 0xff, and non-zero fraction - we have to handle all
     * signaling cases over the quiet ones
     */
    if ((x[0] & 0x7fffffff) >= 0x7fc00000) result = -2;
    else if ((x[0] & 0x7fffffff) > 0x7f800000) result = -3;
    if ((y[0] & 0x7fffffff) >= 0x7fc00000 && result != -3) result = -2;
    else if ((y[0] & 0x7fffffff) > 0x7f800000) result = -3;
    if (result != 0) return result;

    /*
     * The two forms of zero are equal
     */
    if (((x[0] & 0x7fffffff) == 0) && ((y[0] & 0x7fffffff) == 0))
        return 0;

    /*
     * If x and y have different signs we can tell that they're not equal
     * If x is +ve we have x > y return 1 - otherwise y is +ve return -1
     */
    if ((x[0] >> 31) != (y[0] >> 31))
        return ((x[0] >> 31) == 0) - ((y[0] >> 31) == 0);

    /*
     * Now we have both signs the same, let's do an initial compare of the
     * values.
     *
     * Whoever designed IEEE754's floating point formats is very clever and
     * earns my undying admiration.  Once you remove the sign-bit, the
     * floating point numbers can be ordered using the standard <, ==, >
     * operators will treating the fp-numbers as integers with that bit-
     * pattern.
     */
    if ((x[0] & 0x7fffffff) < (y[0] & 0x7fffffff)) result = -1;
    else if ((x[0] & 0x7fffffff) > (y[0] & 0x7fffffff)) result = 1;
    else result = 0;

    /*
     * Now we return the result - is x is positive (and therefore so is y) we
     * return the plain result - otherwise we negate it and return.
     */
    if ((x[0] >> 31) == 0) return result;
    else return -result;
}

char *test_isgreater(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4(x, y);
    *out = (result == 1);
    return result == -3 ? "i" : NULL;
}

char *test_isgreaterequal(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4(x, y);
    *out = (result >= 0);
    return result == -3 ? "i" : NULL;
}

char *test_isless(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4(x, y);
    *out = (result == -1);
    return result == -3 ? "i" : NULL;
}

char *test_islessequal(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4(x, y);
    *out = (result == -1) || (result == 0);
    return result == -3 ? "i" : NULL;
}

char *test_islessgreater(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4(x, y);
    *out = (result == -1) || (result == 1);
    return result == -3 ? "i" : NULL;
}

char *test_isunordered(uint32 *x, uint32 *y, uint32 *out)
{
    int normal = 0;
    int result = fpcmp4(x, y);

    test_isnormal(x, out);
    normal |= *out;
    test_isnormal(y, out);
    normal |= *out;
    *out = (result == -2) || (result == -3);
    return result == -3 ? "i" : NULL;
}

char *test_isgreaterf(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4f(x, y);
    *out = (result == 1);
    return result == -3 ? "i" : NULL;
}

char *test_isgreaterequalf(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4f(x, y);
    *out = (result >= 0);
    return result == -3 ? "i" : NULL;
}

char *test_islessf(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4f(x, y);
    *out = (result == -1);
    return result == -3 ? "i" : NULL;
}

char *test_islessequalf(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4f(x, y);
    *out = (result == -1) || (result == 0);
    return result == -3 ? "i" : NULL;
}

char *test_islessgreaterf(uint32 *x, uint32 *y, uint32 *out)
{
    int result = fpcmp4f(x, y);
    *out = (result == -1) || (result == 1);
    return result == -3 ? "i" : NULL;
}

char *test_isunorderedf(uint32 *x, uint32 *y, uint32 *out)
{
    int normal = 0;
    int result = fpcmp4f(x, y);

    test_isnormalf(x, out);
    normal |= *out;
    test_isnormalf(y, out);
    normal |= *out;
    *out = (result == -2) || (result == -3);
    return result == -3 ? "i" : NULL;
}
