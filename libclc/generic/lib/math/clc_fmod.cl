/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/integer/clc_clz.h>
#include <clc/math/clc_floor.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/shared/clc_max.h>
#include <math/clc_remainder.h>

_CLC_DEF _CLC_OVERLOAD float __clc_fmod(float x, float y)
{
    int ux = as_int(x);
    int ax = ux & EXSIGNBIT_SP32;
    float xa = as_float(ax);
    int sx = ux ^ ax;
    int ex = ax >> EXPSHIFTBITS_SP32;

    int uy = as_int(y);
    int ay = uy & EXSIGNBIT_SP32;
    float ya = as_float(ay);
    int ey = ay >> EXPSHIFTBITS_SP32;

    float xr = as_float(0x3f800000 | (ax & 0x007fffff));
    float yr = as_float(0x3f800000 | (ay & 0x007fffff));
    int c;
    int k = ex - ey;

    while (k > 0) {
        c = xr >= yr;
        xr -= c ? yr : 0.0f;
        xr += xr;
        --k;
    }

    c = xr >= yr;
    xr -= c ? yr : 0.0f;

    int lt = ex < ey;

    xr = lt ? xa : xr;
    yr = lt ? ya : yr;


    float s = as_float(ey << EXPSHIFTBITS_SP32);
    xr *= lt ? 1.0f : s;

    c = ax == ay;
    xr = c ? 0.0f : xr;

    xr = as_float(sx ^ as_int(xr));

    c = ax > PINFBITPATT_SP32 | ay > PINFBITPATT_SP32 | ax == PINFBITPATT_SP32 | ay == 0;
    xr = c ? as_float(QNANBITPATT_SP32) : xr;

    return xr;

}
_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_fmod, float, float);

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_fmod(double x, double y)
{
    ulong ux = as_ulong(x);
    ulong ax = ux & ~SIGNBIT_DP64;
    ulong xsgn = ux ^ ax;
    double dx = as_double(ax);
    int xexp = convert_int(ax >> EXPSHIFTBITS_DP64);
    int xexp1 = 11 - (int) __clc_clz(ax & MANTBITS_DP64);
    xexp1 = xexp < 1 ? xexp1 : xexp;

    ulong uy = as_ulong(y);
    ulong ay = uy & ~SIGNBIT_DP64;
    double dy = as_double(ay);
    int yexp = convert_int(ay >> EXPSHIFTBITS_DP64);
    int yexp1 = 11 - (int) __clc_clz(ay & MANTBITS_DP64);
    yexp1 = yexp < 1 ? yexp1 : yexp;

    // First assume |x| > |y|

    // Set ntimes to the number of times we need to do a
    // partial remainder. If the exponent of x is an exact multiple
    // of 53 larger than the exponent of y, and the mantissa of x is
    // less than the mantissa of y, ntimes will be one too large
    // but it doesn't matter - it just means that we'll go round
    // the loop below one extra time.
    int ntimes = __clc_max(0, (xexp1 - yexp1) / 53);
    double w =  ldexp(dy, ntimes * 53);
    w = ntimes == 0 ? dy : w;
    double scale = ntimes == 0 ? 1.0 : 0x1.0p-53;

    // Each time round the loop we compute a partial remainder.
    // This is done by subtracting a large multiple of w
    // from x each time, where w is a scaled up version of y.
    // The subtraction must be performed exactly in quad
    // precision, though the result at each stage can
    // fit exactly in a double precision number.
    int i;
    double t, v, p, pp;

    for (i = 0; i < ntimes; i++) {
        // Compute integral multiplier
        t = __clc_trunc(dx / w);

        // Compute w * t in quad precision
        p = w * t;
        pp = fma(w, t, -p);

        // Subtract w * t from dx
        v = dx - p;
        dx = v + (((dx - v) - p) - pp);

        // If t was one too large, dx will be negative. Add back one w.
        dx += dx < 0.0 ? w : 0.0;

        // Scale w down by 2^(-53) for the next iteration
        w *= scale;
    }

    // One more time
    // Variable todd says whether the integer t is odd or not
    t = __clc_floor(dx / w);
    long lt = (long)t;
    int todd = lt & 1;

    p = w * t;
    pp = fma(w, t, -p);
    v = dx - p;
    dx = v + (((dx - v) - p) - pp);
    i = dx < 0.0;
    todd ^= i;
    dx += i ? w : 0.0;

    // At this point, dx lies in the range [0,dy)
    double ret = as_double(xsgn ^ as_ulong(dx));
    dx = as_double(ax);

    // Now handle |x| == |y|
    int c = dx == dy;
    t = as_double(xsgn);
    ret = c ? t : ret;

    // Next, handle |x| < |y|
    c = dx < dy;
    ret = c ? x : ret;

    // We don't need anything special for |x| == 0

    // |y| is 0
    c = dy == 0.0;
    ret = c ? as_double(QNANBITPATT_DP64) : ret;

    // y is +-Inf, NaN
    c = yexp > BIASEDEMAX_DP64;
    t = y == y ? x : y;
    ret = c ? t : ret;

    // x is +=Inf, NaN
    c = xexp > BIASEDEMAX_DP64;
    ret = c ? as_double(QNANBITPATT_DP64) : ret;

    return ret;
}
_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_fmod, double, double);
#endif
