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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_INLINE double2
__libclc__sincos_piby4(double x, double xx)
{
    // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
    //                      = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
    //                      = x * f(w)
    // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
    // We use a minimax approximation of (f(w) - 1) / w
    // because this produces an expansion in even powers of x.
    // If xx (the tail of x) is non-zero, we add a correction
    // term g(x,xx) = (1-x*x/2)*xx to the result, where g(x,xx)
    // is an approximation to cos(x)*sin(xx) valid because
    // xx is tiny relative to x.

    // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
    //                      = f(w)
    // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
    // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
    // because this produces an expansion in even powers of x.
    // If xx (the tail of x) is non-zero, we subtract a correction
    // term g(x,xx) = x*xx to the result, where g(x,xx)
    // is an approximation to sin(x)*sin(xx) valid because
    // xx is tiny relative to x.

    const double sc1 = -0.166666666666666646259241729;
    const double sc2 =  0.833333333333095043065222816e-2;
    const double sc3 = -0.19841269836761125688538679e-3;
    const double sc4 =  0.275573161037288022676895908448e-5;
    const double sc5 = -0.25051132068021699772257377197e-7;
    const double sc6 =  0.159181443044859136852668200e-9;

    const double cc1 =  0.41666666666666665390037e-1;
    const double cc2 = -0.13888888888887398280412e-2;
    const double cc3 =  0.248015872987670414957399e-4;
    const double cc4 = -0.275573172723441909470836e-6;
    const double cc5 =  0.208761463822329611076335e-8;
    const double cc6 = -0.113826398067944859590880e-10;

    double x2 = x * x;
    double x3 = x2 * x;
    double r = 0.5 * x2;
    double t = 1.0 - r;

    double sp = fma(fma(fma(fma(sc6, x2, sc5), x2, sc4), x2, sc3), x2, sc2);

    double cp = t + fma(fma(fma(fma(fma(fma(cc6, x2, cc5), x2, cc4), x2, cc3), x2, cc2), x2, cc1),
                        x2*x2, fma(x, xx, (1.0 - t) - r));

    double2 ret;
    ret.lo = x - fma(-x3, sc1, fma(fma(-x3, sp, 0.5*xx), x2, -xx));
    ret.hi = cp;

    return ret;
}

_CLC_INLINE double2
__clc_tan_piby4(double x, double xx)
{
    const double piby4_lead = 7.85398163397448278999e-01; // 0x3fe921fb54442d18
    const double piby4_tail = 3.06161699786838240164e-17; // 0x3c81a62633145c06

    // In order to maintain relative precision transform using the identity:
    // tan(pi/4-x) = (1-tan(x))/(1+tan(x)) for arguments close to pi/4.
    // Similarly use tan(x-pi/4) = (tan(x)-1)/(tan(x)+1) close to -pi/4.

    int ca = x >  0.68;
    int cb = x < -0.68;
    double transform = ca ?  1.0 : 0.0;
    transform = cb ? -1.0 : transform;

    double tx = fma(-transform, x, piby4_lead) + fma(-transform, xx, piby4_tail);
    int c = ca | cb;
    x = c ? tx : x;
    xx = c ? 0.0 : xx;

    // Core Remez [2,3] approximation to tan(x+xx) on the interval [0,0.68].
    double t1 = x;
    double r = fma(2.0, x*xx, x*x);

    double a = fma(r,
                   fma(r, 0.224044448537022097264602535574e-3, -0.229345080057565662883358588111e-1),
                   0.372379159759792203640806338901e0);

    double b = fma(r,
                   fma(r,
                       fma(r, -0.232371494088563558304549252913e-3, 0.260656620398645407524064091208e-1),
                       -0.515658515729031149329237816945e0),
                   0.111713747927937668539901657944e1);

    double t2 = fma(MATH_DIVIDE(a, b), x*r, xx);

    double tp = t1 + t2;

    // Compute -1.0/(t1 + t2) accurately
    double z1 = as_double(as_long(tp) & 0xffffffff00000000L);
    double z2 = t2 - (z1 - t1);
    double trec = -MATH_RECIP(tp);
    double trec_top = as_double(as_long(trec) & 0xffffffff00000000L);

    double tpr = fma(fma(trec_top, z2, fma(trec_top, z1, 1.0)), trec, trec_top);

    double tpt = transform * (1.0 - MATH_DIVIDE(2.0*tp, 1.0 + tp));
    double tptr = transform * (MATH_DIVIDE(2.0*tp, tp - 1.0) - 1.0);

    double2 ret;
    ret.lo = c ? tpt : tp;
    ret.hi = c ? tptr : tpr;
    return ret;
}
