/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
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

#include "math.h"
#include "ep_log.h"
#include "../clcmacro.h"

_CLC_OVERLOAD _CLC_DEF float asinh(float x) {
    uint ux = as_uint(x);
    uint ax = ux & EXSIGNBIT_SP32;
    uint xsgn = ax ^ ux;

    // |x| <= 2
    float t = x * x;
    float a = mad(t,
                  mad(t,
		      mad(t,
		          mad(t, -1.177198915954942694e-4f, -4.162727710583425360e-2f),
		          -5.063201055468483248e-1f),
		      -1.480204186473758321f),
	          -1.152965835871758072f);
    float b = mad(t,
	          mad(t,
		      mad(t,
			  mad(t, 6.284381367285534560e-2f, 1.260024978680227945f),
			  6.582362487198468066f),
		      11.99423176003939087f),
		  6.917795026025976739f);

    float q = MATH_DIVIDE(a, b);
    float z1 = mad(x*t, q, x);

    // |x| > 2

    // Arguments greater than 1/sqrt(epsilon) in magnitude are
    // approximated by asinh(x) = ln(2) + ln(abs(x)), with sign of x
    // Arguments such that 4.0 <= abs(x) <= 1/sqrt(epsilon) are
    // approximated by asinhf(x) = ln(abs(x) + sqrt(x*x+1))
    // with the sign of x (see Abramowitz and Stegun 4.6.20)

    float absx = as_float(ax);
    int hi = ax > 0x46000000U;
    float y = MATH_SQRT(absx * absx + 1.0f) + absx;
    y = hi ? absx : y;
    float r = log(y) + (hi ? 0x1.62e430p-1f : 0.0f);
    float z2 = as_float(xsgn | as_uint(r));

    float z = ax <= 0x40000000 ? z1 : z2;
    z = ax < 0x39800000U | ax >= PINFBITPATT_SP32 ? x : z;

    return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, asinh, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NA0 -0.12845379283524906084997e0
#define NA1 -0.21060688498409799700819e0
#define NA2 -0.10188951822578188309186e0
#define NA3 -0.13891765817243625541799e-1
#define NA4 -0.10324604871728082428024e-3

#define DA0  0.77072275701149440164511e0
#define DA1  0.16104665505597338100747e1
#define DA2  0.11296034614816689554875e1
#define DA3  0.30079351943799465092429e0
#define DA4  0.235224464765951442265117e-1

#define NB0 -0.12186605129448852495563e0
#define NB1 -0.19777978436593069928318e0
#define NB2 -0.94379072395062374824320e-1
#define NB3 -0.12620141363821680162036e-1
#define NB4 -0.903396794842691998748349e-4

#define DB0  0.73119630776696495279434e0
#define DB1  0.15157170446881616648338e1
#define DB2  0.10524909506981282725413e1
#define DB3  0.27663713103600182193817e0
#define DB4  0.21263492900663656707646e-1

#define NC0 -0.81210026327726247622500e-1
#define NC1 -0.12327355080668808750232e0
#define NC2 -0.53704925162784720405664e-1
#define NC3 -0.63106739048128554465450e-2
#define NC4 -0.35326896180771371053534e-4

#define DC0  0.48726015805581794231182e0
#define DC1  0.95890837357081041150936e0
#define DC2  0.62322223426940387752480e0
#define DC3  0.15028684818508081155141e0
#define DC4  0.10302171620320141529445e-1

#define ND0 -0.4638179204422665073e-1
#define ND1 -0.7162729496035415183e-1
#define ND2 -0.3247795155696775148e-1
#define ND3 -0.4225785421291932164e-2
#define ND4 -0.3808984717603160127e-4
#define ND5  0.8023464184964125826e-6

#define DD0  0.2782907534642231184e0
#define DD1  0.5549945896829343308e0
#define DD2  0.3700732511330698879e0
#define DD3  0.9395783438240780722e-1
#define DD4  0.7200057974217143034e-2

#define NE0 -0.121224194072430701e-4
#define NE1 -0.273145455834305218e-3
#define NE2 -0.152866982560895737e-2
#define NE3 -0.292231744584913045e-2
#define NE4 -0.174670900236060220e-2
#define NE5 -0.891754209521081538e-12

#define DE0  0.499426632161317606e-4
#define DE1  0.139591210395547054e-2
#define DE2  0.107665231109108629e-1
#define DE3  0.325809818749873406e-1
#define DE4  0.415222526655158363e-1
#define DE5  0.186315628774716763e-1

#define NF0  -0.195436610112717345e-4
#define NF1  -0.233315515113382977e-3
#define NF2  -0.645380957611087587e-3
#define NF3  -0.478948863920281252e-3
#define NF4  -0.805234112224091742e-12
#define NF5   0.246428598194879283e-13

#define DF0   0.822166621698664729e-4
#define DF1   0.135346265620413852e-2
#define DF2   0.602739242861830658e-2
#define DF3   0.972227795510722956e-2
#define DF4   0.510878800983771167e-2

#define NG0  -0.209689451648100728e-6
#define NG1  -0.219252358028695992e-5
#define NG2  -0.551641756327550939e-5
#define NG3  -0.382300259826830258e-5
#define NG4  -0.421182121910667329e-17
#define NG5   0.492236019998237684e-19

#define DG0   0.889178444424237735e-6
#define DG1   0.131152171690011152e-4
#define DG2   0.537955850185616847e-4
#define DG3   0.814966175170941864e-4
#define DG4   0.407786943832260752e-4

#define NH0  -0.178284193496441400e-6
#define NH1  -0.928734186616614974e-6
#define NH2  -0.923318925566302615e-6
#define NH3  -0.776417026702577552e-19
#define NH4   0.290845644810826014e-21

#define DH0   0.786694697277890964e-6
#define DH1   0.685435665630965488e-5
#define DH2   0.153780175436788329e-4
#define DH3   0.984873520613417917e-5

#define NI0  -0.538003743384069117e-10
#define NI1  -0.273698654196756169e-9
#define NI2  -0.268129826956403568e-9
#define NI3  -0.804163374628432850e-29

#define DI0   0.238083376363471960e-9
#define DI1   0.203579344621125934e-8
#define DI2   0.450836980450693209e-8
#define DI3   0.286005148753497156e-8

_CLC_OVERLOAD _CLC_DEF double asinh(double x) {
    const double rteps = 0x1.6a09e667f3bcdp-27;
    const double recrteps = 0x1.6a09e667f3bcdp+26;

    // log2_lead and log2_tail sum to an extra-precise version of log(2)
    const double log2_lead = 0x1.62e42ep-1;
    const double log2_tail = 0x1.efa39ef35793cp-25;

    ulong ux = as_ulong(x);
    ulong ax = ux & ~SIGNBIT_DP64;
    double absx = as_double(ax);

    double t = x * x;
    double pn, tn, pd, td;

    // XXX we are betting here that we can evaluate 8 pairs of
    // polys faster than we can grab 12 coefficients from a table
    // This also uses fewer registers

    // |x| >= 8
    pn = fma(t, fma(t, fma(t, NI3, NI2), NI1), NI0);
    pd = fma(t, fma(t, fma(t, DI3, DI2), DI1), DI0);

    tn = fma(t, fma(t, fma(t, fma(t, NH4, NH3), NH2), NH1), NH0);
    td = fma(t, fma(t, fma(t, DH3, DH2), DH1), DH0);
    pn = absx < 8.0 ? tn : pn;
    pd = absx < 8.0 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, fma(t, NG5, NG4), NG3), NG2), NG1), NG0);
    td = fma(t, fma(t, fma(t, fma(t, DG4, DG3), DG2), DG1), DG0);
    pn = absx < 4.0 ? tn : pn;
    pd = absx < 4.0 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, fma(t, NF5, NF4), NF3), NF2), NF1), NF0);
    td = fma(t, fma(t, fma(t, fma(t, DF4, DF3), DF2), DF1), DF0);
    pn = absx < 2.0 ? tn : pn;
    pd = absx < 2.0 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, fma(t, NE5, NE4), NE3), NE2), NE1), NE0);
    td = fma(t, fma(t, fma(t, fma(t, fma(t, DE5, DE4), DE3), DE2), DE1), DE0);
    pn = absx < 1.5 ? tn : pn;
    pd = absx < 1.5 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, fma(t, ND5, ND4), ND3), ND2), ND1), ND0);
    td = fma(t, fma(t, fma(t, fma(t, DD4, DD3), DD2), DD1), DD0);
    pn = absx <= 1.0 ? tn : pn;
    pd = absx <= 1.0 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, NC4, NC3), NC2), NC1), NC0);
    td = fma(t, fma(t, fma(t, fma(t, DC4, DC3), DC2), DC1), DC0);
    pn = absx < 0.75 ? tn : pn;
    pd = absx < 0.75 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, NB4, NB3), NB2), NB1), NB0);
    td = fma(t, fma(t, fma(t, fma(t, DB4, DB3), DB2), DB1), DB0);
    pn = absx < 0.5 ? tn : pn;
    pd = absx < 0.5 ? td : pd;

    tn = fma(t, fma(t, fma(t, fma(t, NA4, NA3), NA2), NA1), NA0);
    td = fma(t, fma(t, fma(t, fma(t, DA4, DA3), DA2), DA1), DA0);
    pn = absx < 0.25 ? tn : pn;
    pd = absx < 0.25 ? td : pd;

    double pq = MATH_DIVIDE(pn, pd);

    // |x| <= 1
    double result1 = fma(absx*t, pq, absx);

    // Other ranges
    int xout = absx <= 32.0 | absx > recrteps;
    double y = absx + sqrt(fma(absx, absx, 1.0));
    y = xout ? absx : y;

    double r1, r2;
    int xexp;
    __clc_ep_log(y, &xexp, &r1, &r2);

    double dxexp = (double)(xexp + xout);
    r1 = fma(dxexp, log2_lead, r1);
    r2 = fma(dxexp, log2_tail, r2);

    // 1 < x <= 32
    double v2 = (pq + 0.25) / t;
    double r = v2 + r1;
    double s = ((r1 - r) + v2) + r2;
    double v1 = r + s;
    v2 = (r - v1) + s;
    double result2 = v1 + v2;

    // x > 32
    double result3 = r1 + r2;

    double ret = absx > 1.0 ? result2 : result1;
    ret = absx > 32.0 ? result3 : ret;
    ret = x < 0.0 ? -ret : ret;

    // NaN, +-Inf, or x small enough that asinh(x) = x
    ret = ax >= PINFBITPATT_DP64 | absx < rteps ? x : ret;
    return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, asinh, double)

#endif
