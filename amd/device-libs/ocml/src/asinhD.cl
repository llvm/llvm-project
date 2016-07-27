/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"


PUREATTR double
MATH_MANGLE(asinh)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax <= 1.0) {
        const double NA0 = -0.12845379283524906084997e0;
        const double NA1 = -0.21060688498409799700819e0;
        const double NA2 = -0.10188951822578188309186e0;
        const double NA3 = -0.13891765817243625541799e-1;
        const double NA4 = -0.10324604871728082428024e-3;

        const double DA0 =  0.77072275701149440164511e0;
        const double DA1 =  0.16104665505597338100747e1;
        const double DA2 =  0.11296034614816689554875e1;
        const double DA3 =  0.30079351943799465092429e0;
        const double DA4 =  0.235224464765951442265117e-1;

        const double NB0 = -0.12186605129448852495563e0;
        const double NB1 = -0.19777978436593069928318e0;
        const double NB2 = -0.94379072395062374824320e-1;
        const double NB3 = -0.12620141363821680162036e-1;
        const double NB4 = -0.903396794842691998748349e-4;

        const double DB0 =  0.73119630776696495279434e0;
        const double DB1 =  0.15157170446881616648338e1;
        const double DB2 =  0.10524909506981282725413e1;
        const double DB3 =  0.27663713103600182193817e0;
        const double DB4 =  0.21263492900663656707646e-1;

        const double NC0 = -0.81210026327726247622500e-1;
        const double NC1 = -0.12327355080668808750232e0;
        const double NC2 = -0.53704925162784720405664e-1;
        const double NC3 = -0.63106739048128554465450e-2;
        const double NC4 = -0.35326896180771371053534e-4;

        const double DC0 =  0.48726015805581794231182e0;
        const double DC1 =  0.95890837357081041150936e0;
        const double DC2 =  0.62322223426940387752480e0;
        const double DC3 =  0.15028684818508081155141e0;
        const double DC4 =  0.10302171620320141529445e-1;

        const double ND0 = -0.4638179204422665073e-1;
        const double ND1 = -0.7162729496035415183e-1;
        const double ND2 = -0.3247795155696775148e-1;
        const double ND3 = -0.4225785421291932164e-2;
        const double ND4 = -0.3808984717603160127e-4;
        const double ND5 =  0.8023464184964125826e-6;

        const double DD0 =  0.2782907534642231184e0;
        const double DD1 =  0.5549945896829343308e0;
        const double DD2 =  0.3700732511330698879e0;
        const double DD3 =  0.9395783438240780722e-1;
        const double DD4 =  0.7200057974217143034e-2;

        double pn, pd;
        double x2 = x * x;

        if (ax < 0.25) {
            pn = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, NA4, NA3), NA2), NA1), NA0);
            pd = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, DA4, DA3), DA2), DA1), DA0);
        } else if (ax < 0.5) {
            pn = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, NB4, NB3), NB2), NB1), NB0);
            pd = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, DB4, DB3), DB2), DB1), DB0);
        } else if (ax < 0.75) {
            pn = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, NC4, NC3), NC2), NC1), NC0);
            pd = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, DC4, DC3), DC2), DC1), DC0);
        } else {
            pn = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, ND5, ND4), ND3), ND2), ND1), ND0);
            pd = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, DD4, DD3), DD2), DD1), DD0);
        }
        double pq = MATH_DIV(pn, pd);
        ret = MATH_MAD(ax, x2*pq, ax);
    } else if (ax < 2.0) {
        double x2 = x*x;
        // Use sqrt(x^2+1) = 1 + x^2/(1 + sqrt(x^2+1)) ... this works to x=0
        ret = MATH_MANGLE(log1p)(ax + MATH_FAST_DIV(x2, 1.0 + MATH_FAST_SQRT(MATH_MAD(ax, ax, 1.0))));
    } else if (ax < 0x1.6a09e667f3bcdp+26) {
        // We could instead use x+sqrt(x^2+1) = 2x + 1/(x+sqrt(x^2+1))
        ret = MATH_MANGLE(log)(ax + MATH_FAST_SQRT(MATH_MAD(ax, ax, 1.0)));
    } else {
        const double ln2 = 0x1.62e42fefa39efp-1;
        ret = MATH_MANGLE(log)(ax) + ln2;
    }

    return BUILTIN_COPYSIGN_F64(ret, x);
}

