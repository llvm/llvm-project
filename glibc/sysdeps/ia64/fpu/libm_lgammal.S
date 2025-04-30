.file "libm_lgammal.s"


// Copyright (c) 2002 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2002 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,INCLUDING,BUT NOT
// LIMITED TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT,INDIRECT,INCIDENTAL,SPECIAL,
// EXEMPLARY,OR CONSEQUENTIAL DAMAGES (INCLUDING,BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,DATA,OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY,WHETHER IN CONTRACT,STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE,EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code,and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//
//*********************************************************************
//
// History:
// 03/28/02  Original version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 08/21/02  Added support of SIGN(GAMMA(x)) calculation
// 09/26/02  Algorithm description improved
// 10/21/02  Now it returns SIGN(GAMMA(x))=-1 for negative zero
// 02/10/03  Reordered header: .section, .global, .proc, .align
// 03/31/05  Reformatted delimiters between data tables
//
//*********************************************************************
//
// Function: __libm_lgammal(long double x, int* signgam, int szsigngam)
// computes the principal value of the logarithm of the GAMMA function
// of x. Signum of GAMMA(x) is stored to memory starting at the address
// specified by the signgam.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 (Input and Return Value)
//                              f9-f15
//                              f32-f127
//
//    General Purpose Registers:
//      r2, r3, r8-r11, r14-r31
//      r32-r65
//      r66-r69 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    __libm_lgammal(+inf) = +inf
//    __libm_lgammal(-inf) = QNaN
//    __libm_lgammal(+/-0) = +inf
//    __libm_lgammal(x<0, x - integer) = QNaN
//    __libm_lgammal(SNaN) = QNaN
//    __libm_lgammal(QNaN) = QNaN
//
//*********************************************************************
//
// ALGORITHM DESCRIPTION
//
// Below we suppose that there is log(z) function which takes an long
// double argument and returns result as a pair of long double numbers
// lnHi and lnLo (such that sum lnHi + lnLo provides ~80 correct bits
// of significand). Algorithm description for such log(z) function
// see below.
// Also, it this algorithm description we use the following notational
// conventions:
// a) pair A = (Ahi, Alo) means number A represented as sum of Ahi and Alo
// b) C = A + B = (Ahi, Alo) + (Bhi, Blo) means multi-precision addition.
//    The result would be C = (Chi, Clo). Notice, that Clo shouldn't be
//    equal to Alo + Blo
// c) D = A*B = (Ahi, Alo)*(Bhi, Blo) = (Dhi, Dlo) multi-precisiion
//    multiplication.
//
// So, lgammal has the following computational paths:
// 1) |x| < 0.5
//    P = A1*|x| + A2*|x|^2 + ... + A22*|x|^22
//    A1, A2, A3 represented as a sum of two double precision
//    numbers and multi-precision computations are used for 3 higher
//    terms of the polynomial. We get polynomial as a sum of two
//    double extended numbers: P = (Phi, Plo)
//    1.1) x > 0
//         lgammal(x) = P - log(|x|) = (Phi, Plo) - (lnHi(|x|), lnLo(|x|))
//    1.2) x < 0
//         lgammal(x) = -P - log(|x|) - log(sin(Pi*x)/(Pi*x))
//         P and log(|x|) are computed by the same way as in 1.1;
//         - log(sin(Pi*x)/(Pi*x)) is approximated by a polynomial Plnsin.
//         Plnsin:= fLnSin2*|x|^2 + fLnSin4*|x|^4 + ... + fLnSin36*|x|^36
//         The first coefficient of Plnsin is represented as sum of two
//         double precision numbers (fLnSin2, fLnSin2L). Multi-precision
//         computations for higher two terms of Plnsin are used.
//         So, the final result is reconstructed by the following formula
//         lgammal(x) = (-(Phi, Plo) - (lnHi(|x|), lnLo(|x|))) -
//                      - (PlnsinHi,PlnsinLo)
//
// 2)    0.5 <= x <   0.75  -> t = x - 0.625
//     -0.75 <  x <= -0.5   -> t = x + 0.625
//      2.25 <= x <   4.0   -> t = x/2 - 1.5
//       4.0 <= x <   8.0   -> t = x/4 - 1.5
//       -0.5 < x <= -0.40625 -> t = x + 0.5
//       -2.6005859375 < x <= -2.5 -> t = x + 2.5
//       1.3125 <= x < 1.5625 -> t = x - LOC_MIN, where LOC_MIN is point in
//                                   which lgammal has local minimum. Exact
//                                   value can be found in the table below,
//                                   approximate value is ~1.46
//
//    lgammal(x) is approximated by the polynomial of 25th degree: P25(t)
//    P25(t) = A0 + A1*t + ... + A25*t^25 = (Phi, Plo) + t^4*P21(t),
//    where
//    (Phi, Plo) is sum of four highest terms of the polynomial P25(t):
//    (Phi, Plo) = ((A0, A0L) + (A1, A1L)*t) + t^2 *((A2, A2L) + (A3, A3L)*t),
//    (Ai, AiL) - coefficients represented as pairs of DP numbers.
//
//    P21(t) = (PolC(t)*t^8 + PolD(t))*t^8 + PolE(t),
//    where
//    PolC(t) = C21*t^5 + C20*t^4 + ... + C16,
//    C21 = A25, C20 = A24, ..., C16 = A20
//
//    PolD(t) = D7*t^7 + D6*t^6 + ... + D0,
//    D7 = A19, D6 = A18, ..., D0 = A12
//
//    PolE(t) = E7*t^7 + E6*t^6 + ... + E0,
//    E7 = A11, E6 = A10, ..., E0 = A4
//
//    Cis and Dis are represented as double precision numbers,
//    Eis are represented as double extended numbers.
//
// 3) 0.75 <=  x < 1.3125   -> t = x - 1.0
//    1.5625 <= x < 2.25   -> t = x - 2.0
//    lgammal(x) is approximated by the polynomial of 25th degree: P25(t)
//    P25(t) = A1*t + ... + A25*t^25, and computations are carried out
//    by similar way as in the previous case
//
// 4) 10.0 < x <= Overflow Bound ("positive Sterling" range)
//    lgammal(x) is approximated using Sterling's formula:
//    lgammal(x) ~ ((x*(lnHi(x) - 1, lnLo(x))) - 0.5*(lnHi(x), lnLo(x))) +
//                 + ((Chi, Clo) + S(1/x))
//    where
//    C = (Chi, Clo) - pair of double precision numbers representing constant
//    0.5*ln(2*Pi);
//    S(1/x) = 1/x * (B2 + B4*(1/x)^2 + ... + B20*(1/x)^18), B2, ..., B20 are
//    Bernulli numbers. S is computed in native precision and then added to
//    Clo;
//    lnHi(x) - 1 is computed in native precision and the multiprecision
//    multiplication (x, 0) *(lnHi(x) - 1, lnLo(x)) is used.
//
// 5) -INF < x <= -2^63, any negative integer < 0
//    All numbers in this range are integers -> error handler is called
//
// 6) -2^63 < x <= -0.75 ("negative Sterling" range), x is "far" from root,
//    lgammal(-t) for positive t is approximated using the following formula:
//    lgammal(-t) = -lgammal(t)-log(t)-log(|dT|)+log(sin(Pi*|dT|)/(Pi*|dT|))
//        where dT = -t -round_to_nearest_integer(-t)
//    Last item is approximated by the same polynomial as described in 1.2.
//    We split the whole range into three subranges due to different ways of
//    approximation of the first terms.
//    6.1) -2^63 < x < -6.0 ("negative Sterling" range)
//       lgammal(t) is approximated exactly as in #4. The only difference that
//       for -13.0 < x < -6.0 subrange instead of Bernulli numbers we use their
//       minimax approximation on this range.
//       log(t), log(|dT|) are approximated by the log routine mentioned above.
//    6.2) -6.0 < x <= -0.75, |x + 1|> 2^(-7)
//       log(t), log(|dT|) are approximated by the log routine mentioned above,
//       lgammal(t) is approximated by polynomials of the 25th degree similar
//       to ones from #2. Arguments z of the polynomials are as follows
//       a) 0.75 <= t < 1.0 - 2^(-7),  z = 2*t - 1.5
//       b) 1.0 - 2^(-7)  < t < 2.0,   z = t - 1.5
//       c) 2.0  < t < 3.0,   z = t/2 - 1.5
//       d) 3.0  < t < 4.0,   z = t/2 - 1.5. Notice, that range reduction is
//          the same as in case c) but the set of coefficients is different
//       e) 4.0  < t < 6.0,   z = t/4 - 1.5
//    6.3) |x + 1| <= 2^(-7)
//       log(1 + (x-1)) is approximated by Taylor series,
//       log(sin(Pi*|dT|)/(Pi*|dT|)) is still approximated by polynomial but
//       it has just 4th degree.
//       log(|dT|) is approximated by the log routine mentioned above.
//       lgammal(-x) is approximated by polynomial of 8th degree from (-x + 1).
//
// 7) -20.0 < x < -2.0, x falls in root "neighbourhood".
//    "Neighbourhood" means that |lgammal(x)| < epsilon, where epsilon is
//    different for every root (and it is stored in the table), but typically
//    it is ~ 0.15. There are 35 roots significant from "double extended"
//    point of view. We split all the roots into two subsets: "left" and "right"
//    roots. Considering [-(N+1), -N] range we call root as "left" one if it
//    lies closer to -(N+1) and "right" otherwise. There is no "left" root in
//    the [-20, -19] range (it exists, but is insignificant for double extended
//    precision). To determine if x falls in root "neighbourhood" we store
//    significands of all the 35 roots as well as epsilon values (expressed
//    by the left and right bound).
//    In these ranges we approximate lgammal(x) by polynomial series of 19th
//    degree:
//    lgammal(x) = P19(t) = A0 + A1*t + ...+ A19*t^19, where t = x - EDP_Root,
//    EDP_Root is the exact value of the corresponding root rounded to double
//    extended precision. So, we have 35 different polynomials which make our
//    table rather big. We may hope that x falls in root "neighbourhood"
//    quite rarely -> ther might be no need in frequent use of different
//    polynomials.
//    A0, A1, A2, A3 are represented as pairs of double precision numbers,
//    A4, A5 are long doubles, and to decrease the size of the table we
//    keep the rest of coefficients in just double precision
//
//*********************************************************************
// Algorithm for log(X) = (lnHi(X), lnLo(X))
//
//   ALGORITHM
//
//   Here we use a table lookup method. The basic idea is that in
//   order to compute logl(Arg) for an argument Arg in [1,2), we
//   construct a value G such that G*Arg is close to 1 and that
//   logl(1/G) is obtainable easily from a table of values calculated
//   beforehand. Thus
//
//      logl(Arg) = logl(1/G) + logl(G*Arg)
//                = logl(1/G) + logl(1 + (G*Arg - 1))
//
//   Because |G*Arg - 1| is small, the second term on the right hand
//   side can be approximated by a short polynomial. We elaborate
//   this method in four steps.
//
//   Step 0: Initialization
//
//   We need to calculate logl( X ). Obtain N, S_hi such that
//
//      X = 2^N * S_hi exactly
//
//   where S_hi in [1,2)
//
//   Step 1: Argument Reduction
//
//   Based on S_hi, obtain G_1, G_2, G_3 from a table and calculate
//
//      G := G_1 * G_2 * G_3
//      r := (G * S_hi - 1)
//
//   These G_j's have the property that the product is exactly
//   representable and that |r| < 2^(-12) as a result.
//
//   Step 2: Approximation
//
//
//   logl(1 + r) is approximated by a short polynomial poly(r).
//
//   Step 3: Reconstruction
//
//
//   Finally, logl( X ) is given by
//
//   logl( X )   =   logl( 2^N * S_hi )
//                 ~=~  N*logl(2) + logl(1/G) + logl(1 + r)
//                 ~=~  N*logl(2) + logl(1/G) + poly(r).
//
//   IMPLEMENTATION
//
//   Step 0. Initialization
//   ----------------------
//
//   Z := X
//   N := unbaised exponent of Z
//   S_hi := 2^(-N) * Z
//
//   Step 1. Argument Reduction
//   --------------------------
//
//   Let
//
//      Z = 2^N * S_hi = 2^N * 1.d_1 d_2 d_3 ... d_63
//
//   We obtain G_1, G_2, G_3 by the following steps.
//
//
//      Define          X_0 := 1.d_1 d_2 ... d_14. This is extracted
//                      from S_hi.
//
//      Define          A_1 := 1.d_1 d_2 d_3 d_4. This is X_0 truncated
//                      to lsb = 2^(-4).
//
//      Define          index_1 := [ d_1 d_2 d_3 d_4 ].
//
//      Fetch           Z_1 := (1/A_1) rounded UP in fixed point with
//      fixed point     lsb = 2^(-15).
//                      Z_1 looks like z_0.z_1 z_2 ... z_15
//                      Note that the fetching is done using index_1.
//                      A_1 is actually not needed in the implementation
//                      and is used here only to explain how is the value
//                      Z_1 defined.
//
//      Fetch           G_1 := (1/A_1) truncated to 21 sig. bits.
//      floating pt.    Again, fetching is done using index_1. A_1
//                      explains how G_1 is defined.
//
//      Calculate       X_1 := X_0 * Z_1 truncated to lsb = 2^(-14)
//                           = 1.0 0 0 0 d_5 ... d_14
//                      This is accomplished by integer multiplication.
//                      It is proved that X_1 indeed always begin
//                      with 1.0000 in fixed point.
//
//
//      Define          A_2 := 1.0 0 0 0 d_5 d_6 d_7 d_8. This is X_1
//                      truncated to lsb = 2^(-8). Similar to A_1,
//                      A_2 is not needed in actual implementation. It
//                      helps explain how some of the values are defined.
//
//      Define          index_2 := [ d_5 d_6 d_7 d_8 ].
//
//      Fetch           Z_2 := (1/A_2) rounded UP in fixed point with
//      fixed point     lsb = 2^(-15). Fetch done using index_2.
//                      Z_2 looks like z_0.z_1 z_2 ... z_15
//
//      Fetch           G_2 := (1/A_2) truncated to 21 sig. bits.
//      floating pt.
//
//      Calculate       X_2 := X_1 * Z_2 truncated to lsb = 2^(-14)
//                           = 1.0 0 0 0 0 0 0 0 d_9 d_10 ... d_14
//                      This is accomplished by integer multiplication.
//                      It is proved that X_2 indeed always begin
//                      with 1.00000000 in fixed point.
//
//
//      Define          A_3 := 1.0 0 0 0 0 0 0 0 d_9 d_10 d_11 d_12 d_13 1.
//                      This is 2^(-14) + X_2 truncated to lsb = 2^(-13).
//
//      Define          index_3 := [ d_9 d_10 d_11 d_12 d_13 ].
//
//      Fetch           G_3 := (1/A_3) truncated to 21 sig. bits.
//      floating pt.    Fetch is done using index_3.
//
//      Compute         G := G_1 * G_2 * G_3.
//
//      This is done exactly since each of G_j only has 21 sig. bits.
//
//      Compute
//
//               r := (G*S_hi - 1)
//
//
//  Step 2. Approximation
//  ---------------------
//
//   This step computes an approximation to logl( 1 + r ) where r is the
//   reduced argument just obtained. It is proved that |r| <= 1.9*2^(-13);
//   thus logl(1+r) can be approximated by a short polynomial:
//
//      logl(1+r) ~=~ poly = r + Q1 r^2 + ... + Q4 r^5
//
//
//  Step 3. Reconstruction
//  ----------------------
//
//   This step computes the desired result of logl(X):
//
//      logl(X)  =   logl( 2^N * S_hi )
//               =   N*logl(2) + logl( S_hi )
//               =   N*logl(2) + logl(1/G) +
//                      logl(1 + G*S_hi - 1 )
//
//   logl(2), logl(1/G_j) are stored as pairs of (single,double) numbers:
//   log2_hi, log2_lo, log1byGj_hi, log1byGj_lo. The high parts are
//   single-precision numbers and the low parts are double precision
//   numbers. These have the property that
//
//      N*log2_hi + SUM ( log1byGj_hi )
//
//   is computable exactly in double-extended precision (64 sig. bits).
//   Finally
//
//      lnHi(X) := N*log2_hi + SUM ( log1byGj_hi )
//      lnLo(X) := poly_hi + [ poly_lo +
//              ( SUM ( log1byGj_lo ) + N*log2_lo ) ]
//
//
//*********************************************************************
// General Purpose Registers
// scratch registers
rPolDataPtr              = r2
rLnSinDataPtr            = r3
rExpX                    = r8
rSignifX                 = r9
rDelta                   = r10
rSignExpX                = r11
GR_ad_z_1                = r14
r17Ones                  = r15
GR_Index1                = r16
rSignif1andQ             = r17
GR_X_0                   = r18
GR_X_1                   = r19
GR_X_2                   = r20
GR_Z_1                   = r21
GR_Z_2                   = r22
GR_N                     = r23
rExpHalf                 = r24
rExp8                    = r25
rX0Dx                    = r25
GR_ad_tbl_1              = r26
GR_ad_tbl_2              = r27
GR_ad_tbl_3              = r28
GR_ad_q                  = r29
GR_ad_z_1                = r30
GR_ad_z_2                = r31
// stacked registers
rPFS_SAVED               = r32
GR_ad_z_3                = r33
rSgnGamAddr              = r34
rSgnGamSize              = r35
rLogDataPtr              = r36
rZ1offsett               = r37
rTmpPtr                  = r38
rTmpPtr2                 = r39
rTmpPtr3                 = r40
rExp2                    = r41
rExp2tom7                = r42
rZ625                    = r42
rExpOne                  = r43
rNegSingularity          = r44
rXint                    = r45
rTbl1Addr                = r46
rTbl2Addr                = r47
rTbl3Addr                = r48
rZ2Addr                  = r49
rRootsAddr               = r50
rRootsBndAddr            = r51
rRoot                    = r52
rRightBound              = r53
rLeftBound               = r54
rSignifDx                = r55
rBernulliPtr             = r56
rLnSinTmpPtr             = r56
rIndex1Dx                = r57
rIndexPol                = r58
GR_Index3                = r59
GR_Index2                = r60
rSgnGam                  = r61
rXRnd                    = r62

GR_SAVE_B0               = r63
GR_SAVE_GP               = r64
GR_SAVE_PFS              = r65
// output parameters when calling error handling routine
GR_Parameter_X           = r66
GR_Parameter_Y           = r67
GR_Parameter_RESULT      = r68
GR_Parameter_TAG         = r69

//********************************************************************
// Floating Point Registers
// CAUTION: due to the lack of registers there exist (below in the code)
// sometimes "unconventional" use of declared registers
//
fAbsX                    = f6
fDelX4                   = f6
fSignifX                 = f7
// macros for error handling routine
FR_X                     = f10 // first argument
FR_Y                     = f1  // second argument (lgammal has just one)
FR_RESULT                = f8  // result

// First 7 Bernulli numbers
fB2                      = f9
fLnDeltaL                = f9
fXSqr                    = f9
fB4                      = f10
fX4                      = f10
fB6                      = f11
fX6                      = f11
fB8                      = f12
fXSqrL                   = f12
fB10                     = f13
fRes7H                   = f13
fB12                     = f14
fRes7L                   = f14
fB14                     = f15

// stack registers
// Polynomial coefficients: A0, ..., A25
fA0                      = f32
fA0L                     = f33
fInvXL                   = f33
fA1                      = f34
fA1L                     = f35
fA2                      = f36
fA2L                     = f37
fA3                      = f38
fA3L                     = f39
fA4                      = f40
fA4L                     = f41
fRes6H                   = f41
fA5                      = f42
fB2L                     = f42
fA5L                     = f43
fMinNegStir              = f43
fRes6L                   = f43
fA6                      = f44
fMaxNegStir              = f44
fA7                      = f45
fLnDeltaH                = f45
fA8                      = f46
fBrnL                    = f46
fA9                      = f47
fBrnH                    = f47
fA10                     = f48
fRes5L                   = f48
fA11                     = f49
fRes5H                   = f49
fA12                     = f50
fDx6                     = f50
fA13                     = f51
fDx8                     = f51
fA14                     = f52
fDx4                     = f52
fA15                     = f53
fYL                      = f53
fh3Dx                    = f53
fA16                     = f54
fYH                      = f54
fH3Dx                    = f54
fA17                     = f55
fResLnDxL                = f55
fG3Dx                    = f55
fA18                     = f56
fResLnDxH                = f56
fh2Dx                    = f56
fA19                     = f57
fFloatNDx                = f57
fA20                     = f58
fPolyHiDx                = f58
fhDx                     = f58
fA21                     = f59
fRDxCub                  = f59
fHDx                     = f59
fA22                     = f60
fRDxSq                   = f60
fGDx                     = f60
fA23                     = f61
fPolyLoDx                = f61
fInvX3                   = f61
fA24                     = f62
fRDx                     = f62
fInvX8                   = f62
fA25                     = f63
fInvX4                   = f63
fPol                     = f64
fPolL                    = f65
// Coefficients of ln(sin(Pi*x)/Pi*x)
fLnSin2                  = f66
fLnSin2L                 = f67
fLnSin4                  = f68
fLnSin6                  = f69
fLnSin8                  = f70
fLnSin10                 = f71
fLnSin12                 = f72
fLnSin14                 = f73
fLnSin16                 = f74
fLnSin18                 = f75
fDelX8                   = f75
fLnSin20                 = f76
fLnSin22                 = f77
fDelX6                   = f77
fLnSin24                 = f78
fLnSin26                 = f79
fLnSin28                 = f80
fLnSin30                 = f81
fhDelX                   = f81
fLnSin32                 = f82
fLnSin34                 = f83
fLnSin36                 = f84
fXint                    = f85
fDxSqr                   = f85
fRes3L                   = f86
fRes3H                   = f87
fRes4H                   = f88
fRes4L                   = f89
fResH                    = f90
fResL                    = f91
fDx                      = f92
FR_MHalf                 = f93
fRes1H                   = f94
fRes1L                   = f95
fRes2H                   = f96
fRes2L                   = f97
FR_FracX                 = f98
fRcpX                    = f99
fLnSinH                  = f99
fTwo                     = f100
fMOne                    = f100
FR_G                     = f101
FR_H                     = f102
FR_h                     = f103
FR_G2                    = f104
FR_H2                    = f105
FR_poly_lo               = f106
FR_poly_hi               = f107
FR_h2                    = f108
FR_rsq                   = f109
FR_r                     = f110
FR_log2_hi               = f111
FR_log2_lo               = f112
fFloatN                  = f113
FR_Q4                    = f114
FR_G3                    = f115
FR_H3                    = f116
FR_h3                    = f117
FR_Q3                    = f118
FR_Q2                    = f119
FR_Q1                    = f120
fThirteen                = f121
fSix                     = f121
FR_rcub                  = f121
// Last three Bernulli numbers
fB16                     = f122
fB18                     = f123
fB20                     = f124
fInvX                    = f125
fLnSinL                  = f125
fDxSqrL                  = f126
fFltIntX                 = f126
fRoot                    = f127
fNormDx                  = f127

// Data tables
//==============================================================
RODATA
// ************* DO NOT CHANGE THE ORDER OF THESE TABLES *************
.align 16
LOCAL_OBJECT_START(lgammal_right_roots_data)
// List of all right roots themselves
data8 0x9D3FE4B007C360AB, 0x0000C000 // Range [-3, -2]
data8 0xC9306DE4F2CD7BEE, 0x0000C000 // Range [-4, -3]
data8 0x814273C2CCAC0618, 0x0000C001 // Range [-5, -4]
data8 0xA04352BF85B6C865, 0x0000C001 // Range [-6, -5]
data8 0xC00B592C4BE4676C, 0x0000C001 // Range [-7, -6]
data8 0xE0019FEF6FF0F5BF, 0x0000C001 // Range [-8, -7]
data8 0x80001A01459FC9F6, 0x0000C002 // Range [-9, -8]
data8 0x900002E3BB47D86D, 0x0000C002 // Range [-10, -9]
data8 0xA0000049F93BB992, 0x0000C002 // Range [-11, -10]
data8 0xB0000006B9915316, 0x0000C002 // Range [-12, -11]
data8 0xC00000008F76C773, 0x0000C002 // Range [-13, -12]
data8 0xD00000000B09230A, 0x0000C002 // Range [-14, -13]
data8 0xE000000000C9CBA5, 0x0000C002 // Range [-15, -14]
data8 0xF0000000000D73FA, 0x0000C002 // Range [-16, -15]
data8 0x8000000000006BA0, 0x0000C003 // Range [-17, -16]
data8 0x8800000000000655, 0x0000C003 // Range [-18, -17]
data8 0x900000000000005A, 0x0000C003 // Range [-19, -18]
data8 0x9800000000000005, 0x0000C003 // Range [-20, -19]
// List of bounds of ranges with special polynomial approximation near root
// Only significands of bounds are actually stored
data8 0xA000000000000000, 0x9800000000000000 // Bounds for root on [-3, -2]
data8 0xCAB88035C5EFBB41, 0xC7E05E31F4B02115 // Bounds for root on [-4, -3]
data8 0x817831B899735C72, 0x8114633941B8053A // Bounds for root on [-5, -4]
data8 0xA04E8B34C6AA9476, 0xA039B4A42978197B // Bounds for root on [-6, -5]
data8 0xC00D3D5E588A78A9, 0xC009BA25F7E858A6 // Bounds for root on [-7, -6]
data8 0xE001E54202991EB4, 0xE001648416CE897F // Bounds for root on [-8, -7]
data8 0x80001E56D13A6B9F, 0x8000164A3BAD888A // Bounds for root on [-9, -8]
data8 0x9000035F0529272A, 0x9000027A0E3D94F0 // Bounds for root on [-10, -9]
data8 0xA00000564D705880, 0xA000003F67EA0CC7 // Bounds for root on [-11, -10]
data8 0xB0000007D87EE0EF, 0xB0000005C3A122A5 // Bounds for root on [-12, -11]
data8 0xC0000000A75FE8B1, 0xC00000007AF818AC // Bounds for root on [-13, -12]
data8 0xD00000000CDFFE36, 0xD000000009758BBF // Bounds for root on [-14, -13]
data8 0xE000000000EB6D96, 0xE000000000ACF7B2 // Bounds for root on [-15, -14]
data8 0xF0000000000FB1F9, 0xF0000000000B87FB // Bounds for root on [-16, -15]
data8 0x8000000000007D90, 0x8000000000005C40 // Bounds for root on [-17, -16]
data8 0x8800000000000763, 0x880000000000056D // Bounds for root on [-18, -17]
data8 0x9000000000000069, 0x900000000000004D // Bounds for root on [-19, -18]
data8 0x9800000000000006, 0x9800000000000005 // Bounds for root on [-20, -19]
// List of all left roots themselves
data8 0xAFDA0850DEC8065E, 0x0000C000 // Range [-3, -2]
data8 0xFD238AA3E17F285C, 0x0000C000 // Range [-4, -3]
data8 0x9FBABBD37757E6A2, 0x0000C001 // Range [-5, -4]
data8 0xBFF497AC8FA06AFC, 0x0000C001 // Range [-6, -5]
data8 0xDFFE5FBB5C377FE8, 0x0000C001 // Range [-7, -6]
data8 0xFFFFCBFC0ACE7879, 0x0000C001 // Range [-8, -7]
data8 0x8FFFFD1C425E8100, 0x0000C002 // Range [-9, -8]
data8 0x9FFFFFB606BDFDCD, 0x0000C002 // Range [-10, -9]
data8 0xAFFFFFF9466E9F1B, 0x0000C002 // Range [-11, -10]
data8 0xBFFFFFFF70893874, 0x0000C002 // Range [-12, -11]
data8 0xCFFFFFFFF4F6DCF6, 0x0000C002 // Range [-13, -12]
data8 0xDFFFFFFFFF36345B, 0x0000C002 // Range [-14, -13]
data8 0xEFFFFFFFFFF28C06, 0x0000C002 // Range [-15, -14]
data8 0xFFFFFFFFFFFF28C0, 0x0000C002 // Range [-16, -15]
data8 0x87FFFFFFFFFFF9AB, 0x0000C003 // Range [-17, -16]
data8 0x8FFFFFFFFFFFFFA6, 0x0000C003 // Range [-18, -17]
data8 0x97FFFFFFFFFFFFFB, 0x0000C003 // Range [-19, -18]
data8 0x0000000000000000, 0x00000000 // pad to keep logic in the main path
// List of bounds of ranges with special polynomial approximation near root
// Only significands of bounds are actually stored
data8 0xB235880944CC758E, 0xADD2F1A9FBE76C8B // Bounds for root on [-3, -2]
data8 0xFD8E7844F307B07C, 0xFCA655C2152BDE4D // Bounds for root on [-4, -3]
data8 0x9FC4D876EE546967, 0x9FAEE4AF68BC4292 // Bounds for root on [-5, -4]
data8 0xBFF641FFBFCC44F1, 0xBFF2A47919F4BA89 // Bounds for root on [-6, -5]
data8 0xDFFE9C803DEFDD59, 0xDFFE18932EB723FE // Bounds for root on [-7, -6]
data8 0xFFFFD393FA47AFC3, 0xFFFFC317CF638AE1 // Bounds for root on [-8, -7]
data8 0x8FFFFD8840279925, 0x8FFFFC9DCECEEE92 // Bounds for root on [-9, -8]
data8 0x9FFFFFC0D34E2AF8, 0x9FFFFFA9619AA3B7 // Bounds for root on [-10, -9]
data8 0xAFFFFFFA41C18246, 0xAFFFFFF82025A23C // Bounds for root on [-11, -10]
data8 0xBFFFFFFF857ACB4E, 0xBFFFFFFF58032378 // Bounds for root on [-12, -11]
data8 0xCFFFFFFFF6934AB8, 0xCFFFFFFFF313EF0A // Bounds for root on [-13, -12]
data8 0xDFFFFFFFFF53A9E9, 0xDFFFFFFFFF13B5A5 // Bounds for root on [-14, -13]
data8 0xEFFFFFFFFFF482CB, 0xEFFFFFFFFFF03F4F // Bounds for root on [-15, -14]
data8 0xFFFFFFFFFFFF482D, 0xFFFFFFFFFFFF03F5 // Bounds for root on [-16, -15]
data8 0x87FFFFFFFFFFFA98, 0x87FFFFFFFFFFF896 // Bounds for root on [-17, -16]
data8 0x8FFFFFFFFFFFFFB3, 0x8FFFFFFFFFFFFF97 // Bounds for root on [-18, -17]
data8 0x97FFFFFFFFFFFFFC, 0x97FFFFFFFFFFFFFB // Bounds for root on [-19, -18]
LOCAL_OBJECT_END(lgammal_right_roots_data)

LOCAL_OBJECT_START(lgammal_0_Half_data)
// Polynomial coefficients for the lgammal(x), 0.0 < |x| < 0.5
data8 0xBFD9A4D55BEAB2D6, 0xBC8AA3C097746D1F //A3
data8 0x3FEA51A6625307D3, 0x3C7180E7BD2D0DCC //A2
data8 0xBFE2788CFC6FB618, 0xBC9E9346C4692BCC //A1
data8 0x8A8991563EC1BD13, 0x00003FFD //A4
data8 0xD45CE0BD52C27EF2, 0x0000BFFC //A5
data8 0xADA06587FA2BBD47, 0x00003FFC //A6
data8 0x9381D0ED2194902A, 0x0000BFFC //A7
data8 0x80859B3CF92D4192, 0x00003FFC //A8
data8 0xE4033517C622A946, 0x0000BFFB //A9
data8 0xCD00CE67A51FC82A, 0x00003FFB //A10
data8 0xBA44E2A96C3B5700, 0x0000BFFB //A11
data8 0xAAAD008FA46DBD99, 0x00003FFB //A12
data8 0x9D604AC65A41153D, 0x0000BFFB //A13
data8 0x917CECB864B5A861, 0x00003FFB //A14
data8 0x85A4810EB730FDE4, 0x0000BFFB //A15
data8 0xEF2761C38BD21F77, 0x00003FFA //A16
data8 0xC913043A128367DA, 0x0000BFFA //A17
data8 0x96A29B71FF7AFFAA, 0x00003FFA //A18
data8 0xBB9FFA1A5FE649BB, 0x0000BFF9 //A19
data8 0xB17982CD2DAA0EE3, 0x00003FF8 //A20
data8 0xDE1DDCBFFB9453F0, 0x0000BFF6 //A21
data8 0x87FBF5D7ACD9FA9D, 0x00003FF4 //A22
LOCAL_OBJECT_END(lgammal_0_Half_data)

LOCAL_OBJECT_START(Constants_Q)
// log2_hi, log2_lo, Q_4, Q_3, Q_2, and Q_1
data4  0x00000000,0xB1721800,0x00003FFE,0x00000000
data4  0x4361C4C6,0x82E30865,0x0000BFE2,0x00000000
data4  0x328833CB,0xCCCCCAF2,0x00003FFC,0x00000000
data4  0xA9D4BAFB,0x80000077,0x0000BFFD,0x00000000
data4  0xAAABE3D2,0xAAAAAAAA,0x00003FFD,0x00000000
data4  0xFFFFDAB7,0xFFFFFFFF,0x0000BFFD,0x00000000
LOCAL_OBJECT_END(Constants_Q)

LOCAL_OBJECT_START(Constants_Z_1)
// Z1 - 16 bit fixed
data4  0x00008000
data4  0x00007879
data4  0x000071C8
data4  0x00006BCB
data4  0x00006667
data4  0x00006187
data4  0x00005D18
data4  0x0000590C
data4  0x00005556
data4  0x000051EC
data4  0x00004EC5
data4  0x00004BDB
data4  0x00004925
data4  0x0000469F
data4  0x00004445
data4  0x00004211
LOCAL_OBJECT_END(Constants_Z_1)

LOCAL_OBJECT_START(Constants_G_H_h1)
// G1 and H1 - IEEE single and h1 - IEEE double
data4  0x3F800000,0x00000000,0x00000000,0x00000000
data4  0x3F70F0F0,0x3D785196,0x617D741C,0x3DA163A6
data4  0x3F638E38,0x3DF13843,0xCBD3D5BB,0x3E2C55E6
data4  0x3F579430,0x3E2FF9A0,0xD86EA5E7,0xBE3EB0BF
data4  0x3F4CCCC8,0x3E647FD6,0x86B12760,0x3E2E6A8C
data4  0x3F430C30,0x3E8B3AE7,0x5C0739BA,0x3E47574C
data4  0x3F3A2E88,0x3EA30C68,0x13E8AF2F,0x3E20E30F
data4  0x3F321640,0x3EB9CEC8,0xF2C630BD,0xBE42885B
data4  0x3F2AAAA8,0x3ECF9927,0x97E577C6,0x3E497F34
data4  0x3F23D708,0x3EE47FC5,0xA6B0A5AB,0x3E3E6A6E
data4  0x3F1D89D8,0x3EF8947D,0xD328D9BE,0xBDF43E3C
data4  0x3F17B420,0x3F05F3A1,0x0ADB090A,0x3E4094C3
data4  0x3F124920,0x3F0F4303,0xFC1FE510,0xBE28FBB2
data4  0x3F0D3DC8,0x3F183EBF,0x10FDE3FA,0x3E3A7895
data4  0x3F088888,0x3F20EC80,0x7CC8C98F,0x3E508CE5
data4  0x3F042108,0x3F29516A,0xA223106C,0xBE534874
LOCAL_OBJECT_END(Constants_G_H_h1)

LOCAL_OBJECT_START(Constants_Z_2)
// Z2 - 16 bit fixed
data4  0x00008000
data4  0x00007F81
data4  0x00007F02
data4  0x00007E85
data4  0x00007E08
data4  0x00007D8D
data4  0x00007D12
data4  0x00007C98
data4  0x00007C20
data4  0x00007BA8
data4  0x00007B31
data4  0x00007ABB
data4  0x00007A45
data4  0x000079D1
data4  0x0000795D
data4  0x000078EB
LOCAL_OBJECT_END(Constants_Z_2)

LOCAL_OBJECT_START(Constants_G_H_h2)
// G2 and H2 - IEEE single and h2 - IEEE double
data4  0x3F800000,0x00000000,0x00000000,0x00000000
data4  0x3F7F00F8,0x3B7F875D,0x22C42273,0x3DB5A116
data4  0x3F7E03F8,0x3BFF015B,0x21F86ED3,0x3DE620CF
data4  0x3F7D08E0,0x3C3EE393,0x484F34ED,0xBDAFA07E
data4  0x3F7C0FC0,0x3C7E0586,0x3860BCF6,0xBDFE07F0
data4  0x3F7B1880,0x3C9E75D2,0xA78093D6,0x3DEA370F
data4  0x3F7A2328,0x3CBDC97A,0x72A753D0,0x3DFF5791
data4  0x3F792FB0,0x3CDCFE47,0xA7EF896B,0x3DFEBE6C
data4  0x3F783E08,0x3CFC15D0,0x409ECB43,0x3E0CF156
data4  0x3F774E38,0x3D0D874D,0xFFEF71DF,0xBE0B6F97
data4  0x3F766038,0x3D1CF49B,0x5D59EEE8,0xBE080483
data4  0x3F757400,0x3D2C531D,0xA9192A74,0x3E1F91E9
data4  0x3F748988,0x3D3BA322,0xBF72A8CD,0xBE139A06
data4  0x3F73A0D0,0x3D4AE46F,0xF8FBA6CF,0x3E1D9202
data4  0x3F72B9D0,0x3D5A1756,0xBA796223,0xBE1DCCC4
data4  0x3F71D488,0x3D693B9D,0xB6B7C239,0xBE049391
LOCAL_OBJECT_END(Constants_G_H_h2)

LOCAL_OBJECT_START(Constants_G_H_h3)
// G3 and H3 - IEEE single and h3 - IEEE double
data4  0x3F7FFC00,0x38800100,0x562224CD,0x3D355595
data4  0x3F7FF400,0x39400480,0x06136FF6,0x3D8200A2
data4  0x3F7FEC00,0x39A00640,0xE8DE9AF0,0x3DA4D68D
data4  0x3F7FE400,0x39E00C41,0xB10238DC,0xBD8B4291
data4  0x3F7FDC00,0x3A100A21,0x3B1952CA,0xBD89CCB8
data4  0x3F7FD400,0x3A300F22,0x1DC46826,0xBDB10707
data4  0x3F7FCC08,0x3A4FF51C,0xF43307DB,0x3DB6FCB9
data4  0x3F7FC408,0x3A6FFC1D,0x62DC7872,0xBD9B7C47
data4  0x3F7FBC10,0x3A87F20B,0x3F89154A,0xBDC3725E
data4  0x3F7FB410,0x3A97F68B,0x62B9D392,0xBD93519D
data4  0x3F7FAC18,0x3AA7EB86,0x0F21BD9D,0x3DC18441
data4  0x3F7FA420,0x3AB7E101,0x2245E0A6,0xBDA64B95
data4  0x3F7F9C20,0x3AC7E701,0xAABB34B8,0x3DB4B0EC
data4  0x3F7F9428,0x3AD7DD7B,0x6DC40A7E,0x3D992337
data4  0x3F7F8C30,0x3AE7D474,0x4F2083D3,0x3DC6E17B
data4  0x3F7F8438,0x3AF7CBED,0x811D4394,0x3DAE314B
data4  0x3F7F7C40,0x3B03E1F3,0xB08F2DB1,0xBDD46F21
data4  0x3F7F7448,0x3B0BDE2F,0x6D34522B,0xBDDC30A4
data4  0x3F7F6C50,0x3B13DAAA,0xB1F473DB,0x3DCB0070
data4  0x3F7F6458,0x3B1BD766,0x6AD282FD,0xBDD65DDC
data4  0x3F7F5C68,0x3B23CC5C,0xF153761A,0xBDCDAB83
data4  0x3F7F5470,0x3B2BC997,0x341D0F8F,0xBDDADA40
data4  0x3F7F4C78,0x3B33C711,0xEBC394E8,0x3DCD1BD7
data4  0x3F7F4488,0x3B3BBCC6,0x52E3E695,0xBDC3532B
data4  0x3F7F3C90,0x3B43BAC0,0xE846B3DE,0xBDA3961E
data4  0x3F7F34A0,0x3B4BB0F4,0x785778D4,0xBDDADF06
data4  0x3F7F2CA8,0x3B53AF6D,0xE55CE212,0x3DCC3ED1
data4  0x3F7F24B8,0x3B5BA620,0x9E382C15,0xBDBA3103
data4  0x3F7F1CC8,0x3B639D12,0x5C5AF197,0x3D635A0B
data4  0x3F7F14D8,0x3B6B9444,0x71D34EFC,0xBDDCCB19
data4  0x3F7F0CE0,0x3B7393BC,0x52CD7ADA,0x3DC74502
data4  0x3F7F04F0,0x3B7B8B6D,0x7D7F2A42,0xBDB68F17
LOCAL_OBJECT_END(Constants_G_H_h3)

LOCAL_OBJECT_START(lgammal_data)
// Positive overflow value
data8 0xB8D54C8BFFFDEBF4, 0x00007FF1
LOCAL_OBJECT_END(lgammal_data)

LOCAL_OBJECT_START(lgammal_Stirling)
// Coefficients needed for Strirling's formula
data8 0x3FED67F1C864BEB4 // High part of 0.5*ln(2*Pi)
data8 0x3C94D252F2400510 // Low part of 0.5*ln(2*Pi)
//
// Bernulli numbers used in Striling's formula for -2^63 < |x| < -13.0
//(B1H, B1L) = 8.3333333333333333333262747254e-02
data8 0x3FB5555555555555, 0x3C55555555555555
data8 0xB60B60B60B60B60B, 0x0000BFF6 //B2 = -2.7777777777777777777777777778e-03
data8 0xD00D00D00D00D00D, 0x00003FF4 //B3 = 7.9365079365079365079365079365e-04
data8 0x9C09C09C09C09C0A, 0x0000BFF4 //B4 = -5.9523809523809523809523809524e-04
data8 0xDCA8F158C7F91AB8, 0x00003FF4 //B5 = 8.4175084175084175084175084175e-04
data8 0xFB5586CCC9E3E410, 0x0000BFF5 //B6 = -1.9175269175269175269175269175e-03
data8 0xD20D20D20D20D20D, 0x00003FF7 //B7 = 6.4102564102564102564102564103e-03
data8 0xF21436587A9CBEE1, 0x0000BFF9 //B8 = -2.9550653594771241830065359477e-02
data8 0xB7F4B1C0F033FFD1, 0x00003FFC //B9 = 1.7964437236883057316493849002e-01
data8 0xB23B3808C0F9CF6E, 0x0000BFFF //B10 = -1.3924322169059011164274322169e+00
// Polynomial coefficients for Stirling's formula, -13.0 < x < -6.0
data8 0x3FB5555555555555, 0x3C4D75060289C58B //A0
data8 0xB60B60B60B0F0876, 0x0000BFF6 //A1
data8 0xD00D00CE54B1256C, 0x00003FF4 //A2
data8 0x9C09BF46B58F75E1, 0x0000BFF4 //A3
data8 0xDCA8483BC91ACC6D, 0x00003FF4 //A4
data8 0xFB3965C939CC9FEE, 0x0000BFF5 //A5
data8 0xD0723ADE3F0BC401, 0x00003FF7 //A6
data8 0xE1ED7434E81F0B73, 0x0000BFF9 //A7
data8 0x8069C6982F993283, 0x00003FFC //A8
data8 0xC271F65BFA5BEE3F, 0x0000BFFD //A9
LOCAL_OBJECT_END(lgammal_Stirling)

LOCAL_OBJECT_START(lgammal_lnsin_data)
// polynomial approximation of -ln(sin(Pi*x)/(Pi*x)), 0 < x <= 0.5
data8 0x3FFA51A6625307D3, 0x3C81873332FAF94C //A2
data8 0x8A8991563EC241C3, 0x00003FFE //A4
data8 0xADA06588061805DF, 0x00003FFD //A6
data8 0x80859B57C338D0F7, 0x00003FFD //A8
data8 0xCD00F1C2D78754BD, 0x00003FFC //A10
data8 0xAAB56B1D3A1F4655, 0x00003FFC //A12
data8 0x924B6F2FBBED12B1, 0x00003FFC //A14
data8 0x80008E58765F43FC, 0x00003FFC //A16
data8 0x3FBC718EC115E429//A18
data8 0x3FB99CE544FE183E//A20
data8 0x3FB7251C09EAAD89//A22
data8 0x3FB64A970733628C//A24
data8 0x3FAC92D6802A3498//A26
data8 0x3FC47E1165261586//A28
data8 0xBFCA1BAA434750D4//A30
data8 0x3FE460001C4D5961//A32
data8 0xBFE6F06A3E4908AD//A34
data8 0x3FE300889EBB203A//A36
LOCAL_OBJECT_END(lgammal_lnsin_data)

LOCAL_OBJECT_START(lgammal_half_3Q_data)
// Polynomial coefficients for the lgammal(x), 0.5 <= x < 0.75
data8 0xBFF7A648EE90C62E, 0x3C713F326857E066 // A3, A0L
data8 0xBFF73E4B8BA780AE, 0xBCA953BC788877EF // A1, A1L
data8 0x403774DCD58D0291, 0xC0415254D5AE6623 // D0, D1
data8 0x40B07213855CBFB0, 0xC0B8855E25D2D229 // C20, C21
data8 0x3FFB359F85FF5000, 0x3C9BAECE6EF9EF3A // A2, A2L
data8 0x3FD717D498A3A8CC, 0xBC9088E101CFEDFA  // A0, A3L
data8 0xAFEF36CC5AEC3FF0, 0x00004002 // E6
data8 0xABE2054E1C34E791, 0x00004001 // E4
data8 0xB39343637B2900D1, 0x00004000 // E2
data8 0xD74FB710D53F58F6, 0x00003FFF // E0
data8 0x4070655963BA4256, 0xC078DA9D263C4EA3 // D6, D7
data8 0x405CD2B6A9B90978, 0xC065B3B9F4F4F171 // D4, D5
data8 0x4049BC2204CF61FF, 0xC05337227E0BA152 // D2, D3
data8 0x4095509A50C07A96, 0xC0A0747949D2FB45 // C18, C19
data8 0x4082ECCBAD709414, 0xC08CD02FB088A702 // C16, C17
data8 0xFFE4B2A61B508DD5, 0x0000C002 // E7
data8 0xF461ADB8AE17E0A5, 0x0000C001 // E5
data8 0xF5BE8B0B90325F20, 0x0000C000 // E3
data8 0x877B275F3FB78DCA, 0x0000C000 // E1
LOCAL_OBJECT_END(lgammal_half_3Q_data)

LOCAL_OBJECT_START(lgammal_half_3Q_neg_data)
// Polynomial coefficients for the lgammal(x), -0.75 < x <= -0.5
data8 0xC014836EFD94899C, 0x3C9835679663B44F // A3, A0L
data8 0xBFF276C7B4FB1875, 0xBC92D3D9FA29A1C0 // A1, A1L
data8 0x40C5178F24E1A435, 0xC0D9DE84FBC5D76A // D0, D1
data8 0x41D4D1B236BF6E93, 0xC1EBB0445CE58550 // C20, C21
data8 0x4015718CD67F63D3, 0x3CC5354B6F04B59C // A2, A2L
data8 0x3FF554493087E1ED, 0xBCB72715E37B02B9 // A0, A3L
data8 0xE4AC7E915FA72229, 0x00004009 // E6
data8 0xA28244206395FCC6, 0x00004007 // E4
data8 0xFB045F19C07B2544, 0x00004004 // E2
data8 0xE5C8A6E6A9BA7D7B, 0x00004002 // E0
data8 0x4143943B55BF5118, 0xC158AC05EA675406 // D6, D7
data8 0x4118F6833D19717C, 0xC12F51A6F375CC80 // D4, D5
data8 0x40F00C209483481C, 0xC103F1DABF750259 // D2, D3
data8 0x4191038F2D8F9E40, 0xC1A413066DA8AE4A // C18, C19
data8 0x4170B537EDD833DE, 0xC1857E79424C61CE // C16, C17
data8 0x8941D8AB4855DB73, 0x0000C00B // E7
data8 0xBB822B131BD2E813, 0x0000C008 // E5
data8 0x852B4C03B83D2D4F, 0x0000C006 // E3
data8 0xC754CA7E2DDC0F1F, 0x0000C003 // E1
LOCAL_OBJECT_END(lgammal_half_3Q_neg_data)

LOCAL_OBJECT_START(lgammal_2Q_4_data)
// Polynomial coefficients for the lgammal(x), 2.25 <= |x| < 4.0
data8 0xBFCA4D55BEAB2D6F, 0x3C7ABC9DA14141F5 // A3, A0L
data8 0x3FFD8773039049E7, 0x3C66CB7957A95BA4 // A1, A1L
data8 0x3F45C3CC79E91E7D, 0xBF3A8E5005937E97 // D0, D1
data8 0x3EC951E35E1C9203, 0xBEB030A90026C5DF // C20, C21
data8 0x3FE94699894C1F4C, 0x3C91884D21D123F1 // A2, A2L
data8 0x3FE62E42FEFA39EF, 0xBC66480CEB70870F // A0, A3L
data8 0xF1C2EAFF0B3A7579, 0x00003FF5 // E6
data8 0xB36AF863926B55A3, 0x00003FF7 // E4
data8 0x9620656185BB44CA, 0x00003FF9 // E2
data8 0xA264558FB0906AFF, 0x00003FFB // E0
data8 0x3F03D59E9666C961, 0xBEF91115893D84A6 // D6, D7
data8 0x3F19333611C46225, 0xBF0F89EB7D029870 // D4, D5
data8 0x3F3055A96B347AFE, 0xBF243B5153E178A8 // D2, D3
data8 0x3ED9A4AEF30C4BB2, 0xBED388138B1CEFF2 // C18, C19
data8 0x3EEF7945A3C3A254, 0xBEE36F32A938EF11 // C16, C17
data8 0x9028923F47C82118, 0x0000BFF5 // E7
data8 0xCE0DAAFB6DC93B22, 0x0000BFF6 // E5
data8 0xA0D0983B34AC4C8D, 0x0000BFF8 // E3
data8 0x94D6C50FEB8B0CE7, 0x0000BFFA // E1
LOCAL_OBJECT_END(lgammal_2Q_4_data)

LOCAL_OBJECT_START(lgammal_4_8_data)
// Polynomial coefficients for the lgammal(x), 4.0 <= |x| < 8.0
data8 0xBFD6626BC9B31B54, 0x3CAA53C82493A92B // A3, A0L
data8 0x401B4C420A50AD7C, 0x3C8C6E9929F789A3 // A1, A1L
data8 0x3F49410427E928C2, 0xBF3E312678F8C146 // D0, D1
data8 0x3ED51065F7CD5848, 0xBED052782A03312F // C20, C21
data8 0x3FF735973273D5EC, 0x3C831DFC65BF8CCF // A2, A2L
data8 0x401326643C4479C9, 0xBC6FA0498C5548A6 // A0, A3L
data8 0x9382D8B3CD4EB7E3, 0x00003FF6 // E6
data8 0xE9F92CAD8A85CBCD, 0x00003FF7 // E4
data8 0xD58389FE38258CEC, 0x00003FF9 // E2
data8 0x81310136363AE8AA, 0x00003FFC // E0
data8 0x3F04F0AE38E78570, 0xBEF9E2144BB8F03C // D6, D7
data8 0x3F1B5E992A6CBC2A, 0xBF10F3F400113911 // D4, D5
data8 0x3F323EE00AAB7DEE, 0xBF2640FDFA9FB637 // D2, D3
data8 0x3ED2143EBAFF067A, 0xBEBBDEB92D6FF35D // C18, C19
data8 0x3EF173A42B69AAA4, 0xBEE78B9951A2EAA5 // C16, C17
data8 0xAB3CCAC6344E52AA, 0x0000BFF5 // E7
data8 0x81ACCB8915B16508, 0x0000BFF7 // E5
data8 0xDA62C7221102C426, 0x0000BFF8 // E3
data8 0xDF1BD44C4083580A, 0x0000BFFA // E1
LOCAL_OBJECT_END(lgammal_4_8_data)

LOCAL_OBJECT_START(lgammal_loc_min_data)
// Polynomial coefficients for the lgammal(x), 1.3125 <= x < 1.5625
data8 0xBB16C31AB5F1FB71, 0x00003FFF // xMin - point of local minimum
data8 0xBFC2E4278DC6BC23, 0xBC683DA8DDCA9650 // A3, A0L
data8 0x3BD4DB7D0CA61D5F, 0x386E719EDD01D801 // A1, A1L
data8 0x3F4CC72638E1D93F, 0xBF4228EC9953CCB9 // D0, D1
data8 0x3ED222F97A04613E,0xBED3DDD58095CB6C  // C20, C21
data8 0x3FDEF72BC8EE38AB, 0x3C863AFF3FC48940 // A2, A2L
data8 0xBFBF19B9BCC38A41,  0xBC7425F1BFFC1442// A0, A3L
data8 0x941890032BEB34C3, 0x00003FF6 // E6
data8 0xC7E701591CE534BC, 0x00003FF7 // E4
data8 0x93373CBD05138DD4, 0x00003FF9 // E2
data8 0x845A14A6A81C05D6, 0x00003FFB // E0
data8 0x3F0F6C4DF6D47A13, 0xBF045DCDB5B49E19 // D6, D7
data8 0x3F22E23345DDE59C, 0xBF1851159AFB1735 // D4, D5
data8 0x3F37101EA4022B78, 0xBF2D721E6323AF13 // D2, D3
data8 0x3EE691EBE82DF09D, 0xBEDD42550961F730 // C18, C19
data8 0x3EFA793EDE99AD85, 0xBEF14000108E70BE // C16, C17
data8 0xB7CBC033ACE0C99C, 0x0000BFF5 // E7
data8 0xF178D1F7B1A45E27, 0x0000BFF6 // E5
data8 0xA8FCFCA8106F471C, 0x0000BFF8 // E3
data8 0x864D46FA898A9AD2, 0x0000BFFA // E1
LOCAL_OBJECT_END(lgammal_loc_min_data)

LOCAL_OBJECT_START(lgammal_03Q_1Q_data)
// Polynomial coefficients for the lgammal(x), 0.75 <= |x| < 1.3125
data8 0x3FD151322AC7D848, 0x3C7184DE0DB7B4EE // A4, A2L
data8 0x3FD9A4D55BEAB2D6, 0x3C9E934AAB10845F // A3, A1L
data8 0x3FB111289C381259, 0x3FAFFFCFB32AE18D // D2, D3
data8 0x3FB3B1D9E0E3E00D, 0x3FB2496F0D3768DF // D0, D1
data8 0xBA461972C057D439, 0x00003FFB         // E6
data8 0x3FEA51A6625307D3, 0x3C76ABC886A72DA2 // A2, A4L
data8 0x3FA8EFE46B32A70E, 0x3F8F31B3559576B6 // C17, C20
data8 0xE403383700387D85, 0x00003FFB // E4
data8 0x9381D0EE74BF7251, 0x00003FFC // E2
data8 0x3FAA2177A6D28177, 0x3FA4895E65FBD995 // C18, C19
data8 0x3FAAED2C77DBEE5D, 0x3FA94CA59385512C // D6, D7
data8 0x3FAE1F522E8A5941, 0x3FAC785EF56DD87E // D4, D5
data8 0x3FB556AD5FA56F0A, 0x3FA81F416E87C783 // E7, C16
data8 0xCD00F1C2DC2C9F1E, 0x00003FFB // E5
data8 0x3FE2788CFC6FB618, 0x3C8E52519B5B17CB // A1, A3L
data8 0x80859B57C3E7F241, 0x00003FFC // E3
data8 0xADA065880615F401, 0x00003FFC // E1
data8 0xD45CE0BD530AB50E, 0x00003FFC // E0
LOCAL_OBJECT_END(lgammal_03Q_1Q_data)

LOCAL_OBJECT_START(lgammal_13Q_2Q_data)
// Polynomial coefficients for the lgammal(x), 1.5625 <= |x| < 2.25
data8 0x3F951322AC7D8483, 0x3C71873D88C6539D // A4, A2L
data8 0xBFB13E001A557606, 0x3C56CB907018A101 // A3, A1L
data8 0xBEC11B2EC1E7F6FC, 0x3EB0064ED9824CC7 // D2, D3
data8 0xBEE3CBC963EC103A, 0x3ED2597A330C107D // D0, D1
data8 0xBC6F2DEBDFE66F38, 0x0000BFF0 // E6
data8 0x3FD4A34CC4A60FA6, 0x3C3AFC9BF775E8A0 // A2, A4L
data8 0x3E48B0C542F85B32, 0xBE347F12EAF787AB // C17, C20
data8 0xE9FEA63B6984FA1E, 0x0000BFF2 // E4
data8 0x9C562E15FC703BBF, 0x0000BFF5 // E2
data8 0xBE3C12A50AB0355E, 0xBE1C941626AE4717 // C18, C19
data8 0xBE7AFA8714342BC4,0x3E69A12D2B7761CB // D6, D7
data8 0xBE9E25EF1D526730, 0x3E8C762291889B99 // D4, D5
data8 0x3EF580DCEE754733, 0xBE57C811D070549C // E7, C16
data8 0xD093D878BE209C98, 0x00003FF1 // E5
data8 0x3FDB0EE6072093CE, 0xBC6024B9E81281C4 // A1, A3L
data8 0x859B57C31CB77D96, 0x00003FF4 // E3
data8 0xBD6EB756DB617E8D, 0x00003FF6 // E1
data8 0xF2027E10C7AF8C38, 0x0000BFF7 // E0
LOCAL_OBJECT_END(lgammal_13Q_2Q_data)

LOCAL_OBJECT_START(lgammal_8_10_data)
// Polynomial coefficients for the lgammal(x), 8.0 <= |x| < 10.0
// Multi Precision terms
data8 0x40312008A3A23E5C, 0x3CE020B4F2E4083A //A1
data8 0x4025358E82FCB70C, 0x3CD4A5A74AF7B99C //A0
// Native precision terms
data8 0xF0AA239FFBC616D2, 0x00004000 //A2
data8 0x96A8EA798FE57D66, 0x0000BFFF //A3
data8 0x8D501B7E3B9B9BDB, 0x00003FFE //A4
data8 0x9EE062401F4B1DC2, 0x0000BFFD //A5
data8 0xC63FD8CD31E93431, 0x00003FFC //A6
data8 0x8461101709C23C30, 0x0000BFFC //A7
data8 0xB96D7EA7EF3648B2, 0x00003FFB //A8
data8 0x86886759D2ACC906, 0x0000BFFB //A9
data8 0xC894B6E28265B183, 0x00003FFA //A10
data8 0x98C4348CAD821662, 0x0000BFFA //A11
data8 0xEC9B092226A94DF2, 0x00003FF9 //A12
data8 0xB9F169FF9B98CDDC, 0x0000BFF9 //A13
data8 0x9A3A32BB040894D3, 0x00003FF9 //A14
data8 0xF9504CCC1003B3C3, 0x0000BFF8 //A15
LOCAL_OBJECT_END(lgammal_8_10_data)

LOCAL_OBJECT_START(lgammal_03Q_6_data)
// Polynomial coefficients for the lgammal(x), 0.75 <= |x| < 1.0
data8 0xBFBC47DCA479E295, 0xBC607E6C1A379D55 //A3
data8 0x3FCA051C372609ED, 0x3C7B02D73EB7D831 //A0
data8 0xBFE15FAFA86B04DB, 0xBC3F52EE4A8945B5 //A1
data8 0x3FD455C4FF28F0BF, 0x3C75F8C6C99F30BB //A2
data8 0xD2CF04CD934F03E1, 0x00003FFA //A4
data8 0xDB4ED667E29256E1, 0x0000BFF9 //A5
data8 0xF155A33A5B6021BF, 0x00003FF8 //A6
data8 0x895E9B9D386E0338, 0x0000BFF8 //A7
data8 0xA001BE94B937112E, 0x00003FF7 //A8
data8 0xBD82846E490ED048, 0x0000BFF6 //A9
data8 0xE358D24EC30DBB5D, 0x00003FF5 //A10
data8 0x89C4F3652446B78B, 0x0000BFF5 //A11
data8 0xA86043E10280193D, 0x00003FF4 //A12
data8 0xCF3A2FBA61EB7682, 0x0000BFF3 //A13
data8 0x3F300900CC9200EC //A14
data8 0xBF23F42264B94AE8 //A15
data8 0x3F18EEF29895FE73 //A16
data8 0xBF0F3C4563E3EDFB //A17
data8 0x3F0387DBBC385056 //A18
data8 0xBEF81B4004F92900 //A19
data8 0x3EECA6692A9A5B81 //A20
data8 0xBEDF61A0059C15D3 //A21
data8 0x3ECDA9F40DCA0111 //A22
data8 0xBEB60FE788217BAF //A23
data8 0x3E9661D795DFC8C6 //A24
data8 0xBE66C7756A4EDEE5 //A25
// Polynomial coefficients for the lgammal(x), 1.0 <= |x| < 2.0
data8 0xBFC1AE55B180726B, 0xBC7DE1BC478453F5 //A3
data8 0xBFBEEB95B094C191, 0xBC53456FF6F1C9D9 //A0
data8 0x3FA2AED059BD608A, 0x3C0B65CC647D557F //A1
data8 0x3FDDE9E64DF22EF2, 0x3C8993939A8BA8E4 //A2
data8 0xF07C206D6B100CFF, 0x00003FFA //A4
data8 0xED2CEA9BA52FE7FB, 0x0000BFF9 //A5
data8 0xFCE51CED52DF3602, 0x00003FF8 //A6
data8 0x8D45D27872326619, 0x0000BFF8 //A7
data8 0xA2B78D6BCEBE27F7, 0x00003FF7 //A8
data8 0xBF6DC0996A895B6F, 0x0000BFF6 //A9
data8 0xE4B9AD335AF82D79, 0x00003FF5 //A10
data8 0x8A451880195362A1, 0x0000BFF5 //A11
data8 0xA8BE35E63089A7A9, 0x00003FF4 //A12
data8 0xCF7FA175FA11C40C, 0x0000BFF3 //A13
data8 0x3F300C282FAA3B02 //A14
data8 0xBF23F6AEBDA68B80 //A15
data8 0x3F18F6860E2224DD //A16
data8 0xBF0F542B3CE32F28 //A17
data8 0x3F039436218C9BF8 //A18
data8 0xBEF8AE6307677AEC //A19
data8 0x3EF0B55527B3A211 //A20
data8 0xBEE576AC995E7605 //A21
data8 0x3ED102DDC1365D2D //A22
data8 0xBEC442184F97EA54 //A23
data8 0x3ED4D2283DFE5FC6 //A24
data8 0xBECB9219A9B46787 //A25
// Polynomial coefficients for the lgammal(x), 2.0 <= |x| < 3.0
data8 0xBFCA4D55BEAB2D6F, 0xBC66F80E5BFD5AF5 //A3
data8 0x3FE62E42FEFA39EF, 0x3C7ABC9E3B347E3D //A0
data8 0x3FFD8773039049E7, 0x3C66CB9007C426EA //A1
data8 0x3FE94699894C1F4C, 0x3C918726EB111663 //A2
data8 0xA264558FB0906209, 0x00003FFB //A4
data8 0x94D6C50FEB902ADC, 0x0000BFFA //A5
data8 0x9620656184243D17, 0x00003FF9 //A6
data8 0xA0D0983B8BCA910B, 0x0000BFF8 //A7
data8 0xB36AF8559B222BD3, 0x00003FF7 //A8
data8 0xCE0DACB3260AE6E5, 0x0000BFF6 //A9
data8 0xF1C2C0BF0437C7DB, 0x00003FF5 //A10
data8 0x902A2F2F3AB74A92, 0x0000BFF5 //A11
data8 0xAE05009B1B2C6E4C, 0x00003FF4 //A12
data8 0xD5B71F6456D7D4CB, 0x0000BFF3 //A13
data8 0x3F2F0351D71BC9C6 //A14
data8 0xBF2B53BC56A3B793 //A15
data8 0xBF18B12DC6F6B861 //A16
data8 0xBF43EE6EB5215C2F //A17
data8 0xBF5474787CDD455E //A18
data8 0xBF642B503C9C060A //A19
data8 0xBF6E07D1AA254AA3 //A20
data8 0xBF71C785443AAEE8 //A21
data8 0xBF6F67BF81B71052 //A22
data8 0xBF63E4BCCF4FFABF //A23
data8 0xBF50067F8C671D5A //A24
data8 0xBF29C770D680A5AC //A25
// Polynomial coefficients for the lgammal(x), 4.0 <= |x| < 6.0
data8 0xBFD6626BC9B31B54, 0xBC85AABE08680902 //A3
data8 0x401326643C4479C9, 0x3CAA53C26F31E364 //A0
data8 0x401B4C420A50AD7C, 0x3C8C76D55E57DD8D //A1
data8 0x3FF735973273D5EC, 0x3C83A0B78E09188A //A2
data8 0x81310136363AAB6D, 0x00003FFC //A4
data8 0xDF1BD44C4075C0E6, 0x0000BFFA //A5
data8 0xD58389FE38D8D664, 0x00003FF9 //A6
data8 0xDA62C7221D5B5F87, 0x0000BFF8 //A7
data8 0xE9F92CAD0263E157, 0x00003FF7 //A8
data8 0x81ACCB8606C165FE, 0x0000BFF7 //A9
data8 0x9382D8D263D1C2A3, 0x00003FF6 //A10
data8 0xAB3CCBA4C853B12C, 0x0000BFF5 //A11
data8 0xCA0818BBCCC59296, 0x00003FF4 //A12
data8 0xF18912691CBB5BD0, 0x0000BFF3 //A13
data8 0x3F323EF5D8330339 //A14
data8 0xBF2641132EA571F7 //A15
data8 0x3F1B5D9576175CA9 //A16
data8 0xBF10F56A689C623D //A17
data8 0x3F04CACA9141A18D //A18
data8 0xBEFA307AC9B4E85D //A19
data8 0x3EF4B625939FBE32 //A20
data8 0xBECEE6AC1420F86F //A21
data8 0xBE9A95AE2E485964 //A22
data8 0xBF039EF47F8C09BB //A23
data8 0xBF05345957F7B7A9 //A24
data8 0xBEF85AE6385D4CCC //A25
// Polynomial coefficients for the lgammal(x), 3.0 <= |x| < 4.0
data8 0xBFCA4D55BEAB2D6F, 0xBC667B20FF46C6A8 //A3
data8 0x3FE62E42FEFA39EF, 0x3C7ABC9E3B398012 //A0
data8 0x3FFD8773039049E7, 0x3C66CB9070238D77 //A1
data8 0x3FE94699894C1F4C, 0x3C91873D8839B1CD //A2
data8 0xA264558FB0906D7E, 0x00003FFB //A4
data8 0x94D6C50FEB8AFD72, 0x0000BFFA //A5
data8 0x9620656185B68F14, 0x00003FF9 //A6
data8 0xA0D0983B34B7088A, 0x0000BFF8 //A7
data8 0xB36AF863964AA440, 0x00003FF7 //A8
data8 0xCE0DAAFB5497AFB8, 0x0000BFF6 //A9
data8 0xF1C2EAFA79CC2864, 0x00003FF5 //A10
data8 0x9028922A839572B8, 0x0000BFF5 //A11
data8 0xAE1E62F870BA0278, 0x00003FF4 //A12
data8 0xD4726F681E2ABA29, 0x0000BFF3 //A13
data8 0x3F30559B9A02FADF //A14
data8 0xBF243ADEB1266CAE //A15
data8 0x3F19303B6F552603 //A16
data8 0xBF0F768C288EC643 //A17
data8 0x3F039D5356C21DE1 //A18
data8 0xBEF81BCA8168E6BE //A19
data8 0x3EEC74A53A06AD54 //A20
data8 0xBEDED52D1A5DACDF //A21
data8 0x3ECCB4C2C7087342 //A22
data8 0xBEB4F1FAFDFF5C2F //A23
data8 0x3E94C80B52D58904 //A24
data8 0xBE64A328CBE92A27 //A25
LOCAL_OBJECT_END(lgammal_03Q_6_data)

LOCAL_OBJECT_START(lgammal_1pEps_data)
// Polynomial coefficients for the lgammal(x), 1 - 2^(-7) <= |x| < 1 + 2^(-7)
data8 0x93C467E37DB0C7A5, 0x00003FFE //A1
data8 0xD28D3312983E9919, 0x00003FFE //A2
data8 0xCD26AADF559A47E3, 0x00003FFD //A3
data8 0x8A8991563EC22E81, 0x00003FFD //A4
data8 0x3FCA8B9C168D52FE //A5
data8 0x3FC5B40CB0696370 //A6
data8 0x3FC270AC2229A65D //A7
data8 0x3FC0110AF10FCBFC //A8
// Polynomial coefficients for the log1p(x), - 2^(-7) <= |x| <  2^(-7)
data8 0x3FBC71C71C71C71C //P8
data8 0xBFC0000000000000 //P7
data8 0x3FC2492492492492 //P6
data8 0xBFC5555555555555 //P5
data8 0x3FC999999999999A //P4
data8 0xBFD0000000000000 //P3
data8 0x3FD5555555555555 //P2
data8 0xBFE0000000000000 //P1
// short version of "lnsin" polynomial
data8 0xD28D3312983E9918, 0x00003FFF //A2
data8 0x8A8991563EC241B6, 0x00003FFE //A4
data8 0xADA06588061830A5, 0x00003FFD //A6
data8 0x80859B57C31CB746, 0x00003FFD //A8
LOCAL_OBJECT_END(lgammal_1pEps_data)

LOCAL_OBJECT_START(lgammal_neg2andHalf_data)
// Polynomial coefficients for the lgammal(x), -2.005859375 <= x < -2.5
data8 0xBF927781D4BB093A, 0xBC511D86D85B7045 // A3, A0L
data8 0x3FF1A68793DEFC15, 0x3C9852AE2DA7DEEF // A1, A1L
data8 0x408555562D45FAFD, 0xBF972CDAFE5FEFAD // D0, D1
data8 0xC18682331EF492A5, 0xC1845E3E0D29606B // C20, C21
data8 0x4013141822E16979, 0x3CCF8718B6E75F6C // A2, A2L
data8 0xBFACCBF9F5ED0F15, 0xBBDD1AEB73297401 // A0, A3L
data8 0xCCCDB17423046445, 0x00004006 // E6
data8 0x800514E230A3A452, 0x00004005 // E4
data8 0xAAE9A48EC162E76F, 0x00004003 // E2
data8 0x81D4F88B3F3EA0FC, 0x00004002 // E0
data8 0x40CF3F3E35238DA0, 0xC0F8B340945F1A7E // D6, D7
data8 0x40BF89EC0BD609C6, 0xC095897242AEFEE2 // D4, D5
data8 0x40A2482FF01DBC5C, 0xC02095E275FDCF62 // D2, D3
data8 0xC1641354F2312A6A, 0xC17B3657F85258E9 // C18, C19
data8 0xC11F964E9ECBE2C9, 0xC146D7A90F70696C // C16, C17
data8 0xE7AECDE6AF8EA816, 0x0000BFEF // E7
data8 0xD711252FEBBE1091, 0x0000BFEB // E5
data8 0xE648BD10F8C43391, 0x0000BFEF // E3
data8 0x948A1E78AA00A98D, 0x0000BFF4 // E1
LOCAL_OBJECT_END(lgammal_neg2andHalf_data)

LOCAL_OBJECT_START(lgammal_near_neg_half_data)
// Polynomial coefficients for the lgammal(x), -0.5 < x < -0.40625
data8 0xBFC1AE55B180726C, 0x3C8053CD734E6A1D // A3, A0L
data8 0x3FA2AED059BD608A, 0x3C0CD3D2CDBA17F4 // A1, A1L
data8 0x40855554DBCD1E1E, 0x3F96C51AC2BEE9E1 // D0, D1
data8 0xC18682331EF4927D, 0x41845E3E0D295DFC // C20, C21
data8 0x4011DE9E64DF22EF, 0x3CA692B70DAD6B7B // A2, A2L
data8 0x3FF43F89A3F0EDD6, 0xBC4955AED0FA087D // A0, A3L
data8 0xCCCD3F1DF4A2C1DD, 0x00004006 // E6
data8 0x80028ADE33C7FCD9, 0x00004005 // E4
data8 0xAACA474E485507EF, 0x00004003 // E2
data8 0x80F07C206D6B0ECD, 0x00004002 // E0
data8 0x40CF3F3E33E83056, 0x40F8B340944633D9 // D6, D7
data8 0x40BF89EC059931F0, 0x409589723307AD20 // D4, D5
data8 0x40A2482FD0054824, 0x402095CE7F19D011 // D2, D3
data8 0xC1641354F2313614, 0x417B3657F8525354 // C18, C19
data8 0xC11F964E9ECFD21C, 0x4146D7A90F701836 // C16, C17
data8 0x86A9C01F0EA11E5A, 0x0000BFF5 // E7
data8 0xBF6D8469142881C0, 0x0000BFF6 // E5
data8 0x8D45D277BA8255F1, 0x0000BFF8 // E3
data8 0xED2CEA9BA528BCC3, 0x0000BFF9 // E1
LOCAL_OBJECT_END(lgammal_near_neg_half_data)

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
////////////// POLYNOMIAL COEFFICIENTS FOR "NEAR ROOTS" RANGES    /////////////
////////////// THIS PART OF TABLE SHOULD BE ADDRESSED REALLY RARE /////////////
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
LOCAL_OBJECT_START(lgammal_right_roots_polynomial_data)
// Polynomial coefficients for right root on [-3, -2]
// Lgammal is approximated by polynomial within [-.056244 ; .158208 ] range
data8 0xBBBD5E9DCD11030B, 0xB867411D9FF87DD4 //A0
data8 0x3FF83FE966AF535E, 0x3CAA21235B8A769A //A1
data8 0x40136EEBB002F55C, 0x3CC3959A6029838E //A2
data8 0xB4A5302C53C2BEDD, 0x00003FFF //A3
data8 0x8B8C6BE504F2DA1C, 0x00004002 //A4
data8 0xB99CFF02593B4D98, 0x00004001 //A5
data8 0x4038D32F682AA1CF //A6
data8 0x403809F04EE6C5B5 //A7
data8 0x40548EAA81634CEE //A8
data8 0x4059297ADB6BC03D //A9
data8 0x407286FB8EC5C9DA //A10
data8 0x407A92E05B744CFB //A11
data8 0x4091A9D4144258CD //A12
data8 0x409C4D01D24F367E //A13
data8 0x40B1871B9A426A83 //A14
data8 0x40BE51C48BD9A583 //A15
data8 0x40D2140D0C6153E7 //A16
data8 0x40E0FB2C989CE4A3 //A17
data8 0x40E52739AB005641 //A18
data8 0x41161E3E6DDF503A //A19
// Polynomial coefficients for right root on [-4, -3]
// Lgammal is approximated by polynomial within [-.172797 ; .171573 ] range
data8 0x3C172712B248E42E, 0x38CB8D17801A5D67 //A0
data8 0x401F20A65F2FAC54, 0x3CCB9EA1817A824E //A1
data8 0x4039D4D2977150EF, 0x3CDA42E149B6276A //A2
data8 0xE089B8926AE2D9CB, 0x00004005 //A3
data8 0x933901EBBB586C37, 0x00004008 //A4
data8 0xCCD319BED1CFA1CD, 0x0000400A //A5
data8 0x40D293C3F78D3C37 //A6
data8 0x40FBB97AA0B6DD02 //A7
data8 0x41251EA3345E5EB9 //A8
data8 0x415057F65C92E7B0 //A9
data8 0x41799C865241B505 //A10
data8 0x41A445209EFE896B //A11
data8 0x41D02D21880C953B //A12
data8 0x41F9FFDE8C63E16D //A13
data8 0x422504DC8302D2BE //A14
data8 0x425111BF18C95414 //A15
data8 0x427BCBE74A2B8EF7 //A16
data8 0x42A7256F59B286F7 //A17
data8 0x42D462D1586DE61F //A18
data8 0x42FBB1228D6C5118 //A19
// Polynomial coefficients for right root on [-5, -4]
// Lgammal is approximated by polynomial within [-.163171 ; .161988 ] range
data8 0x3C5840FBAFDEE5BB, 0x38CAC0336E8C490A //A0
data8 0x403ACA5CF4921642, 0x3CCEDCDDA5491E56 //A1
data8 0x40744415CD813F8E, 0x3CFBFEBC17E39146 //A2
data8 0xAACD88D954E3E1BD, 0x0000400B //A3
data8 0xCB68C710D75ED802, 0x0000400F //A4
data8 0x8130F5AB997277AC, 0x00004014 //A5
data8 0x41855E3DBF99EBA7 //A6
data8 0x41CD14FE49C49FC2 //A7
data8 0x421433DCE281F07D //A8
data8 0x425C8399C7A92B6F //A9
data8 0x42A45FBE67840F1A //A10
data8 0x42ED68D75F9E6C98 //A11
data8 0x433567291C27E5BE //A12
data8 0x437F5ED7A9D9FD28 //A13
data8 0x43C720A65C8AB711 //A14
data8 0x441120A6C1D40B9B //A15
data8 0x44596F561F2D1CBE //A16
data8 0x44A3507DA81D5C01 //A17
data8 0x44EF06A31E39EEDF //A18
data8 0x45333774C99F523F //A19
// Polynomial coefficients for right root on [-6, -5]
// Lgammal is approximated by polynomial within [-.156450 ; .156126 ] range
data8 0x3C71B82D6B2B3304, 0x3917186E3C0DC231 //A0
data8 0x405ED72E0829AE02, 0x3C960C25157980EB //A1
data8 0x40BCECC32EC22F9B, 0x3D5D8335A32F019C //A2
data8 0x929EC2B1FB931F17, 0x00004012 //A3
data8 0xD112EF96D37316DE, 0x00004018 //A4
data8 0x9F00BB9BB13416AB, 0x0000401F //A5
data8 0x425F7D8D5BDCB223 //A6
data8 0x42C9A8D00C776CC6 //A7
data8 0x433557FD8C481424 //A8
data8 0x43A209221A953EF0 //A9
data8 0x440EDC98D5618AB7 //A10
data8 0x447AABD25E367378 //A11
data8 0x44E73DE20CC3B288 //A12
data8 0x455465257B4E0BD8 //A13
data8 0x45C2011532085353 //A14
data8 0x462FEE4CC191945B //A15
data8 0x469C63AEEFEF0A7F //A16
data8 0x4709D045390A3810 //A17
data8 0x4778D360873C9F64 //A18
data8 0x47E26965BE9A682A //A19
// Polynomial coefficients for right root on [-7, -6]
// Lgammal is approximated by polynomial within [-.154582 ; .154521 ] range
data8 0x3C75F103A1B00A48, 0x391C041C190C726D //A0
data8 0x40869DE49E3AF2AA, 0x3D1C17E1F813063B //A1
data8 0x410FCE23484CFD10, 0x3DB6F38C2F11DAB9 //A2
data8 0xEF281D1E1BE2055A, 0x00004019 //A3
data8 0xFCE3DA92AC55DFF8, 0x00004022 //A4
data8 0x8E9EA838A20BD58E, 0x0000402C //A5
data8 0x4354F21E2FB9E0C9 //A6
data8 0x43E9500994CD4F09 //A7
data8 0x447F3A2C23C033DF //A8
data8 0x45139152656606D8 //A9
data8 0x45A8D45F8D3BF2E8 //A10
data8 0x463FD32110E5BFE5 //A11
data8 0x46D490B3BDBAE0BE //A12
data8 0x476AC3CAD905DD23 //A13
data8 0x48018558217AD473 //A14
data8 0x48970AF371D30585 //A15
data8 0x492E6273A8BEFFE3 //A16
data8 0x49C47CC9AE3F1073 //A17
data8 0x4A5D38E8C35EFF45 //A18
data8 0x4AF0123E89694CD8 //A19
// Polynomial coefficients for right root on [-8, -7]
// Lgammal is approximated by polynomial within [-.154217 ; .154208 ] range
data8 0xBCD2507D818DDD68, 0xB97F6940EA2871A0 //A0
data8 0x40B3B407AA387BCB, 0x3D6320238F2C43D1 //A1
data8 0x41683E85DAAFBAC7, 0x3E148D085958EA3A //A2
data8 0x9F2A95AF1E10A548, 0x00004022 //A3
data8 0x92F21522F482300E, 0x0000402E //A4
data8 0x90B51AB03A1F244D, 0x0000403A //A5
data8 0x44628E1C70EF534F //A6
data8 0x452393E2BC32D244 //A7
data8 0x45E5164141F4BA0B //A8
data8 0x46A712B3A8AF5808 //A9
data8 0x47698FD36CEDD0F2 //A10
data8 0x482C9AE6BBAA3637 //A11
data8 0x48F023821857C8E9 //A12
data8 0x49B2569053FC106F //A13
data8 0x4A74F646D5C1604B //A14
data8 0x4B3811CF5ABA4934 //A15
data8 0x4BFBB5DD6C84E233 //A16
data8 0x4CC05021086F637B //A17
data8 0x4D8450A345B0FB49 //A18
data8 0x4E43825848865DB2 //A19
// Polynomial coefficients for right root on [-9, -8]
// Lgammal is approximated by polynomial within [-.154160 ; .154158 ] range
data8 0x3CDF4358564F2B46, 0x397969BEE6042F81 //A0
data8 0x40E3B088FED67721, 0x3D82787BA937EE85 //A1
data8 0x41C83A3893550EF4, 0x3E542ED57E244DA8 //A2
data8 0x9F003C6DC56E0B8E, 0x0000402B //A3
data8 0x92BDF64A3213A699, 0x0000403A //A4
data8 0x9074F503AAD417AF, 0x00004049 //A5
data8 0x4582843E1313C8CD //A6
data8 0x467387BD6A7826C1 //A7
data8 0x4765074E788CF440 //A8
data8 0x4857004DD9D1E09D //A9
data8 0x4949792ED7530EAF //A10
data8 0x4A3C7F089A292ED3 //A11
data8 0x4B30125BF0AABB86 //A12
data8 0x4C224175195E307E //A13
data8 0x4D14DC4C8B32C08D //A14
data8 0x4E07F1DB2786197E //A15
data8 0x4EFB8EA1C336DACB //A16
data8 0x4FF03797EACD0F23 //A17
data8 0x50E4304A8E68A730 //A18
data8 0x51D3618FB2EC9F93 //A19
// Polynomial coefficients for right root on [-10, -9]
// Lgammal is approximated by polynomial within [-.154152 ; .154152 ] range
data8 0x3D42F34DA97ECF0C, 0x39FD1256F345B0D0 //A0
data8 0x4116261203919787, 0x3DC12D44055588EB //A1
data8 0x422EA8F32FB7FE99, 0x3ED849CE4E7B2D77 //A2
data8 0xE25BAF73477A57B5, 0x00004034 //A3
data8 0xEB021FD10060504A, 0x00004046 //A4
data8 0x8220A208EE206C5F, 0x00004059 //A5
data8 0x46B2C3903EC9DA14 //A6
data8 0x47D64393744B9C67 //A7
data8 0x48FAF79CCDC604DD //A8
data8 0x4A20975DB8061EBA //A9
data8 0x4B44AB9CBB38DB21 //A10
data8 0x4C6A032F60094FE9 //A11
data8 0x4D908103927634B4 //A12
data8 0x4EB516CA21D30861 //A13
data8 0x4FDB1BF12C58D318 //A14
data8 0x510180AAE094A553 //A15
data8 0x5226A8F2A2D45D57 //A16
data8 0x534E00B6B0C8B809 //A17
data8 0x5475022FE21215B2 //A18
data8 0x5596B02BF6C5E19B //A19
// Polynomial coefficients for right root on [-11, -10]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0x3D7AA9C2E2B1029C, 0x3A15FB37578544DB //A0
data8 0x414BAF825A0C91D4, 0x3DFB9DA2CE398747 //A1
data8 0x4297F3EC8AE0AF03, 0x3F34208B55FB8781 //A2
data8 0xDD0C97D3197F56DE, 0x0000403E //A3
data8 0x8F6F3AF7A5499674, 0x00004054 //A4
data8 0xC68DA1AF6D878EEB, 0x00004069 //A5
data8 0x47F1E4E1E2197CE0 //A6
data8 0x494A8A28E597C3EB //A7
data8 0x4AA4175D0D35D705 //A8
data8 0x4BFEE6F0AF69E814 //A9
data8 0x4D580FE7B3DBB3C6 //A10
data8 0x4EB2ECE60E4608AF //A11
data8 0x500E04BE3E2B4F24 //A12
data8 0x5167F9450F0FB8FD //A13
data8 0x52C342BDE747603F //A14
data8 0x541F1699D557268C //A15
data8 0x557927C5F079864E //A16
data8 0x56D4D10FEEDB030C //A17
data8 0x5832385DF86AD28A //A18
data8 0x598898914B4D6523 //A19
// Polynomial coefficients for right root on [-12, -11]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0xBD96F61647C58B03, 0xBA3ABB0C2A6C755B //A0
data8 0x418308A82714B70D, 0x3E1088FC6A104C39 //A1
data8 0x4306A493DD613C39, 0x3FB2341ECBF85741 //A2
data8 0x8FA8FE98339474AB, 0x00004049 //A3
data8 0x802CCDF570BA7942, 0x00004062 //A4
data8 0xF3F748AF11A32890, 0x0000407A //A5
data8 0x493E3B567EF178CF //A6
data8 0x4ACED38F651BA362 //A7
data8 0x4C600B357337F946 //A8
data8 0x4DF0F71A52B54CCF //A9
data8 0x4F8229F3B9FA2C70 //A10
data8 0x5113A4C4979B770E //A11
data8 0x52A56BC367F298D5 //A12
data8 0x543785CF31842DC0 //A13
data8 0x55C9FC37E3E40896 //A14
data8 0x575CD5D1BA556C82 //A15
data8 0x58F00A7AD99A9E08 //A16
data8 0x5A824088688B008D //A17
data8 0x5C15F75EF7E08EBD //A18
data8 0x5DA462EA902F0C90 //A19
// Polynomial coefficients for right root on [-13, -12]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0x3DC3191752ACFC9D, 0x3A26CB6629532DBF //A0
data8 0x41BC8CFC051191BD, 0x3E68A84DA4E62AF2 //A1
data8 0x43797926294A0148, 0x400F345FF3723CFF //A2
data8 0xF26D2AF700B82625, 0x00004053 //A3
data8 0xA238B24A4B1F7B15, 0x00004070 //A4
data8 0xE793B5C0A41A264F, 0x0000408C //A5
data8 0x4A9585BDDACE863D //A6
data8 0x4C6075953448088A //A7
data8 0x4E29B2F38D1FC670 //A8
data8 0x4FF4619B079C440F //A9
data8 0x51C05DAE118D8AD9 //A10
data8 0x538A8C7F87326AD4 //A11
data8 0x5555B6937588DAB3 //A12
data8 0x5721E1F8B6E6A7DB //A13
data8 0x58EDA1D7A77DD6E5 //A14
data8 0x5AB8A9616B7DC9ED //A15
data8 0x5C84942AA209ED17 //A16
data8 0x5E518FC34C6F54EF //A17
data8 0x601FB3F17BCCD9A0 //A18
data8 0x61E61128D512FE97 //A1
// Polynomial coefficients for right root on [-14, -13]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0xBE170D646421B3F5, 0xBAAD95F79FCB5097 //A0
data8 0x41F7328CBFCD9AC7, 0x3E743B8B1E8AEDB1 //A1
data8 0x43F0D0FA2DBDA237, 0x40A0422D6A227B55 //A2
data8 0x82082DF2D32686CC, 0x0000405F //A3
data8 0x8D64EE9B42E68B43, 0x0000407F //A4
data8 0xA3FFD82E08C5F1F1, 0x0000409F //A5
data8 0x4BF8C49D99123454 //A6
data8 0x4DFEC79DDF11342F //A7
data8 0x50038615A892F6BD //A8
data8 0x520929453DB32EF1 //A9
data8 0x54106A7808189A7F //A10
data8 0x5615A302D03C207B //A11
data8 0x581CC175AA736F5E //A12
data8 0x5A233E071147C017 //A13
data8 0x5C29E81917243F22 //A14
data8 0x5E3184B0B5AC4707 //A15
data8 0x6037C11DE62D8388 //A16
data8 0x6240787C4B1C9D6C //A17
data8 0x6448289235E80977 //A18
data8 0x664B5352C6C3449E //A19
// Polynomial coefficients for right root on [-15, -14]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0x3E562C2E34A9207D, 0x3ADC00DA3DFF7A83 //A0
data8 0x42344C3B2F0D90AB, 0x3EB8A2E979F24536 //A1
data8 0x4469BFFF28B50D07, 0x41181E3D05C1C294 //A2
data8 0xAE38F64DCB24D9F8, 0x0000406A //A3
data8 0xA5C3F52C1B350702, 0x0000408E //A4
data8 0xA83BC857BCD67A1B, 0x000040B2 //A5
data8 0x4D663B4727B4D80A //A6
data8 0x4FA82C965B0F7788 //A7
data8 0x51EAD58C02908D95 //A8
data8 0x542E427970E073D8 //A9
data8 0x56714644C558A818 //A10
data8 0x58B3EC2040C77BAE //A11
data8 0x5AF72AE6A83D45B1 //A12
data8 0x5D3B214F611F5D12 //A13
data8 0x5F7FF5E49C54E92A //A14
data8 0x61C2E917AB765FB2 //A15
data8 0x64066FD70907B4C1 //A16
data8 0x664B3998D60D0F9B //A17
data8 0x689178710782FA8B //A18
data8 0x6AD14A66C1C7BEC3 //A19
// Polynomial coefficients for right root on [-16, -15]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0xBE6D7E7192615BAE, 0xBB0137677D7CC719 //A0
data8 0x4273077763F6628C, 0x3F09250FB8FC8EC9 //A1
data8 0x44E6A1BF095B1AB3, 0x4178D5A74F6CB3B3 //A2
data8 0x8F8E0D5060FCC76E, 0x00004076 //A3
data8 0x800CC1DCFF092A63, 0x0000409E //A4
data8 0xF3AB0BA9D14CDA72, 0x000040C5 //A5
data8 0x4EDE3000A2F6D54F //A6
data8 0x515EC613B9C8E241 //A7
data8 0x53E003309FEEEA96 //A8
data8 0x5660ED908D7C9A90 //A9
data8 0x58E21E9B517B1A50 //A10
data8 0x5B639745E4374EE2 //A11
data8 0x5DE55BB626B2075D //A12
data8 0x606772B7506BA747 //A13
data8 0x62E9E581AB2E057B //A14
data8 0x656CBAD1CF85D396 //A15
data8 0x67EFF4EBD7989872 //A16
data8 0x6A722D2B19B7E2F9 //A17
data8 0x6CF5DEB3073B0743 //A18
data8 0x6F744AC11550B93A //A19
// Polynomial coefficients for right root on [-17, -16]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0xBEDCC6291188207E, 0xBB872E3FDD48F5B7 //A0
data8 0x42B3076EE7525EF9, 0x3F6687A5038CA81C //A1
data8 0x4566A1AAD96EBCB5, 0x421F0FEDFBF548D2 //A2
data8 0x8F8D4D3DE9850DBA, 0x00004082 //A3
data8 0x800BDD6DA2CE1859, 0x000040AE //A4
data8 0xF3A8EC4C9CDC1CE5, 0x000040D9 //A5
data8 0x505E2FAFDB812628 //A6
data8 0x531EC5B3A7508719 //A7
data8 0x55E002F77E99B628 //A8
data8 0x58A0ED4C9B4DAE54 //A9
data8 0x5B621E4A8240F90C //A10
data8 0x5E2396E5C8849814 //A11
data8 0x60E55B43D8C5CE71 //A12
data8 0x63A7722F5D45D01D //A13
data8 0x6669E4E010DCE45A //A14
data8 0x692CBA120D5E78F6 //A15
data8 0x6BEFF4045350B22E //A16
data8 0x6EB22C9807C21819 //A17
data8 0x7175DE20D04617C4 //A18
data8 0x74344AB87C6D655F //A19
// Polynomial coefficients for right root on [-18, -17]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0xBF28AEEE7B61D77C, 0xBBDBBB5FC57ABF79 //A0
data8 0x42F436F56B3B8A0C, 0x3FA43EE3C5C576E9 //A1
data8 0x45E98A22535D115D, 0x42984678BE78CC48 //A2
data8 0xAC176F3775E6FCFC, 0x0000408E //A3
data8 0xA3114F53A9FEB922, 0x000040BE //A4
data8 0xA4D168A8334ABF41, 0x000040EE //A5
data8 0x51E5B0E7EC7182BB //A6
data8 0x54E77D67B876EAB6 //A7
data8 0x57E9F7C30C09C4B6 //A8
data8 0x5AED29B0488614CA //A9
data8 0x5DF09486F87E79F9 //A10
data8 0x60F30B199979654E //A11
data8 0x63F60E02C7DCCC5F //A12
data8 0x66F9B8A00EB01684 //A13
data8 0x69FE2D3ED0700044 //A14
data8 0x6D01C8363C7DCC84 //A15
data8 0x700502B29C2F06E3 //A16
data8 0x730962B4500F4A61 //A17
data8 0x76103C6ED099192A //A18
data8 0x79100C7132CFD6E3 //A19
// Polynomial coefficients for right root on [-19, -18]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0x3F3C19A53328A0C3, 0x3BE04ADC3FBE1458 //A0
data8 0x4336C16C16C16C19, 0x3FE58CE3AC4A7C28 //A1
data8 0x46702E85C0898B70, 0x432C922E412CEC6E //A2
data8 0xF57B99A1C034335D, 0x0000409A //A3
data8 0x82EC9634223DF909, 0x000040CF //A4
data8 0x94F66D7557E2EA60, 0x00004103 //A5
data8 0x5376118B79AE34D0 //A6
data8 0x56BAE7106D52E548 //A7
data8 0x5A00BD48CC8E25AB //A8
data8 0x5D4529722821B493 //A9
data8 0x608B1654AF31BBC1 //A10
data8 0x63D182CC98AEA859 //A11
data8 0x6716D43D5EEB05E8 //A12
data8 0x6A5DF884FC172E1C //A13
data8 0x6DA3CA7EBB97976B //A14
data8 0x70EA416D0BE6D2EF //A15
data8 0x743176C31EBB65F2 //A16
data8 0x7777C401A8715CF9 //A17
data8 0x7AC1110C6D350440 //A18
data8 0x7E02D0971CF84865 //A19
// Polynomial coefficients for right root on [-20, -19]
// Lgammal is approximated by polynomial within [-.154151 ; .154151 ] range
data8 0xBFAB767F9BE21803, 0xBC5ACEF5BB1BD8B5 //A0
data8 0x4379999999999999, 0x4029241C7F5914C8 //A1
data8 0x46F47AE147AE147A, 0x43AC2979B64B9D7E //A2
data8 0xAEC33E1F67152993, 0x000040A7 //A3
data8 0xD1B71758E219616F, 0x000040DF //A4
data8 0x8637BD05AF6CF468, 0x00004118 //A5
data8 0x55065E9F80F293DE //A6
data8 0x588EADA78C44EE66 //A7
data8 0x5C15798EE22DEF09 //A8
data8 0x5F9E8ABFD644FA63 //A9
data8 0x6325FD7FE29BD7CD //A10
data8 0x66AFFC5C57E1F802 //A11
data8 0x6A3774CD7D5C0181 //A12
data8 0x6DC152724DE2A6FE //A13
data8 0x7149BB138EB3D0C2 //A14
data8 0x74D32FF8A70896C2 //A15
data8 0x785D3749F9C72BD7 //A16
data8 0x7BE5CCF65EBC4E40 //A17
data8 0x7F641A891B5FC652 //A18
data8 0x7FEFFFFFFFFFFFFF //A19
LOCAL_OBJECT_END(lgammal_right_roots_polynomial_data)

LOCAL_OBJECT_START(lgammal_left_roots_polynomial_data)
// Polynomial coefficients for left root on [-3, -2]
// Lgammal is approximated by polynomial within [.084641 ; -.059553 ] range
data8 0xBC0844590979B82E, 0xB8BC7CE8CE2ECC3B //A0
data8 0xBFFEA12DA904B18C, 0xBC91A6B2BAD5EF6E //A1
data8 0x4023267F3C265A51, 0x3CD7055481D03AED //A2
data8 0xA0C2D618645F8E00, 0x0000C003 //A3
data8 0xFA8256664F8CD2BE, 0x00004004 //A4
data8 0xC2C422C103F57158, 0x0000C006 //A5
data8 0x4084373F7CC70AF5 //A6
data8 0xC0A12239BDD6BB95 //A7
data8 0x40BDBA65E2709397 //A8
data8 0xC0DA2D2504DFB085 //A9
data8 0x40F758173CA5BF3C //A10
data8 0xC11506C65C267E72 //A11
data8 0x413318EE3A6B05FC //A12
data8 0xC1517767F247DA98 //A13
data8 0x41701237B4754D73 //A14
data8 0xC18DB8A03BC5C3D8 //A15
data8 0x41AB80953AC14A07 //A16
data8 0xC1C9B7B76638D0A4 //A17
data8 0x41EA727E3033E2D9 //A18
data8 0xC20812C297729142 //A19
//
// Polynomial coefficients for left root on [-4, -3]
// Lgammal is approximated by polynomial within [.147147 ; -.145158 ] range
data8 0xBC3130AE5C4F54DB, 0xB8ED23294C13398A //A0
data8 0xC034B99D966C5646, 0xBCE2E5FE3BC3DBB9 //A1
data8 0x406F76DEAE0436BD, 0x3D14974DDEC057BD //A2
data8 0xE929ACEA5979BE96, 0x0000C00A //A3
data8 0xF47C14F8A0D52771, 0x0000400E //A4
data8 0x88B7BC036937481C, 0x0000C013 //A5
data8 0x4173E8F3AB9FC266 //A6
data8 0xC1B7DBBE062FB11B //A7
data8 0x41FD2F76DE7A47A7 //A8
data8 0xC242225FE53B124D //A9
data8 0x4286D12AE2FBFA30 //A10
data8 0xC2CCFFC267A3C4C0 //A11
data8 0x431294E10008E014 //A12
data8 0xC357FAC8C9A2DF6A //A13
data8 0x439F2190AB9FAE01 //A14
data8 0xC3E44C1D8E8C67C3 //A15
data8 0x442A8901105D5A38 //A16
data8 0xC471C4421E908C3A //A17
data8 0x44B92CD4D59D6D17 //A18
data8 0xC4FB3A078B5247FA //A19
// Polynomial coefficients for left root on [-5, -4]
// Lgammal is approximated by polynomial within [.155671 ; -.155300 ] range
data8 0xBC57BF3C6E8A94C1, 0xB902FB666934AC9E //A0
data8 0xC05D224A3EF9E41F, 0xBCF6F5713913E440 //A1
data8 0x40BB533C678A3955, 0x3D688E53E3C72538 //A2
data8 0x869FBFF732E99B84, 0x0000C012 //A3
data8 0xBA9537AD61392DEC, 0x00004018 //A4
data8 0x89EAE8B1DEA06B05, 0x0000C01F //A5
data8 0x425A8C5C53458D3C //A6
data8 0xC2C5068B3ED6509B //A7
data8 0x4330FFA575E99B4E //A8
data8 0xC39BEC12DDDF7669 //A9
data8 0x44073825725F74F9 //A10
data8 0xC47380EBCA299047 //A11
data8 0x44E084DD9B666437 //A12
data8 0xC54C2DA6BF787ACF //A13
data8 0x45B82D65C8D6FA42 //A14
data8 0xC624D62113FE950A //A15
data8 0x469200CC19B45016 //A16
data8 0xC6FFDDC6DD938E2E //A17
data8 0x476DD7C07184B9F9 //A18
data8 0xC7D554A30085C052 //A19
// Polynomial coefficients for left root on [-6, -5]
// Lgammal is approximated by polynomial within [.157425 ; -.157360 ] range
data8 0x3C9E20A87C8B79F1, 0x39488BE34B2427DB //A0
data8 0xC08661F6A43A5E12, 0xBD3D912526D759CC //A1
data8 0x410F79DCB794F270, 0x3DB9BEE7CD3C1BF5 //A2
data8 0xEB7404450D0005DB, 0x0000C019 //A3
data8 0xF7AE9846DFE4D4AB, 0x00004022 //A4
data8 0x8AF535855A95B6DA, 0x0000C02C //A5
data8 0x43544D54E9FE240E //A6
data8 0xC3E8684E40CE6CFC //A7
data8 0x447DF44C1D803454 //A8
data8 0xC512AC305439B2BA //A9
data8 0x45A79226AF79211A //A10
data8 0xC63E0DFF7244893A //A11
data8 0x46D35216C3A83AF3 //A12
data8 0xC76903BE0C390E28 //A13
data8 0x48004A4DECFA4FD5 //A14
data8 0xC8954FBD243DB8BE //A15
data8 0x492BF3A31EB18DDA //A16
data8 0xC9C2C6A864521F3A //A17
data8 0x4A5AB127C62E8DA1 //A18
data8 0xCAECF60EF3183C57 //A19
// Polynomial coefficients for left root on [-7, -6]
// Lgammal is approximated by polynomial within [.157749 ; -.157739 ] range
data8 0x3CC9B9E8B8D551D6, 0x3961813C8E1E10DB //A0
data8 0xC0B3ABF7A5CEA91F, 0xBD55638D4BCB4CC4 //A1
data8 0x4168349A25504236, 0x3E0287ECE50CCF76 //A2
data8 0x9EC8ED6E4C219E67, 0x0000C022 //A3
data8 0x9279EB1B799A3FF3, 0x0000402E //A4
data8 0x90213EF8D9A5DBCF, 0x0000C03A //A5
data8 0x4462775E857FB71C //A6
data8 0xC52377E70B45FDBF //A7
data8 0x45E4F3D28EDA8C28 //A8
data8 0xC6A6E85571BD2D0B //A9
data8 0x47695BB17E74DF74 //A10
data8 0xC82C5AC0ED6A662F //A11
data8 0x48EFF8159441C2E3 //A12
data8 0xC9B22602C1B68AE5 //A13
data8 0x4A74BA8CE7B34100 //A14
data8 0xCB37C7E208482E4B //A15
data8 0x4BFB5A1D57352265 //A16
data8 0xCCC01CB3021212FF //A17
data8 0x4D841613AC3431D1 //A18
data8 0xCE431C9E9EE43AD9 //A19
// Polynomial coefficients for left root on [-8, -7]
// Lgammal is approximated by polynomial within [.157799 ; -.157798 ] range
data8 0xBCF9C7A33AD9478C, 0xB995B0470F11E5ED //A0
data8 0xC0E3AF76FE4C2F8B, 0xBD8DBCD503250511 //A1
data8 0x41C838E76CAAF0D5, 0x3E5D79F5E2E069C3 //A2
data8 0x9EF345992B262CE0, 0x0000C02B //A3
data8 0x92AE0292985FD559, 0x0000403A //A4
data8 0x90615420C08F7D8C, 0x0000C049 //A5
data8 0x45828139342CEEB7 //A6
data8 0xC67384066C31E2D3 //A7
data8 0x476502BC4DAC2C35 //A8
data8 0xC856FAADFF22ADC6 //A9
data8 0x49497243255AB3CE //A10
data8 0xCA3C768489520F6B //A11
data8 0x4B300D1EA47AF838 //A12
data8 0xCC223B0508AC620E //A13
data8 0x4D14D46583338CD8 //A14
data8 0xCE07E7A87AA068E4 //A15
data8 0x4EFB811AD2F8BEAB //A16
data8 0xCFF0351B51508523 //A17
data8 0x50E4364CCBF53100 //A18
data8 0xD1D33CFD0BF96FA6 //A19
// Polynomial coefficients for left root on [-9, -8]
// Lgammal is approximated by polynomial within [.157806 ; -.157806 ] range
data8 0x3D333E4438B1B9D4, 0x39E7B956B83964C1 //A0
data8 0xC11625EDFC63DCD8, 0xBDCF39625709EFAC //A1
data8 0x422EA8C150480F16, 0x3EC16ED908AB7EDD //A2
data8 0xE2598725E2E11646, 0x0000C034 //A3
data8 0xEAFF2346DE3EBC98, 0x00004046 //A4
data8 0x821E90DE12A0F05F, 0x0000C059 //A5
data8 0x46B2C334AE5366FE //A6
data8 0xC7D64314B43191B6 //A7
data8 0x48FAF6ED5899E01B //A8
data8 0xCA2096E4472AF37D //A9
data8 0x4B44AAF49FB7E4C8 //A10
data8 0xCC6A02469F2BD920 //A11
data8 0x4D9080626D2EFC07 //A12
data8 0xCEB515EDCF0695F7 //A13
data8 0x4FDB1AC69BF36960 //A14
data8 0xD1017F8274339270 //A15
data8 0x5226A684961BAE2F //A16
data8 0xD34E085C088404A5 //A17
data8 0x547511892FF8960E //A18
data8 0xD5968FA3B1ED67A9 //A19
// Polynomial coefficients for left root on [-10, -9]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBD355818A2B42BA2, 0xB9B7320B6A0D61EA //A0
data8 0xC14BAF7DA5F3770E, 0xBDE64AF9A868F719 //A1
data8 0x4297F3E8791F9CD3, 0x3F2A553E59B4835E //A2
data8 0xDD0C5F7E551BD13C, 0x0000C03E //A3
data8 0x8F6F0A3B2EB08BBB, 0x00004054 //A4
data8 0xC68D4D5AD230BA08, 0x0000C069 //A5
data8 0x47F1E4D8C35D1A3E //A6
data8 0xC94A8A191DB0A466 //A7
data8 0x4AA4174F65FE6AE8 //A8
data8 0xCBFEE6D90F94E9DD //A9
data8 0x4D580FD3438BE16C //A10
data8 0xCEB2ECD456D50224 //A11
data8 0x500E049F7FE64546 //A12
data8 0xD167F92D9600F378 //A13
data8 0x52C342AE2B43261A //A14
data8 0xD41F15DEEDA4B67E //A15
data8 0x55792638748AFB7D //A16
data8 0xD6D4D760074F6E6B //A17
data8 0x5832469D58ED3FA9 //A18
data8 0xD988769F3DC76642 //A19
// Polynomial coefficients for left root on [-11, -10]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBDA050601F39778A, 0xBA0D4D1CE53E8241 //A0
data8 0xC18308A7D8EA4039, 0xBE370C379D3EAD41 //A1
data8 0x4306A49380644E6C, 0x3FBBB143C0E7B5C8 //A2
data8 0x8FA8FB233E4AA6D2, 0x0000C049 //A3
data8 0x802CC9D8AEAC207D, 0x00004062 //A4
data8 0xF3F73EE651A37A13, 0x0000C07A //A5
data8 0x493E3B550A7B9568 //A6
data8 0xCACED38DAA060929 //A7
data8 0x4C600B346BAB3BC6 //A8
data8 0xCDF0F719193E3D26 //A9
data8 0x4F8229F24528B151 //A10
data8 0xD113A4C2D32FBBE2 //A11
data8 0x52A56BC13DC4474D //A12
data8 0xD43785CFAF5E3CE3 //A13
data8 0x55C9FC3EA5941202 //A14
data8 0xD75CD545A3341AF5 //A15
data8 0x58F009911F77C282 //A16
data8 0xDA8246294D210BEC //A17
data8 0x5C1608AAC32C3A8E //A18
data8 0xDDA446E570A397D5 //A19
// Polynomial coefficients for left root on [-12, -11]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0x3DEACBB3081C502E, 0x3A8AA6F01DEDF745 //A0
data8 0xC1BC8CFBFB0A9912, 0xBE6556B6504A2AE6 //A1
data8 0x43797926206941D7, 0x40289A9644C2A216 //A2
data8 0xF26D2A78446D0839, 0x0000C053 //A3
data8 0xA238B1D937FFED38, 0x00004070 //A4
data8 0xE793B4F6DE470538, 0x0000C08C //A5
data8 0x4A9585BDC44DC45D //A6
data8 0xCC60759520342C47 //A7
data8 0x4E29B2F3694C0404 //A8
data8 0xCFF4619AE7B6BBAB //A9
data8 0x51C05DADF52B89E8 //A10
data8 0xD38A8C7F48819A4A //A11
data8 0x5555B6932D687860 //A12
data8 0xD721E1FACB6C1B5B //A13
data8 0x58EDA1E2677C8F91 //A14
data8 0xDAB8A8EC523C1F71 //A15
data8 0x5C84930133F30411 //A16
data8 0xDE51952FDFD1EC49 //A17
data8 0x601FCCEC1BBD25F1 //A18
data8 0xE1E5F2D76B610920 //A19
// Polynomial coefficients for left root on [-13, -12]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBE01612F373268ED, 0xBA97B7A18CDF103B //A0
data8 0xC1F7328CBF7A4FAC, 0xBE89A25A6952F481 //A1
data8 0x43F0D0FA2DBDA237, 0x40A0422EC1CE6084 //A2
data8 0x82082DF2D32686C5, 0x0000C05F //A3
data8 0x8D64EE9B42E68B36, 0x0000407F //A4
data8 0xA3FFD82E08C630C9, 0x0000C09F //A5
data8 0x4BF8C49D99123466 //A6
data8 0xCDFEC79DDF1119ED //A7
data8 0x50038615A892D242 //A8
data8 0xD20929453DC8B537 //A9
data8 0x54106A78083BA1EE //A10
data8 0xD615A302C69E27B2 //A11
data8 0x581CC175870FF16F //A12
data8 0xDA233E0979E12B74 //A13
data8 0x5C29E822BC568C80 //A14
data8 0xDE31845DB5340FBC //A15
data8 0x6037BFC6D498D5F9 //A16
data8 0xE2407D92CD613E82 //A17
data8 0x64483B9B62367EB7 //A18
data8 0xE64B2DC830E8A799 //A1
// Polynomial coefficients for left root on [-14, -13]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0x3E563D0B930B371F, 0x3AE779957E14F012 //A0
data8 0xC2344C3B2F083767, 0xBEC0B7769AA3DD66 //A1
data8 0x4469BFFF28B50D07, 0x41181E3F13ED2401 //A2
data8 0xAE38F64DCB24D9EE, 0x0000C06A //A3
data8 0xA5C3F52C1B3506F2, 0x0000408E //A4
data8 0xA83BC857BCD6BA92, 0x0000C0B2 //A5
data8 0x4D663B4727B4D81A //A6
data8 0xCFA82C965B0F62E9 //A7
data8 0x51EAD58C02905B71 //A8
data8 0xD42E427970FA56AD //A9
data8 0x56714644C57D8476 //A10
data8 0xD8B3EC2037EC95F2 //A11
data8 0x5AF72AE68BBA5B3D //A12
data8 0xDD3B2152C67AA6B7 //A13
data8 0x5F7FF5F082861B8B //A14
data8 0xE1C2E8BE125A5B7A //A15
data8 0x64066E92FE9EBE7D //A16
data8 0xE64B4201CDF9F138 //A17
data8 0x689186351E58AA88 //A18
data8 0xEAD132A585DFC60A //A19
// Polynomial coefficients for left root on [-15, -14]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBE6D7DDE12700AC1, 0xBB1E025BF1667FB5 //A0
data8 0xC273077763F60AD5, 0xBF2A1698184C7A9A //A1
data8 0x44E6A1BF095B1AB3, 0x4178D5AE8A4A2874 //A2
data8 0x8F8E0D5060FCC767, 0x0000C076 //A3
data8 0x800CC1DCFF092A57, 0x0000409E //A4
data8 0xF3AB0BA9D14D37D1, 0x0000C0C5 //A5
data8 0x4EDE3000A2F6D565 //A6
data8 0xD15EC613B9C8C800 //A7
data8 0x53E003309FEECCAA //A8
data8 0xD660ED908D8B15C4 //A9
data8 0x58E21E9B51A1C4AE //A10
data8 0xDB639745DB82210D //A11
data8 0x5DE55BB60C68FCF6 //A12
data8 0xE06772BA3FCA23C6 //A13
data8 0x62E9E58B4F702C31 //A14
data8 0xE56CBA49B071ABE2 //A15
data8 0x67EFF31E4F2BA36A //A16
data8 0xEA7232C8804F32C3 //A17
data8 0x6CF5EFEE929A0928 //A18
data8 0xEF742EE03EC3E8FF //A19
// Polynomial coefficients for left root on [-16, -15]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBEDCC628FEAC7A1B, 0xBB80582C8BEBB198 //A0
data8 0xC2B3076EE752595E, 0xBF5388F55AFAE53E //A1
data8 0x4566A1AAD96EBCB5, 0x421F0FEFE2444293 //A2
data8 0x8F8D4D3DE9850DB2, 0x0000C082 //A3
data8 0x800BDD6DA2CE184C, 0x000040AE //A4
data8 0xF3A8EC4C9CDC7A43, 0x0000C0D9 //A5
data8 0x505E2FAFDB81263F //A6
data8 0xD31EC5B3A7506CD9 //A7
data8 0x55E002F77E999810 //A8
data8 0xD8A0ED4C9B5C2900 //A9
data8 0x5B621E4A8267C401 //A10
data8 0xDE2396E5BFCFDA7A //A11
data8 0x60E55B43BE6F9A79 //A12
data8 0xE3A772324C7405FA //A13
data8 0x6669E4E9B7E57A2D //A14
data8 0xE92CB989F8A8FB37 //A15
data8 0x6BEFF2368849A36E //A16
data8 0xEEB23234FE191D55 //A17
data8 0x7175EF5D1080B105 //A18
data8 0xF4342ED7B1B7BE31 //A19
// Polynomial coefficients for left root on [-17, -16]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBF28AEEE7B58C790, 0xBBC4448DE371FA0A //A0
data8 0xC2F436F56B3B89B1, 0xBF636755245AC63A //A1
data8 0x45E98A22535D115D, 0x4298467DA93DB784 //A2
data8 0xAC176F3775E6FCF2, 0x0000C08E //A3
data8 0xA3114F53A9FEB908, 0x000040BE //A4
data8 0xA4D168A8334AFE5A, 0x0000C0EE //A5
data8 0x51E5B0E7EC7182CF //A6
data8 0xD4E77D67B876D6B4 //A7
data8 0x57E9F7C30C098C83 //A8
data8 0xDAED29B0489EF7A7 //A9
data8 0x5DF09486F8A524B8 //A10
data8 0xE0F30B19910A2393 //A11
data8 0x63F60E02AB3109F4 //A12
data8 0xE6F9B8A3431854D5 //A13
data8 0x69FE2D4A6D94218E //A14
data8 0xED01C7E272A73560 //A15
data8 0x7005017D82B186B6 //A16
data8 0xF3096A81A69BD8AE //A17
data8 0x76104951BAD67D5C //A18
data8 0xF90FECC99786FD5B //A19
// Polynomial coefficients for left root on [-18, -17]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0x3F3C19A53328E26A, 0x3BE238D7BA036B3B //A0
data8 0xC336C16C16C16C13, 0xBFEACE245DEC56F3 //A1
data8 0x46702E85C0898B70, 0x432C922B64FD1DA4 //A2
data8 0xF57B99A1C0343350, 0x0000C09A //A3
data8 0x82EC9634223DF90D, 0x000040CF //A4
data8 0x94F66D7557E3237D, 0x0000C103 //A5
data8 0x5376118B79AE34D6 //A6
data8 0xD6BAE7106D52CE49 //A7
data8 0x5A00BD48CC8E11AB //A8
data8 0xDD4529722833E2DF //A9
data8 0x608B1654AF5F46AF //A10
data8 0xE3D182CC90D8723F //A11
data8 0x6716D43D46706AA0 //A12
data8 0xEA5DF888C5B428D3 //A13
data8 0x6DA3CA85888931A6 //A14
data8 0xF0EA40EF2AC7E070 //A15
data8 0x743175D1A251AFCD //A16
data8 0xF777CB6E2B550D73 //A17
data8 0x7AC11E468A134A51 //A18
data8 0xFE02B6BDD0FC40AA //A19
// Polynomial coefficients for left root on [-19, -18]
// Lgammal is approximated by polynomial within [.157807 ; -.157807 ] range
data8 0xBFAB767F9BE217FC, 0xBC4A5541CE0D8D0D //A0
data8 0xC379999999999999, 0xC01A84981B490BE8 //A1
data8 0x46F47AE147AE147A, 0x43AC2987BBC466EB //A2
data8 0xAEC33E1F67152987, 0x0000C0A7 //A3
data8 0xD1B71758E2196153, 0x000040DF //A4
data8 0x8637BD05AF6D420E, 0x0000C118 //A5
data8 0x55065E9F80F293B2 //A6
data8 0xD88EADA78C44BFA7 //A7
data8 0x5C15798EE22EC6CD //A8
data8 0xDF9E8ABFD67895CF //A9
data8 0x6325FD7FE13B0DE0 //A10
data8 0xE6AFFC5C3DE70858 //A11
data8 0x6A3774CE81C70D43 //A12
data8 0xEDC1527412D8129F //A13
data8 0x7149BABCDA8B7A72 //A14
data8 0xF4D330AD49071BB5 //A15
data8 0x785D4046F4C5F1FD //A16
data8 0xFBE59BFEDBA73FAF //A17
data8 0x7F64BEF2B2EC8DA1 //A18
data8 0xFFEFFFFFFFFFFFFF //A19
LOCAL_OBJECT_END(lgammal_left_roots_polynomial_data)


//==============================================================
// Code
//==============================================================

.section .text
GLOBAL_LIBM_ENTRY(__libm_lgammal)
{ .mfi
      getf.exp           rSignExpX = f8
      // Test x for NaTVal, NaN, +/-0, +/-INF, denormals
      fclass.m           p6,p0  = f8,0x1EF
      addl               r17Ones = 0x1FFFF, r0 // exponent mask
}
{ .mfi
      addl               GR_ad_z_1 = @ltoff(Constants_Z_1#),gp
      fcvt.fx.s1         fXint = f8 // Convert arg to int (int repres. in FR)
      adds               rDelta = 0x3FC, r0
}
;;
{ .mfi
      getf.sig           rSignifX = f8
      fcmp.lt.s1         p15, p14 = f8, f0
      shl                rDelta = rDelta, 20 // single precision 1.5
}
{ .mfi
      ld8                GR_ad_z_1 = [GR_ad_z_1]// get pointer to Constants_Z_1
      fma.s1             fTwo = f1, f1, f1      // 2.0
      addl               rExp8 = 0x10002, r0    // exponent of 8.0
}
;;
{ .mfi
      alloc              rPFS_SAVED = ar.pfs, 0, 34, 4, 0 // get some registers
      fmerge.s           fAbsX = f1, f8                   // |x|
      and                rExpX = rSignExpX, r17Ones       // mask sign bit
}
{ .mib
      addl               rExpHalf = 0xFFFE, r0 // exponent of 0.5
      addl               rExp2 = 0x10000, r0 // exponent of 2.0
      // branch out if x is NaTVal, NaN, +/-0, +/-INF, or denormalized number
(p6)  br.cond.spnt       lgammal_spec
}
;;
_deno_back_to_main_path:
{ .mfi
      // Point to Constants_G_H_h1
      add                rTbl1Addr = 0x040, GR_ad_z_1
      frcpa.s1           fRcpX, p0 = f1, f8 // initial approximation of 1/x
      extr.u             GR_Index1 = rSignifX, 59, 4
}
{ .mib
(p14) cmp.ge.unc         p8, p0 = rExpX, rExp8 // p8 = 1 if x >= 8.0
      adds               rZ625 = 0x3F2, r0
(p8)  br.cond.spnt       lgammal_big_positive // branch out if x >= 8.0
}
;;
{ .mfi
      shladd             rZ1offsett = GR_Index1, 2, GR_ad_z_1  // Point to Z_1
      fmerge.se          fSignifX =  f1, f8 // sifnificand of x
      // Get high 15 bits of significand
      extr.u             GR_X_0 = rSignifX, 49, 15
}
{ .mib
      cmp.lt.unc         p9, p0 = rExpX, rExpHalf // p9 = 1 if |x| < 0.5
      // set p11 if 2 <= x < 4
(p14) cmp.eq.unc         p11, p0 = rExpX, rExp2
(p9)  br.cond.spnt       lgammal_0_half // branch out if |x| < 0.5
}
;;
{ .mfi
      ld4                GR_Z_1 = [rZ1offsett] // Load Z_1
      fms.s1             fA5L = f1, f1, f8 // for 0.75 <= x < 1.3125 path
      shl                rZ625 = rZ625, 20 // sinfle precision 0.625
}
{ .mib
      setf.s             FR_MHalf = rDelta
      // set p10 if x >= 4.0
(p14) cmp.gt.unc         p10, p0 = rExpX, rExp2
      // branch to special path for 4.0 <= x < 8
(p10) br.cond.spnt       lgammal_4_8
}
;;
{ .mfi
      // for 1.3125 <= x < 1.5625 path
      addl               rPolDataPtr= @ltoff(lgammal_loc_min_data),gp
      // argument of polynomial approximation for 1.5625 <= x < 2.25
      fms.s1             fB4 = f8, f1, fTwo
      cmp.eq             p12, p0 = rExpX, rExpHalf
}
{ .mib
      addl               rExpOne = 0xFFFF, r0 // exponent of 1.0
     // set p10 if significand of x >= 1.125
(p11) cmp.le             p11, p0 = 2, GR_Index1
(p11) br.cond.spnt       lgammal_2Q_4
}
;;
{ .mfi
      // point to xMin for 1.3125 <= x < 1.5625 path
      ld8                rPolDataPtr = [rPolDataPtr]
      fcvt.xf            fFltIntX = fXint // RTN(x)
(p14) cmp.eq.unc         p13, p7 = rExpX, rExpOne // p13 set if 1.0 <= x < 2.0
}
{ .mib
      setf.s             FR_FracX = rZ625
      // set p12 if |x| < 0.75
(p12) cmp.gt.unc         p12, p0 = 8, GR_Index1
      // branch out to special path for |x| < 0.75
(p12) br.cond.spnt       lgammal_half_3Q
}
;;
.pred.rel "mutex", p7, p13
{ .mfi
      getf.sig           rXRnd = fXint // integer part of the input value
      fnma.s1            fInvX = f8, fRcpX, f1 // start of 1st NR iteration
      // Get bits 30-15 of X_0 * Z_1
      pmpyshr2.u         GR_X_1 = GR_X_0,GR_Z_1,15
}
{ .mib
(p7)  cmp.eq             p6, p0 = rExpX, rExp2 // p6 set if 2.0 <= x < 2.25
(p13) cmp.le             p6, p0 = 9, GR_Index1
      // branch to special path 1.5625 <= x < 2.25
(p6)  br.cond.spnt       lgammal_13Q_2Q
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      shladd             GR_ad_tbl_1 = GR_Index1, 4, rTbl1Addr // Point to G_1
      fma.s1             fSix = fTwo, fTwo, fTwo // 6.0
      add                GR_ad_q = -0x60, GR_ad_z_1   // Point to Constants_Q
}
{ .mib
      add                rTmpPtr3 = -0x50, GR_ad_z_1
(p13) cmp.gt             p7, p0 = 5, GR_Index1
      // branch to special path 0.75 <= x < 1.3125
(p7)  br.cond.spnt       lgammal_03Q_1Q
}
;;
{ .mfi
      add                rTmpPtr = 8, GR_ad_tbl_1
      fma.s1             fRoot = f8, f1, f1 // x + 1
       // Absolute value of int arg. Will be used as index in table with roots
      sub                rXRnd = r0, rXRnd
}
{ .mib
      ldfe               fA5L = [rPolDataPtr], 16 // xMin
      addl               rNegSingularity = 0x3003E, r0
(p14) br.cond.spnt       lgammal_loc_min
}
;;
{ .mfi
      ldfps              FR_G, FR_H = [GR_ad_tbl_1], 8 // Load G_1, H_1
      nop.f              0
      add                rZ2Addr = 0x140, GR_ad_z_1  // Point to Constants_Z_2
}
{ .mib
      ldfd               FR_h = [rTmpPtr] // Load h_1
      // If arg is less or equal to -2^63
      cmp.geu.unc        p8,p0 = rSignExpX, rNegSingularity
      // Singularity for x < -2^63 since all such arguments are integers
      // branch to special code which deals with singularity
(p8)  br.cond.spnt       lgammal_singularity
}
;;
{ .mfi
      ldfe               FR_log2_hi = [GR_ad_q], 32 // Load log2_hi
      nop.f              0
      extr.u             GR_Index2 = GR_X_1, 6, 4 // Extract bits 6-9 of X_1
}
{ .mfi
      ldfe               FR_log2_lo = [rTmpPtr3], 32 // Load log2_lo
      fms.s1             fDx = f8, f1, fFltIntX // x - RTN(x)
      // index in table with roots and bounds
      adds               rXint = -2, rXRnd
}
;;
{ .mfi
      ldfe               FR_Q4 = [GR_ad_q], 32      // Load Q4
      nop.f              0
      // set p12 if x may be close to negative root: -19.5 < x < -2.0
      cmp.gtu            p12, p0 = 18, rXint
}
{ .mfi
      shladd             GR_ad_z_2 = GR_Index2, 2, rZ2Addr  // Point to Z_2
      fma.s1             fRcpX = fInvX, fRcpX, fRcpX // end of 1st NR iteration
      // Point to Constants_G_H_h2
      add                rTbl2Addr = 0x180, GR_ad_z_1
}
;;
{ .mfi
      shladd             GR_ad_tbl_2 = GR_Index2, 4, rTbl2Addr // Point to G_2
      // set p9 if x is integer and negative
      fcmp.eq.s1         p9, p0 = f8,fFltIntX
      // Point to Constants_G_H_h3
      add                rTbl3Addr = 0x280, GR_ad_z_1
}
{ .mfi
      ld4                GR_Z_2 = [GR_ad_z_2] // Load Z_2
      nop.f              0
      sub                GR_N = rExpX, rExpHalf, 1
}
;;
{ .mfi
      ldfe               FR_Q3 = [rTmpPtr3], 32 // Load Q3
      nop.f              0
      // Point to lnsin polynomial coefficients
      adds               rLnSinDataPtr = 864, rTbl3Addr
}
{ .mfi
      ldfe               FR_Q2 = [GR_ad_q],32 // Load Q2
      nop.f              0
      add                rTmpPtr = 8, GR_ad_tbl_2
}
;;
{ .mfi
      ldfe               FR_Q1 = [rTmpPtr3] // Load Q1
      fcmp.lt.s1         p0, p15 = fAbsX, fSix // p15 is set when x < -6.0
      // point to table with roots and bounds
      adds               rRootsBndAddr = -1296, GR_ad_z_1
}
{ .mfb
      // Put integer N into rightmost significand
      setf.sig           fFloatN = GR_N
      fma.s1             fThirteen = fSix, fTwo, f1 // 13.0
      // Singularity if -2^63 < x < 0 and x is integer
      // branch to special code which deals with singularity
(p9)  br.cond.spnt       lgammal_singularity
}
;;
{ .mfi
      ldfps              FR_G2, FR_H2 = [GR_ad_tbl_2]  // Load G_2, H_2
      // y = |x|/2^(exponent(x)) - 1.5
      fms.s1             FR_FracX = fSignifX, f1, FR_MHalf
      // Get bits 30-15 of X_1 * Z_2
      pmpyshr2.u         GR_X_2 = GR_X_1,GR_Z_2,15
}
{ .mfi
      ldfd               FR_h2 = [rTmpPtr] // Load h_2
      fma.s1             fDxSqr = fDx, fDx, f0 // deltaX^2
      adds               rTmpPtr3 = 128, rLnSinDataPtr
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      getf.exp           rRoot = fRoot // sign and biased exponent of (x + 1)
      nop.f              0
      // set p6 if -4 < x <= -2
      cmp.eq             p6, p0 = rExpX, rExp2
}
{ .mfi
      ldfpd              fLnSin2, fLnSin2L = [rLnSinDataPtr], 16
      fnma.s1            fInvX = f8, fRcpX, f1 // start of 2nd NR iteration
      sub                rIndexPol = rExpX, rExpHalf // index of polynom
}
;;
{ .mfi
      ldfe               fLnSin4 = [rLnSinDataPtr], 96
      // p10 is set if x is potential "right" root
      // p11 set for possible "left" root
      fcmp.lt.s1         p10, p11 = fDx, f0
      shl                rIndexPol = rIndexPol, 6  // (i*16)*4
}
{ .mfi
      ldfpd              fLnSin18, fLnSin20 = [rTmpPtr3], 16
      nop.f              0
      mov                rExp2tom7 = 0x0fff8 // Exponent of 2^-7
}
;;
{ .mfi
      getf.sig           rSignifDx = fDx // Get significand of RTN(x)
      nop.f              0
      // set p6 if -4 < x <= -3.0
(p6)  cmp.le.unc         p6, p0 = 0x8, GR_Index1
}
{ .mfi
      ldfpd              fLnSin22, fLnSin24 = [rTmpPtr3], 16
      nop.f              0
      // mask sign bit in the exponent of (x + 1)
      and                rRoot = rRoot, r17Ones
}
;;
{ .mfi
      ldfe               fLnSin16 = [rLnSinDataPtr], -80
      nop.f              0
      extr.u             GR_Index3 = GR_X_2, 1, 5 // Extract bits 1-5 of X_2
}
{ .mfi
      ldfpd              fLnSin26, fLnSin28 = [rTmpPtr3], 16
      nop.f              0
      and                rXRnd = 1, rXRnd
}
;;
{ .mfi
      shladd             GR_ad_tbl_3 = GR_Index3, 4, rTbl3Addr // Point to G_3
      fms.s1             fDxSqrL = fDx, fDx, fDxSqr // low part of deltaX^2
      // potential "left" root
(p11) adds               rRootsBndAddr = 560, rRootsBndAddr
}
{ .mib
      ldfpd              fLnSin30, fLnSin32 = [rTmpPtr3], 16
      // set p7 if |x+1| < 2^-7
      cmp.lt             p7, p0 =  rRoot, rExp2tom7
      // branch to special path for |x+1| < 2^-7
(p7)  br.cond.spnt       _closeToNegOne
}
;;
{ .mfi
      ldfps              FR_G3, FR_H3 = [GR_ad_tbl_3], 8 // Load G_3, H_3
      fcmp.lt.s1         p14, p0 = fAbsX, fThirteen // set p14 if x > -13.0
      // base address of polynomial on range [-6.0, -0.75]
      adds               rPolDataPtr = 3440, rTbl3Addr
}
{ .mfi
      // (i*16)*4 + (i*16)*8 - offsett of polynomial on range [-6.0, -0.75]
      shladd             rTmpPtr = rIndexPol, 2, rIndexPol
      fma.s1             fXSqr = FR_FracX, FR_FracX, f0 // y^2
      // point to left "near root" bound
(p12) shladd             rRootsBndAddr = rXint, 4, rRootsBndAddr
}
;;
{ .mfi
      ldfpd              fLnSin34, fLnSin36 = [rTmpPtr3], 16
      fma.s1             fRcpX = fInvX, fRcpX, fRcpX // end of 2nd NR iteration
      // add special offsett if -4 < x <= -3.0
(p6)  adds               rPolDataPtr = 640, rPolDataPtr
}
{ .mfi
      // point to right "near root" bound
      adds               rTmpPtr2 = 8, rRootsBndAddr
      fnma.s1            fMOne = f1, f1, f0 // -1.0
      // Point to Bernulli numbers
      adds               rBernulliPtr = 544, rTbl3Addr
}
;;
{ .mfi
      // left bound of "near root" range
(p12) ld8                rLeftBound = [rRootsBndAddr]
      fmerge.se          fNormDx = f1, fDx // significand of DeltaX
      // base address + offsett for polynomial coeff. on range [-6.0, -0.75]
      add                rPolDataPtr = rPolDataPtr, rTmpPtr
}
{ .mfi
      // right bound of "near root" range
(p12) ld8                rRightBound = [rTmpPtr2]
      fcvt.xf            fFloatN = fFloatN
      // special "Bernulli" numbers for Stirling's formula for -13 < x < -6
(p14) adds               rBernulliPtr = 160, rBernulliPtr
}
;;
{ .mfi
      ldfd               FR_h3 = [GR_ad_tbl_3] // Load h_3
      fmpy.s1            FR_G = FR_G, FR_G2 // G = G_1 * G_2
      adds               rTmpPtr3 = -160, rTmpPtr3
}
{ .mfb
      adds               rTmpPtr = 80, rPolDataPtr
      fadd.s1            FR_H = FR_H, FR_H2 // H = H_1 + H_2
      // p15 is set if -2^63 < x < 6.0 and x is not an integer
      // branch to path with implementation using Stirling's formula for neg. x
(p15) br.cond.spnt       _negStirling
}
;;
{ .mfi
      ldfpd              fA3, fA3L = [rPolDataPtr], 16 // A3
      fma.s1             fDelX4 = fDxSqr, fDxSqr, f0 // deltaX^4
      // Get high 4 bits of signif
      extr.u             rIndex1Dx = rSignifDx, 59, 4
}
{ .mfi
      ldfe               fA5 = [rTmpPtr], -16 // A5
      fadd.s1            FR_h = FR_h, FR_h2 // h = h_1 + h_2
      adds               rLnSinTmpPtr = 16, rLnSinDataPtr
}
;;
{ .mfi
      ldfpd              fA0, fA0L = [rPolDataPtr], 16 // A0
      fma.s1             fLnSin20 = fLnSin20, fDxSqr, fLnSin18
      // Get high 15 bits of significand
      extr.u             rX0Dx = rSignifDx, 49, 15
}
{ .mfi
      ldfe               fA4 = [rTmpPtr], 192 // A4
      fms.s1             fXSqrL = FR_FracX, FR_FracX, fXSqr // low part of y^2
      shladd             GR_ad_z_1 = rIndex1Dx, 2, GR_ad_z_1 // Point to Z_1
}
;;
{ .mfi
      ldfpd              fA1, fA1L = [rPolDataPtr], 16 // A1
      fma.s1             fX4 = fXSqr, fXSqr, f0 // y^4
      adds               rTmpPtr2 = 32, rTmpPtr
}
{ .mfi
      ldfpd              fA18, fA19 = [rTmpPtr], 16 // A18, A19
      fma.s1             fLnSin24 = fLnSin24, fDxSqr, fLnSin22
      nop.i              0
}
;;
{ .mfi
      ldfe               fLnSin6 = [rLnSinDataPtr], 32
      fma.s1             fLnSin28 = fLnSin28, fDxSqr, fLnSin26
      nop.i              0
}
{ .mfi
      ldfe               fLnSin8 = [rLnSinTmpPtr], 32
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA20, fA21 = [rTmpPtr], 16 // A20, A21
      fma.s1             fLnSin32 = fLnSin32, fDxSqr, fLnSin30
      nop.i              0
}
{ .mfi
      ldfpd              fA22, fA23 = [rTmpPtr2], 16 // A22, A23
      fma.s1             fB20 = f1, f1, FR_MHalf // 2.5
(p12) cmp.ltu.unc        p6, p0 = rSignifX, rLeftBound
}
;;
{ .mfi
      ldfpd              fA2, fA2L = [rPolDataPtr], 16 // A2
      fmpy.s1            FR_G = FR_G, FR_G3 // G = (G_1 * G_2) * G_3
      // set p6 if x falls in "near root" range
(p6)  cmp.geu.unc        p6, p0 = rSignifX, rRightBound
}
{ .mfb
      adds               rTmpPtr3 = -64, rTmpPtr
      fadd.s1            FR_H = FR_H, FR_H3 // H = (H_1 + H_2) + H_3
      // branch to special path if x falls in "near root" range
(p6)  br.cond.spnt       _negRoots
}
;;
{ .mfi
      ldfpd              fA24, fA25 = [rTmpPtr2], 16 // A24, A25
      fma.s1             fLnSin36 = fLnSin36, fDxSqr, fLnSin34
(p11) cmp.eq.unc         p7, p0 = 1,rXint // p7 set if  -3.0 < x < -2.5
}
{ .mfi
      adds               rTmpPtr = -48, rTmpPtr
      fma.s1             fLnSin20 = fLnSin20, fDxSqr, fLnSin16
      addl               rDelta = 0x5338, r0 // significand of -2.605859375
}
;;
{ .mfi
      getf.exp           GR_N =  fDx // Get N = exponent of DeltaX
      fma.s1             fX6 = fX4, fXSqr, f0 // y^6
      // p7 set if -2.605859375 <= x < -2.5
(p7)  cmp.gt.unc         p7, p0 = rDelta, GR_X_0
}
{ .mfb
      ld4                GR_Z_1 = [GR_ad_z_1] // Load Z_1
      fma.s1             fDelX8 = fDelX4, fDelX4, f0 // deltaX^8
      // branch to special path for -2.605859375 <= x < -2.5
(p7)  br.cond.spnt       _neg2andHalf
}
;;
{ .mfi
      ldfpd              fA14, fA15 = [rTmpPtr3], 16 // A14, A15
      fadd.s1            FR_h = FR_h, FR_h3 // h = (h_1 + h_2) + h_3
      adds               rTmpPtr2 = 128 , rPolDataPtr
}
{ .mfi
      ldfpd              fA16, fA17 = [rTmpPtr], 16 // A16, A17
      fma.s1             fLnSin28 = fLnSin28, fDelX4, fLnSin24
      adds               rPolDataPtr = 144 , rPolDataPtr
}
;;
{ .mfi
      ldfe               fLnSin10 = [rLnSinDataPtr], 32
      fma.s1             fRes1H = fA3, FR_FracX, f0 // (A3*y)hi
      and                GR_N = GR_N, r17Ones // mask sign bit
}
{ .mfi
      ldfe               fLnSin12 = [rLnSinTmpPtr]
      fma.s1             fDelX6 = fDxSqr, fDelX4, f0 // DeltaX^6
      shladd             GR_ad_tbl_1 = rIndex1Dx, 4, rTbl1Addr // Point to G_1
}
;;
{ .mfi
      ldfe               fA13 = [rPolDataPtr], -32 // A13
      fma.s1             fA4 = fA5, FR_FracX, fA4   // A5*y + A4
      // Get bits 30-15 of X_0 * Z_1
      pmpyshr2.u         GR_X_1 = rX0Dx, GR_Z_1, 15
}
{ .mfi
      ldfe               fA12 = [rTmpPtr2], -32 // A12
      fms.s1             FR_r = FR_G, fSignifX, f1 // r = G * S_hi - 1
      sub                GR_N = GR_N, rExpHalf, 1 // unbisaed exponent of DeltaX
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
.pred.rel "mutex",p10,p11
{ .mfi
      ldfe               fA11 = [rPolDataPtr], -32 // A11
      // High part of log(|x|) = Y_hi = N * log2_hi + H
      fma.s1             fResH = fFloatN, FR_log2_hi, FR_H
(p10) cmp.eq             p8, p9 = rXRnd, r0
}
{ .mfi
      ldfe               fA10 = [rTmpPtr2], -32 // A10
      fma.s1             fRes6H = fA1, FR_FracX, f0 // (A1*y)hi
(p11) cmp.eq             p9, p8 = rXRnd, r0
}
;;
{ .mfi
      ldfe               fA9 = [rPolDataPtr], -32 // A9
      fma.s1             fB14 = fLnSin6, fDxSqr, f0 // (LnSin6*deltaX^2)hi
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
{ .mfi
      ldfe               fA8 = [rTmpPtr2], -32 // A8
      fma.s1             fA18 = fA19, FR_FracX, fA18
      nop.i              0
}
;;
{ .mfi
      ldfe               fA7 = [rPolDataPtr] // A7
      fma.s1             fA23 = fA23, FR_FracX, fA22
      nop.i              0
}
{ .mfi
      ldfe               fA6 = [rTmpPtr2] // A6
      fma.s1             fA21 = fA21, FR_FracX, fA20
      nop.i              0
}
;;
{ .mfi
      ldfe               fLnSin14 = [rLnSinDataPtr]
      fms.s1             fRes1L = fA3, FR_FracX, fRes1H // delta((A3*y)hi)
      extr.u             GR_Index2 = GR_X_1, 6, 4 // Extract bits 6-9 of X_1
}
{ .mfi
      setf.sig           fFloatNDx = GR_N
      fadd.s1            fPol = fRes1H, fA2 // (A3*y + A2)hi
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G, FR_H = [GR_ad_tbl_1], 8 // Load G_1, H_1
      fma.s1             fRes2H = fA4, fXSqr, f0 // ((A5 + A4*y)*y^2)hi
      nop.i              0
}
{ .mfi
      shladd             GR_ad_z_2 = GR_Index2, 2, rZ2Addr  // Point to Z_2
      fma.s1             fA25 = fA25, FR_FracX, fA24
      shladd             GR_ad_tbl_2 = GR_Index2, 4, rTbl2Addr  // Point to G_2
}
;;
.pred.rel "mutex",p8,p9
{ .mfi
      ld4                GR_Z_2 = [GR_ad_z_2] // Load Z_2
      fms.s1             fRes6L = fA1, FR_FracX, fRes6H // delta((A1*y)hi)
      // sign of GAMMA(x) is negative
(p8)  adds               rSgnGam = -1, r0
}
{ .mfi
      adds               rTmpPtr = 8, GR_ad_tbl_2
      fadd.s1            fRes3H = fRes6H, fA0 // (A1*y + A0)hi
      // sign of GAMMA(x) is positive
(p9)  adds               rSgnGam = 1, r0
}
;;
{ .mfi
      ldfps              FR_G2, FR_H2 = [GR_ad_tbl_2] // Load G_2, H_2
      // (LnSin6*deltaX^2 + LnSin4)hi
      fadd.s1            fLnSinH = fB14, fLnSin4
      nop.i              0
}
{ .mfi
      ldfd               FR_h2 = [rTmpPtr] // Load h_2
      fms.s1             fB16 = fLnSin6, fDxSqr, fB14 // delta(LnSin6*deltaX^2)
      nop.i              0
}
;;
{ .mfi
      ldfd               fhDelX = [GR_ad_tbl_1] // Load h_1
      fma.s1             fA21 = fA21, fXSqr, fA18
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDelX4, fLnSin32
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes1L = fA3L, FR_FracX, fRes1L // (A3*y)lo
      // Get bits 30-15 of X_1 * Z_
      pmpyshr2.u         GR_X_2 = GR_X_1,GR_Z_2,15
}
{ .mfi
      nop.m              0
      fsub.s1            fPolL = fA2, fPol
      nop.i              0
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      nop.m              0
     // delta(((A5 + A4*y)*y^2)hi)
      fms.s1             fRes2L = fA4, fXSqr, fRes2H
      nop.i              0
}
{ .mfi
      nop.m              0
      // (((A5 + A4*y)*y^2) + A3*y + A2)hi
      fadd.s1            fRes4H = fRes2H, fPol
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s1             fRes6L = fA1L, FR_FracX, fRes6L // (A1*y)lo
      nop.i              0
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fsub.s1            fRes3L = fA0, fRes3H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fLnSinL = fLnSin4, fLnSinH
      nop.i              0
}
{ .mfi
      nop.m              0
      // ((LnSin6*deltaX^2 + LnSin4)*deltaX^2)hi
      fma.s1             fB18 = fLnSinH, fDxSqr, f0
      nop.i              0
}
;;
{ .mfi
      adds               rTmpPtr = 8, rTbl3Addr
      fma.s1             fB16 = fLnSin6, fDxSqrL, fB16 // (LnSin6*deltaX^2)lo
      extr.u             GR_Index3 = GR_X_2, 1, 5 // Extract bits 1-5 of X_2
}
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fXSqr, fA23
      nop.i              0
}
;;
{ .mfi
      shladd             GR_ad_tbl_3 = GR_Index3, 4, rTbl3Addr // Point to G_3
      fadd.s1            fPolL = fPolL, fRes1H
      nop.i              0
}
{ .mfi
      shladd             rTmpPtr = GR_Index3, 4, rTmpPtr // Point to G_3
      fadd.s1            fRes1L = fRes1L, fA2L // (A3*y)lo + A2lo
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G3, FR_H3 = [GR_ad_tbl_3] // Load G_3, H_3
      fma.s1             fRes2L = fA4, fXSqrL, fRes2L // ((A5 + A4*y)*y^2)lo
      nop.i              0
}
{ .mfi
      ldfd               FR_h3 = [rTmpPtr] // Load h_3
      fsub.s1            fRes4L = fPol, fRes4H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // ((((A5 + A4*y)*y^2) + A3*y + A2)*y^2)hi
      fma.s1             fRes7H = fRes4H, fXSqr, f0
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, FR_FracX, fA14
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fRes6H
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes6L = fRes6L, fA0L // (A1*y)lo + A0lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fLnSinL = fLnSinL, fB14

      nop.i              0
}
{ .mfi
      nop.m              0
      // delta((LnSin6*deltaX^2 + LnSin4)*deltaX^2)
      fms.s1             fB20 = fLnSinH, fDxSqr, fB18
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fPolL = fPolL, fRes1L // (A3*y + A2)lo

      nop.i              0
}
{ .mfi
      nop.m              0
      // ((LnSin6*deltaX^2 + LnSin4)*deltaX^2 + LnSin2)hi
      fadd.s1            fLnSin6 = fB18, fLnSin2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fRes2H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, FR_FracX, fA16
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // delta(((((A5 + A4*y)*y^2) + A3*y + A2)*y^2)
      fms.s1             fRes7L = fRes4H, fXSqr, fRes7H
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fPol = fRes7H, fRes3H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fRes6L // (A1*y + A0)lo
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fX4, fA21
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (LnSin6*deltaX^2 + LnSin4)lo
      fadd.s1            fLnSinL = fLnSinL, fB16
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB20 = fLnSinH, fDxSqrL, fB20
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fLnSin4 = fLnSin2, fLnSin6
      nop.i              0
}
{ .mfi
      nop.m              0
      // (((LnSin6*deltaX^2 + LnSin4)*deltaX^2 + LnSin2)*DeltaX^2)hi
      fma.s1             fLnSinH = fLnSin6, fDxSqr, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // ((A5 + A4*y)*y^2)lo + (A3*y + A2)lo
      fadd.s1            fRes2L = fRes2L, fPolL
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, fXSqr, fA15
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // ((((A5 + A4*y)*y^2) + A3*y + A2)*y^2)lo
      fma.s1             fRes7L = fRes4H, fXSqrL, fRes7L
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fPolL = fRes3H, fPol
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA13 = fA13, FR_FracX, fA12
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, FR_FracX, fA10
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // ((LnSin6*deltaX^2 + LnSin4)*deltaX^2)lo
      fma.s1             fB20 = fLnSinL, fDxSqr, fB20
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            FR_G = FR_G, FR_G2 // G = G_1 * G_2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fLnSin4 = fLnSin4, fB18
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fLnSinL = fLnSin6, fDxSqr, fLnSinH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (((A5 + A4*y)*y^2) + A3*y + A2)lo
      fadd.s1            fRes4L = fRes4L, fRes2L
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fhDelX = fhDelX, FR_h2 // h = h_1 + h_2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes7L = fRes7L, fRes3L
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fPolL = fPolL, fRes7H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fcvt.xf            fFloatNDx = fFloatNDx
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            FR_H = FR_H, FR_H2 // H = H_1 + H_2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fmpy.s1            FR_G = FR_G, FR_G3 // G = (G_1 * G_2) * G_3
      nop.i              0
}
{ .mfi
      nop.m              0
      // ((LnSin6*deltaX^2 + LnSin4)*deltaX^2)lo + (LnSin2)lo
      fadd.s1            fLnSin2L = fLnSin2L, fB20
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fX4, fA17
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA13 = fA13, fXSqr, fA11
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, FR_FracX, fA8
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA7 = fA7, FR_FracX, fA6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDelX8, fLnSin28
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin14 = fLnSin14, fDxSqr, fLnSin12
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin10 = fLnSin10, fDxSqr, fLnSin8
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            FR_H = FR_H, FR_H3 // H = (H_1 + H_2) + H_3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRDx = FR_G, fNormDx, f1 // r = G * S_hi - 1
      nop.i              0
}
{ .mfi
      nop.m              0
      // poly_lo = r * Q4 + Q3
      fma.s1             FR_poly_lo = FR_r, FR_Q4, FR_Q3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fmpy.s1            FR_rsq = FR_r, FR_r // rsq = r * r
      nop.i              0
}
{ .mfi
      nop.m              0
      // ((((A5 + A4*y)*y^2) + A3*y + A2)*y^2)lo + (A1*y + A0)lo
      fma.s1             fRes7L = fRes4L, fXSqr, fRes7L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fX4, fA13
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, fXSqr, fA7
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // h = N * log2_lo + h
      fma.s1             FR_h = fFloatN, FR_log2_lo, FR_h
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fhDelX = fhDelX, FR_h3 // h = (h_1 + h_2) + h_3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDelX6, fLnSin20
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin14 = fLnSin14, fDelX4, fLnSin10
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = r * Q4 + Q3
      fma.s1             fPolyLoDx = fRDx, FR_Q4, FR_Q3
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            fRDxSq = fRDx, fRDx // rsq = r * r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Y_hi = N * log2_hi + H
      fma.s1             fResLnDxH = fFloatNDx, FR_log2_hi, FR_H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             FR_rcub = FR_rsq, FR_r, f0 // rcub = r^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA25, fX4, fA9
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fPolL = fPolL, fRes7L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fLnSin4 = fLnSin4, fLnSin2L
      nop.i              0
}
{ .mfi
      nop.m              0
      // h = N * log2_lo + h
      fma.s1             fhDelX = fFloatNDx, FR_log2_lo, fhDelX
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDelX8, fLnSin14
      nop.i              0
}
{ .mfi
      nop.m              0
      // ((LnSin6*deltaX^2 + LnSin4)*deltaX^2 + LnSin2)lo
      fma.s1             fLnSinL = fLnSin6, fDxSqrL, fLnSinL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             fPolyLoDx = fPolyLoDx, fRDx, FR_Q2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fRDxCub = fRDxSq, fRDx, f0 // rcub = r^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      famax.s0              fRes5H = fPol, fResH
      nop.i              0
}
{ .mfi
      nop.m              0
      //  High part of (lgammal(|x|) + log(|x|))
      fadd.s1            fRes1H = fPol, fResH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             FR_poly_lo = FR_poly_lo, FR_r, FR_Q2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPolL = fA9, fX6, fPolL // P25lo
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      famin.s0              fRes5L = fPol, fResH
      nop.i              0
}
{ .mfi
      nop.m              0
      // High part of -(LnSin + log(|DeltaX|))
      fnma.s1            fRes2H = fResLnDxH, f1, fLnSinH
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // (((LnSin6*deltaX^2 + LnSin4)*deltaX^2 + LnSin2)*DeltaX^2)lo
      fma.s1             fLnSinL = fLnSin4, fDxSqr, fLnSinL
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDelX6, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             fPolyHiDx = FR_Q1, fRDxSq, fRDx
      nop.i              0
}
{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             fPolyLoDx = fPolyLoDx, fRDxCub, fhDelX
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes1L = fRes5H, fRes1H
      nop.i              0
}
{ .mfi
      nop.m              0
      //  -(lgammal(|x|) + log(|x|))hi
      fnma.s1            fRes1H = fRes1H, f1, f0

      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             FR_poly_lo = FR_poly_lo, FR_rcub, FR_h
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes2L = fResLnDxH, fMOne, fRes2H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSinL = fLnSin36, fDxSqr, fLnSinL
      nop.i              0
}
{ .mfi
      nop.m              0
      // Y_lo = poly_hi + poly_lo
      fadd.s1            fResLnDxL = fPolyHiDx, fPolyLoDx
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fRes5L
      nop.i              0
}
{ .mfi
      nop.m              0
      // high part of the final result
      fadd.s1            fYH = fRes2H, fRes1H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Y_lo = poly_hi + poly_lo
      fadd.s1            fResL = FR_poly_hi, FR_poly_lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      famax.s0              fRes4H = fRes2H, fRes1H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      famin.s0              fRes4L = fRes2H, fRes1H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (LnSin)lo + (log(|DeltaX|))lo
      fsub.s1            fLnSinL = fLnSinL, fResLnDxL
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fLnSinH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      //(lgammal(|x|))lo + (log(|x|))lo
      fadd.s1            fPolL = fResL, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fYL = fRes4H, fYH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Low part of -(LnSin + log(|DeltaX|))
      fadd.s1            fRes2L = fRes2L, fLnSinL
      nop.i              0
}
{ .mfi
      nop.m              0
      //  High part of (lgammal(|x|) + log(|x|))
      fadd.s1            fRes1L = fRes1L, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fYL = fYL, fRes4L
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes2L = fRes2L, fRes1L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // low part of the final result
      fadd.s1            fYL = fYL, fRes2L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for -6.0 < x <= -0.75, non-integer, "far" from roots
      fma.s0             f8 = fYH, f1, fYL
      // exit here for -6.0 < x <= -0.75, non-integer, "far" from roots
      br.ret.sptk        b0
}
;;

// here if |x+1| < 2^(-7)
.align 32
_closeToNegOne:
{ .mfi
      getf.exp           GR_N =  fDx // Get N = exponent of x
      fmerge.se          fAbsX =  f1, fDx // Form |deltaX|
      // Get high 4 bits of significand of deltaX
      extr.u             rIndex1Dx = rSignifDx, 59, 4
}
{ .mfi
      addl               rPolDataPtr= @ltoff(lgammal_1pEps_data),gp
      fma.s1             fA0L = fDxSqr, fDxSqr, f0 // deltaX^4
      // sign of GAMMA is positive if p10 is set to 1
(p10) adds               rSgnGam = 1, r0
}
;;
{ .mfi
      shladd             GR_ad_z_1 = rIndex1Dx, 2, GR_ad_z_1 // Point to Z_1
      fnma.s1            fResL = fDx, f1, f0 // -(x+1)
      // Get high 15 bits of significand
      extr.u             GR_X_0 = rSignifDx, 49, 15
}
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      shladd             GR_ad_tbl_1 = rIndex1Dx, 4, rTbl1Addr // Point to G_1
}
;;
{ .mfi
      ld4                GR_Z_1 = [GR_ad_z_1] // Load Z_1
      nop.f              0
      and                GR_N = GR_N, r17Ones // mask sign bit
}
{ .mfi
      adds               rTmpPtr = 8, GR_ad_tbl_1
      nop.f              0
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
;;
{ .mfi
      ldfps              FR_G, FR_H = [GR_ad_tbl_1],8 // Load G_1, H_1
      nop.f              0
      adds               rTmpPtr2 = 96, rPolDataPtr
}
{ .mfi
      ldfd               FR_h = [rTmpPtr] // Load h_1
      nop.f              0
      // unbiased exponent of deltaX
      sub                GR_N = GR_N, rExpHalf, 1
}
;;
{ .mfi
      adds               rTmpPtr3 = 192, rPolDataPtr
      nop.f              0
      // sign of GAMMA is negative if p11 is set to 1
(p11) adds               rSgnGam = -1, r0
}
{ .mfi
      ldfe               fA1 = [rPolDataPtr], 16 // A1
      nop.f              0
      nop.i              0
}
;;
{.mfi
      ldfe               fA2 = [rPolDataPtr], 16 // A2
      nop.f              0
      // Get bits 30-15 of X_0 * Z_1
      pmpyshr2.u         GR_X_1 = GR_X_0,GR_Z_1,15
}
{ .mfi
      ldfpd              fA20, fA19 = [rTmpPtr2], 16 // P8, P7
      nop.f              0
      nop.i              0
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe               fA3 = [rPolDataPtr], 16 // A3
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA18, fA17 = [rTmpPtr2], 16 // P6, P5
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               fA4 = [rPolDataPtr], 16 // A4
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA16, fA15 = [rTmpPtr2], 16 // P4, p3
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA5L, fA6 = [rPolDataPtr], 16 // A5, A6
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA14, fA13 = [rTmpPtr2], 16 // P2, P1
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA7, fA8 = [rPolDataPtr], 16 // A7, A8
      nop.f              0
      extr.u             GR_Index2 = GR_X_1, 6, 4 // Extract bits 6-9 of X_1
}
{ .mfi
      ldfe               fLnSin2 = [rTmpPtr2], 16
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      shladd             GR_ad_z_2 = GR_Index2, 2, rZ2Addr  // Point to Z_2
      nop.f              0
      shladd             GR_ad_tbl_2 = GR_Index2, 4, rTbl2Addr // Point to G_2
}
{ .mfi
      ldfe               fLnSin4 = [rTmpPtr2], 32
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ld4                GR_Z_2 = [GR_ad_z_2] // Load Z_2
      nop.f              0
      adds               rTmpPtr = 8, GR_ad_tbl_2
}
{ .mfi
      // Put integer N into rightmost significand
      setf.sig           fFloatN = GR_N
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               fLnSin6 = [rTmpPtr3]
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfe               fLnSin8 = [rTmpPtr2]
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G2, FR_H2 = [GR_ad_tbl_2],8 // Load G_2, H_2
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfd               FR_h2 = [rTmpPtr] // Load h_2
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s1             fResH = fA20, fResL, fA19 //polynomial for log(|x|)
     // Get bits 30-15 of X_1 * Z_2
      pmpyshr2.u         GR_X_2 = GR_X_1,GR_Z_2,15
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s1             fA2 = fA2, fDx, fA1 // polynomial for lgammal(|x|)
      nop.i              0
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      nop.m              0
      fma.s1             fA18 = fA18, fResL, fA17 //polynomial for log(|x|)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA16 = fA16, fResL, fA15 //polynomial for log(|x|)
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA4 = fA4, fDx, fA3 // polynomial for lgammal(|x|)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA14 = fA14, fResL, fA13 //polynomial for log(|x|)
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA6 = fA6, fDx, fA5L // polynomial for lgammal(|x|)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPol = fA8, fDx, fA7 // polynomial for lgammal(|x|)
      extr.u             GR_Index3 = GR_X_2, 1, 5 // Extract bits 1-5 of X_2
}
;;
{ .mfi
      shladd             GR_ad_tbl_3 = GR_Index3, 4, rTbl3Addr // Point to G_3
      // loqw part of lnsin polynomial
      fma.s1             fRes3L = fLnSin4, fDxSqr, fLnSin2
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G3, FR_H3 = [GR_ad_tbl_3], 8 // Load G_3, H_3
      fcvt.xf            fFloatN = fFloatN // N as FP number
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fResH = fResH, fDxSqr, fA18 // High part of log(|x|)
      nop.i              0
}
;;
{ .mfi
      ldfd               FR_h3 = [GR_ad_tbl_3] // Load h_3
      fma.s1             fA4 = fA4, fDxSqr, fA2 // Low part of lgammal(|x|)
      nop.i              0
}
{ .mfi
      nop.m              0
      // high part of lnsin polynomial
      fma.s1             fRes3H = fLnSin8, fDxSqr, fLnSin6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fmpy.s1            FR_G = FR_G, FR_G2 // G = G_1 * G_2
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            FR_H = FR_H, FR_H2 // H = H_1 + H_2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            FR_h = FR_h, FR_h2 // h = h_1 + h_2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA16 = fA16, fDxSqr, fA14 // Low part of log(|x|)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, fDxSqr, fA6 // High part of lgammal(|x|)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResH = fResH, fA0L, fA16 // log(|x|)/deltaX^2 - deltaX
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fmpy.s1            FR_G = FR_G, FR_G3 // G = (G_1 * G_2) * G_3
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            FR_H = FR_H, FR_H3 // H = (H_1 + H_2) + H_3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            FR_h = FR_h, FR_h3 // h = (h_1 + h_2) + h_3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResH = fResH, fDxSqr, fResL // log(|x|)
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, fA0L, fA4 // lgammal(|x|)/|x|
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             FR_r = FR_G, fAbsX, f1 // r = G * S_hi - 1
      nop.i              0
}
{ .mfi
      nop.m              0
      // high part of log(deltaX)= Y_hi = N * log2_hi + H
      fma.s1             fRes4H = fFloatN, FR_log2_hi, FR_H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // h = N * log2_lo + h
      fma.s1             FR_h = fFloatN, FR_log2_lo, FR_h
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResH = fPol, fDx, fResH // lgammal(|x|) + log(|x|)
      nop.i              0
}
{ .mfi
      nop.m              0
      // lnsin/deltaX^2
      fma.s1             fRes3H = fRes3H, fA0L, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = r * Q4 + Q3
      fma.s1             FR_poly_lo = FR_r, FR_Q4, FR_Q3
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            FR_rsq = FR_r, FR_r // rsq = r * r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // lnSin - log(|x|) - lgammal(|x|)
      fms.s1             fResH = fRes3H, fDxSqr, fResH
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             FR_poly_lo = FR_poly_lo, FR_r, FR_Q2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             FR_rcub = FR_rsq, FR_r, f0 // rcub = r^3
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             FR_poly_lo = FR_poly_lo, FR_rcub, FR_h
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // low part of log(|deltaX|) = Y_lo = poly_hi + poly_lo
      fadd.s1            fRes4L = FR_poly_hi, FR_poly_lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fResH = fResH, fRes4L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for |x+1|< 2^(-7) path
      fsub.s0            f8 = fResH, fRes4H
      // exit for |x+1|< 2^(-7) path
      br.ret.sptk        b0
}
;;


// here if -2^63 < x < -6.0 and x is not an integer
// Also we are going to filter out cases when x falls in
// range which is "close enough" to negative root. Rhis case
// may occur only for -19.5 < x since other roots of lgamma are
// insignificant from double extended point of view (they are closer
// to RTN(x) than one ulp(x).
.align 32
_negStirling:
{ .mfi
      ldfe               fLnSin6 = [rLnSinDataPtr], 32
      fnma.s1            fInvX = f8, fRcpX, f1 // start of 3rd NR iteration
      // Get high 4 bits of significand of deltaX
      extr.u             rIndex1Dx = rSignifDx, 59, 4
}
{ .mfi
      ldfe               fLnSin8 = [rTmpPtr3], 32
      fadd.s1            FR_h = FR_h, FR_h2 // h = h_1 + h_2
(p12) cmp.ltu.unc        p6, p0 = rSignifX, rLeftBound
}
;;
{ .mfi
      ldfe               fLnSin10 = [rLnSinDataPtr], 32
      fmpy.s1            FR_G = FR_G, FR_G3 // G = (G_1 * G_2) * G_3
      // Get high 15 bits of significand
      extr.u             GR_X_0 = rSignifDx, 49, 15
}
{ .mfi
      shladd             GR_ad_z_1 = rIndex1Dx, 2, GR_ad_z_1  // Point to Z_1
      fadd.s1            FR_H = FR_H, FR_H3 // H = (H_1 + H_2) + H_3
      // set p6 if x falls in "near root" range
(p6)  cmp.geu.unc        p6, p0 = rSignifX, rRightBound
}
;;
{ .mfi
      getf.exp           GR_N =  fDx // Get N = exponent of x
      fma.s1             fDx4 = fDxSqr, fDxSqr, f0 // deltaX^4
      adds               rTmpPtr = 96, rBernulliPtr
}
{ .mfb
      ld4                GR_Z_1 = [GR_ad_z_1] // Load Z_1
      fma.s1             fLnSin34 = fLnSin34, fDxSqr, fLnSin32
      // branch to special path if x falls in "near root" range
(p6)  br.cond.spnt       _negRoots
}
;;
.pred.rel "mutex",p10,p11
{ .mfi
      ldfe               fLnSin12 = [rTmpPtr3]
      fma.s1             fLnSin26 = fLnSin26, fDxSqr, fLnSin24
(p10) cmp.eq             p8, p9 = rXRnd, r0
}
{ .mfi
      ldfe               fLnSin14 = [rLnSinDataPtr]
      fma.s1             fLnSin30 = fLnSin30, fDxSqr, fLnSin28
(p11) cmp.eq             p9, p8 = rXRnd, r0
}
;;
{ .mfi
      ldfpd              fB2, fB2L = [rBernulliPtr], 16
      fma.s1             fLnSin18 = fLnSin18, fDxSqr, fLnSin16
      shladd             GR_ad_tbl_1 = rIndex1Dx, 4, rTbl1Addr // Point to G_1

}
{ .mfi
      ldfe               fB14 = [rTmpPtr], 16
      fma.s1             fLnSin22 = fLnSin22, fDxSqr, fLnSin20
      and                GR_N = GR_N, r17Ones // mask sign bit
}
;;
{ .mfi
      ldfe               fB4 = [rBernulliPtr], 16
      fma.s1             fInvX = fInvX, fRcpX, fRcpX // end of 3rd NR iteration
      // Get bits 30-15 of X_0 * Z_1
      pmpyshr2.u         GR_X_1 = GR_X_0,GR_Z_1,15
}
{ .mfi
      ldfe               fB16 = [rTmpPtr], 16
      fadd.s1            FR_h = FR_h, FR_h3 // h = (h_1 + h_2) + h_3
      adds               rTmpPtr2 = 8, GR_ad_tbl_1
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe               fB6 = [rBernulliPtr], 16
      fms.s1             FR_r = FR_G, fSignifX, f1 // r = G * S_hi - 1
      adds               rTmpPtr3 = -48, rTmpPtr
}
{ .mfi
      ldfe               fB18 = [rTmpPtr], 16
      // High part of the log(|x|) = Y_hi = N * log2_hi + H
      fma.s1             fResH = fFloatN, FR_log2_hi, FR_H
      sub                GR_N = GR_N, rExpHalf, 1 // unbiased exponent of deltaX
}
;;
.pred.rel "mutex",p8,p9
{ .mfi
      ldfe               fB8 = [rBernulliPtr], 16
      fma.s1             fLnSin36 = fLnSin36, fDx4, fLnSin34
      // sign of GAMMA(x) is negative
(p8)  adds               rSgnGam = -1, r0
}
{ .mfi
      ldfe               fB20 = [rTmpPtr], -160
      fma.s1             fRes5H = fLnSin4, fDxSqr, f0
      // sign of GAMMA(x) is positive
(p9)  adds               rSgnGam = 1, r0

}
;;
{ .mfi
      ldfe               fB10 = [rBernulliPtr], 16
      fma.s1             fLnSin30 = fLnSin30, fDx4, fLnSin26
(p14) adds               rTmpPtr = -160, rTmpPtr
}
{ .mfi
      ldfe               fB12 = [rTmpPtr3], 16
      fma.s1             fDx8 = fDx4, fDx4, f0 // deltaX^8
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
;;
{ .mfi
      ldfps              fGDx, fHDx = [GR_ad_tbl_1], 8 // Load G_1, H_1
      fma.s1             fDx6 = fDx4, fDxSqr, f0 // deltaX^6
      extr.u             GR_Index2 = GR_X_1, 6, 4 // Extract bits 6-9 of X_1
}
{ .mfi
      ldfd               fhDx = [rTmpPtr2] // Load h_1
      fma.s1             fLnSin22 = fLnSin22, fDx4, fLnSin18
      nop.i              0
}
;;
{ .mfi
      // Load two parts of C
      ldfpd              fRes1H, fRes1L = [rTmpPtr], 16
      fma.s1             fRcpX = fInvX, fInvX, f0  // (1/x)^2
      shladd             GR_ad_tbl_2 = GR_Index2, 4, rTbl2Addr // Point to G_2
}
{ .mfi
      shladd             GR_ad_z_2 = GR_Index2, 2, rZ2Addr  // Point to Z_2
      fma.s1             FR_h = fFloatN, FR_log2_lo, FR_h// h = N * log2_lo + h
      nop.i              0
}
;;
{ .mfi
      ld4                GR_Z_2 = [GR_ad_z_2] // Load Z_2
      fnma.s1            fInvXL = f8, fInvX, f1 // relative error of 1/x
      nop.i              0
}
{ .mfi
      adds               rTmpPtr2 = 8, GR_ad_tbl_2
      fma.s1             fLnSin8 = fLnSin8, fDxSqr, fLnSin6
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G2, FR_H2 = [GR_ad_tbl_2],8   // Load G_2, H_2
      // poly_lo = r * Q4 + Q3
      fma.s1             FR_poly_lo = FR_r, FR_Q4, FR_Q3
      nop.i              0
}
{ .mfi
      ldfd               fh2Dx = [rTmpPtr2] // Load h_2
      fmpy.s1            FR_rsq = FR_r, FR_r // rsq = r * r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA1L = fB2, fInvX, f0 // (B2*(1/x))hi
      nop.i              0
}
{ .mfi
      // Put integer N into rightmost significand
      setf.sig           fFloatNDx = GR_N
      fms.s1             fRes4H = fResH, f1, f1  // ln(|x|)hi - 1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2H = fRes5H, fLnSin2//(lnSin4*DeltaX^2 + lnSin2)hi
      // Get bits 30-15 of X_1 * Z_2
      pmpyshr2.u         GR_X_2 = GR_X_1,GR_Z_2,15
}
{ .mfi
      nop.m              0
      fms.s1             fRes5L = fLnSin4, fDxSqr, fRes5H
      nop.i              0
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      nop.m              0
      fma.s1             fInvX4 = fRcpX, fRcpX, f0 // (1/x)^4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB6 = fB6, fRcpX, fB4
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s1             fB18 = fB18, fRcpX, fB16
      nop.i              0
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s1             fInvXL = fInvXL, fInvX, f0 // low part of 1/x
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             FR_poly_lo = FR_poly_lo, FR_r, FR_Q2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             FR_rcub = FR_rsq, FR_r, f0 // rcub = r^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes3H = fRes4H, f8, f0 // (-|x|*(ln(|x|)-1))hi
      extr.u             GR_Index3 = GR_X_2, 1, 5  // Extract bits 1-5 of X_2
}
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
;;
{ .mfi
      shladd             GR_ad_tbl_3 = GR_Index3, 4, rTbl3Addr // Point to G_3
      fms.s1             fA2L = fB2, fInvX, fA1L // delta(B2*(1/x))
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            fBrnH = fRes1H, f1, fA1L // (-C - S(1/x))hi
      nop.i              0
}
;;
{ .mfi
      ldfps              fG3Dx, fH3Dx = [GR_ad_tbl_3],8 // Load G_3, H_3
      fma.s1             fInvX8 = fInvX4, fInvX4, f0 // (1/x)^8
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB10 = fB10, fRcpX, fB8
      nop.i              0
}
;;

{ .mfi
      ldfd               fh3Dx = [GR_ad_tbl_3] // Load h_3
      fma.s1             fB20 = fB20, fInvX4, fB18
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB14 = fB14, fRcpX, fB12
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDx8, fLnSin30
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin12 = fLnSin12, fDxSqr, fLnSin10
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes2L = fLnSin2, fRes2H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fRes2H, fDxSqr, f0 // high part of LnSin
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fnma.s1            fResH = fResH, FR_MHalf, fResH // -0.5*ln(|x|)hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            fGDx = fGDx, FR_G2 // G = G_1 * G_2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             FR_poly_lo = FR_poly_lo, FR_rcub, FR_h
      nop.i              0
}
{ .mfi
      nop.m              0
      // B2lo*(1/x)hi+ delta(B2*(1/x))
      fma.s1             fA2L = fB2L, fInvX, fA2L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB20 = fB20, fInvX4, fB14
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB10 = fB10, fInvX4, fB6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fcvt.xf            fFloatNDx = fFloatNDx
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin14 = fLnSin14, fDx4, fLnSin12
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36, fDx8, fLnSin22
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fRes3L = fRes4H, f8, fRes3H // delta(-|x|*(ln(|x|)-1))
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fmpy.s1            fGDx = fGDx, fG3Dx // G = (G_1 * G_2) * G_3
      nop.i              0
}
{ .mfi
      nop.m              0
      // (-|x|*(ln(|x|)-1) - 0.5ln(|x|))hi
      fadd.s1            fRes4H = fRes3H, fResH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA2L = fInvXL, fB2, fA2L //(B2*(1/x))lo
      nop.i              0
}
{ .mfi
      nop.m              0
      // low part of log(|x|) = Y_lo = poly_hi + poly_lo
      fadd.s1            fResL = FR_poly_hi, FR_poly_lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB20 = fB20, fInvX8, fB10
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fInvX3 = fInvX, fRcpX, f0 // (1/x)^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fHDx = fHDx, FR_H2 // H = H_1 + H_2
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes5L = fRes5L, fLnSin2L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fRes5H
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fhDx = fhDx, fh2Dx // h = h_1 + h_2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fBrnL = fRes1H, fMOne, fBrnH
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             FR_r = fGDx, fNormDx, f1 // r = G * S_hi - 1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes3L = fResL, f8 , fRes3L // (-|x|*(ln(|x|)-1))lo
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes4L = fRes3H, fRes4H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // low part of "Bernulli" polynomial
      fma.s1             fB20 = fB20, fInvX3, fA2L
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            fResL = fResL, FR_MHalf, fResL // -0.5*ln(|x|)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fHDx = fHDx, fH3Dx // H = (H_1 + H_2) + H_3
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fPolL = fRes2H, fDxSqr, fPol
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fhDx = fhDx, fh3Dx // h = (h_1 + h_2) + h_3
      nop.i              0
}
{ .mfi
      nop.m              0
      // (-|x|*(ln(|x|)-1) - 0.5ln(|x|) - C - S(1/x))hi
      fadd.s1            fB14 = fRes4H, fBrnH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = r * Q4 + Q3
      fma.s1             FR_poly_lo = FR_r, FR_Q4, FR_Q3
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            FR_rsq = FR_r, FR_r // rsq = r * r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fResH
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fBrnL = fBrnL, fA1L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (-|x|*(ln(|x|)-1))lo + (-0.5ln(|x|))lo
      fadd.s1            fRes3L = fRes3L, fResL
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            fB20 = fRes1L, f1, fB20 // -Clo - S(1/x)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fRes5L // (lnSin4*DeltaX^2 + lnSin2)lo
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPolL = fDxSqrL, fRes2H, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin14 = fLnSin14, fDx4, fLnSin8
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin36,  fDx8, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             FR_poly_lo = FR_poly_lo, FR_r, FR_Q2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             FR_rcub = FR_rsq, FR_r, f0 // rcub = r^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fB12 = fRes4H, fB14
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (-|x|*(ln(|x|)-1) - 0.5ln(|x|))lo
      fadd.s1            fRes4L = fRes4L, fRes3L
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fBrnL = fBrnL, fB20 // (-C - S(1/x))lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // high part of log(|DeltaX|) = Y_hi = N * log2_hi + H
      fma.s1             fLnDeltaH = fFloatNDx, FR_log2_hi, fHDx
      nop.i              0
}
{ .mfi
      nop.m              0
      // h = N * log2_lo + h
      fma.s1             fhDx = fFloatNDx, FR_log2_lo, fhDx
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPolL = fRes2L, fDxSqr, fPolL
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin14 = fLnSin36,  fDxSqr, fLnSin14
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (-|x|*(ln(|x|)-1) - 0.5ln(|x|))lo + (- C - S(1/x))lo
      fadd.s1            fBrnL = fBrnL, fRes4L
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fB12 = fB12, fBrnH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             FR_poly_lo = FR_poly_lo, FR_rcub, fhDx
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            fRes1H = fLnDeltaH, f1, fPol//(-ln(|DeltaX|) + LnSin)hi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPolL = fDxSqrL, fRes2L, fPolL
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin36 = fLnSin14, fDx6, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (-|x|*(ln(|x|)-1) - 0.5ln(|x|) - C - S(1/x))lo
      fadd.s1            fB12 = fB12, fBrnL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // low part of log(|DeltaX|) =  Y_lo = poly_hi + poly_lo
      fadd.s1            fLnDeltaL= FR_poly_hi, FR_poly_lo
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fRes1L = fLnDeltaH, fMOne, fRes1H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fPolL = fPolL, fLnSin36
      nop.i              0
}
{ .mfi
      nop.m              0
      //(-|x|*(ln(|x|)-1)-0.5ln(|x|) - C - S(1/x))hi + (-ln(|DeltaX|) + LnSin)hi
      fadd.s1            f8 = fRes1H, fB14
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      //max((-|x|*(ln(|x|)-1)-0.5ln(|x|) - C - S(1/x))hi,
      //    (-ln(|DeltaX|) + LnSin)hi)
      famax.s1           fMaxNegStir = fRes1H, fB14
      nop.i              0
}
{ .mfi
      nop.m              0
      //min((-|x|*(ln(|x|)-1)-0.5ln(|x|) - C - S(1/x))hi,
      //    (-ln(|DeltaX|) + LnSin)hi)
      famin.s1           fMinNegStir = fRes1H, fB14
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fPol
      nop.i              0
}
{ .mfi
      nop.m              0
      // (-ln(|DeltaX|))lo + (LnSin)lo
      fnma.s1            fPolL = fLnDeltaL, f1, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            f9 = fMaxNegStir, f8 // delta1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fPolL // (-ln(|DeltaX|) + LnSin)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            f9 = f9, fMinNegStir
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fB12
      nop.i              0
}
;;
{ .mfi
      // low part of the result
      fadd.s1            f9 = f9, fRes1L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for -2^63 < x < -6.0 path
      fma.s0             f8 = f8, f1, f9
      // exit here  for -2^63 < x < -6.0 path
      br.ret.sptk        b0
}
;;

// here if x falls in neighbourhood of any negative root
// "neighbourhood" typically means that |lgammal(x)| < 0.17
// on the [-3.0,-2.0] range |lgammal(x)| has even less
// magnitude
// rXint contains index of the root
// p10 is set if root belongs to "right" ones
// p11 is set if root belongs to "left" ones
// lgammal(x) is approximated by polynomial of
// 19th degree from (x - root) argument
.align 32
_negRoots:
{ .mfi
      addl          rPolDataPtr= @ltoff(lgammal_right_roots_polynomial_data),gp
      nop.f              0
      shl                rTmpPtr2 = rXint, 7 // (i*16)*8
}
{ .mfi
      adds               rRootsAddr = -288, rRootsBndAddr
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               fRoot = [rRootsAddr] // FP representation of root
      nop.f              0
      shl                rTmpPtr = rXint, 6  // (i*16)*4
}
{ .mfi
(p11) adds               rTmpPtr2 = 3536, rTmpPtr2
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      shladd             rTmpPtr = rXint, 4, rTmpPtr // (i*16) + (i*16)*4
}
{ .mfi
      adds               rTmpPtr3 = 32, rTmpPtr2
      nop.f              0
      nop.i              0
}
;;
.pred.rel "mutex",p10,p11
{ .mfi
      add                rTmpPtr3 = rTmpPtr, rTmpPtr3
      nop.f              0
(p10) cmp.eq             p8, p9 = rXRnd, r0
}
{ .mfi
      // (i*16) + (i*16)*4 + (i*16)*8
      add                rTmpPtr = rTmpPtr, rTmpPtr2
      nop.f              0
(p11) cmp.eq             p9, p8 = rXRnd, r0
}
;;
{ .mfi
      add                rTmpPtr2 = rPolDataPtr, rTmpPtr3
      nop.f              0
      nop.i              0
}
{ .mfi
      add                rPolDataPtr = rPolDataPtr, rTmpPtr // begin + offsett
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA0, fA0L = [rPolDataPtr], 16 // A0
      nop.f              0
      adds               rTmpPtr = 112, rTmpPtr2
}
{ .mfi
      ldfpd              fA2, fA2L = [rTmpPtr2], 16 // A2
      nop.f              0
      cmp.eq             p12, p13 = 4, rSgnGamSize
}
;;
{ .mfi
      ldfpd              fA1, fA1L = [rPolDataPtr], 16 // A1
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfe               fA3 = [rTmpPtr2], 128 // A4
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA12, fA13 = [rTmpPtr], 16 // A12, A13
      nop.f              0
      adds               rTmpPtr3 = 64, rPolDataPtr
}
{ .mfi
      ldfpd              fA16, fA17 = [rTmpPtr2], 16 // A16, A17
      nop.f              0
      adds               rPolDataPtr = 32, rPolDataPtr
}
;;
.pred.rel "mutex",p8,p9
{ .mfi
      ldfpd              fA14, fA15 = [rTmpPtr], 16 // A14, A15
      nop.f              0
      // sign of GAMMA(x) is negative
(p8)  adds               rSgnGam = -1, r0
}
{ .mfi
      ldfpd              fA18, fA19 = [rTmpPtr2], 16 // A18, A19
      nop.f              0
      // sign of GAMMA(x) is positive
(p9)  adds               rSgnGam = 1, r0
}
;;
{ .mfi
      ldfe               fA4 = [rPolDataPtr], 16 // A4
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA6, fA7 = [rTmpPtr3], 16 // A6, A7
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               fA5 = [rPolDataPtr], 16 // A5
      // if x equals to (rounded) root exactly
      fcmp.eq.s1         p6, p0 = f8, fRoot
      nop.i              0
}
{ .mfi
      ldfpd              fA8, fA9 = [rTmpPtr3], 16 // A8, A9
      fms.s1             FR_FracX = f8, f1, fRoot
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p12) st4                [rSgnGamAddr] = rSgnGam
      nop.f              0
      nop.i              0
}
{ .mfb
      // store signgam if size of variable is 8 bytes
(p13) st8                [rSgnGamAddr] = rSgnGam
      // answer if x equals to (rounded) root exactly
(p6)  fadd.s0            f8 = fA0, fA0L
      // exit if x equals to (rounded) root exactly
(p6)  br.ret.spnt        b0
}
;;
{ .mmf
      ldfpd              fA10, fA11 = [rTmpPtr3], 16 // A10, A11
      nop.m              0
      nop.f              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResH = fA2, FR_FracX, f0 // (A2*x)hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA4L = FR_FracX, FR_FracX, f0 // x^2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, FR_FracX, fA16
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fA13 = fA13, FR_FracX, fA12
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA19 = fA19, FR_FracX, fA18
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fA15 = fA15, FR_FracX, fA14
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fPol = fA7, FR_FracX, fA6
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fA9 = fA9, FR_FracX, fA8
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fResL = fA2, FR_FracX, fResH // delta(A2*x)
      nop.i              0
}
{.mfi
      nop.m              0
      fadd.s1            fRes1H = fResH, fA1 // (A2*x + A1)hi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, FR_FracX, fA10
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fA5L = fA4L, fA4L, f0 // x^4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA19 = fA19, fA4L, fA17
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fA15 = fA15, fA4L, fA13
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, FR_FracX, fA5
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fA3L = fA4L, FR_FracX, f0 // x^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // delta(A2*x) + A2L*x = (A2*x)lo
      fma.s1             fResL = fA2L, FR_FracX, fResL
      nop.i              0
}
{.mfi
      nop.m              0
      fsub.s1            fRes1L = fA1, fRes1H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, fA4L, fA9
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fA19 = fA19, fA5L, fA15
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fPol = fPol, FR_FracX, fA4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fResL = fResL, fA1L // (A2*x)lo + A1
      nop.i              0
}
{.mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fResH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes2H = fRes1H, FR_FracX, f0 // ((A2*x + A1)*x)hi
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fA19 = fA19, fA5L, fA11
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fPol = fPol, FR_FracX, fA3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fResL // (A2*x + A1)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // delta((A2*x + A1)*x)
      fms.s1             fRes2L = fRes1H, FR_FracX, fRes2H
      nop.i              0
}
{.mfi
      nop.m              0
      fadd.s1            fRes3H = fRes2H, fA0 // ((A2*x + A1)*x + A0)hi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA19 = fA19, fA5L, f0
      nop.i              0
}

;;
{ .mfi
      nop.m              0
      fma.s1             fRes2L = fRes1L, FR_FracX, fRes2L // ((A2*x + A1)*x)lo
      nop.i              0
}
{.mfi
      nop.m              0
      fsub.s1            fRes3L = fRes2H, fRes3H
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fPol = fA19, FR_FracX, fPol
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fA0
      nop.i              0
}
{.mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fA0L // ((A2*x + A1)*x)lo + A0L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fRes2L // (((A2*x + A1)*x) + A0)lo
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fRes3L = fPol, fA3L, fRes3L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for arguments which are close to negative roots
      fma.s0             f8 = fRes3H, f1, fRes3L
      // exit here for arguments which are close to negative roots
      br.ret.sptk        b0
}
;;

// here if |x| < 0.5
.align 32
lgammal_0_half:
{ .mfi
      ld4                GR_Z_1 = [rZ1offsett] // Load Z_1
      fma.s1             fA4L = f8, f8, f0 // x^2
      addl               rPolDataPtr    = @ltoff(lgammal_0_Half_data), gp
}
{ .mfi
      shladd             GR_ad_tbl_1 = GR_Index1, 4, rTbl1Addr// Point to G_1
      nop.f              0
      addl               rLnSinDataPtr    = @ltoff(lgammal_lnsin_data), gp
}
;;
{ .mfi
      ldfps              FR_G, FR_H = [GR_ad_tbl_1],8 // Load G_1, H_1
      nop.f              0
      // Point to Constants_Z_2
      add                GR_ad_z_2 = 0x140, GR_ad_z_1
}
{ .mfi
      add                GR_ad_q = -0x60, GR_ad_z_1 // Point to Constants_Q
      nop.f              0
      // Point to Constants_G_H_h2
      add                GR_ad_tbl_2 = 0x180, GR_ad_z_1
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      // Point to Constants_G_H_h3
      add                GR_ad_tbl_3 = 0x280, GR_ad_z_1
}
{ .mfi
      ldfd               FR_h = [GR_ad_tbl_1] // Load h_1
      nop.f              0
      sub                GR_N = rExpX, rExpHalf, 1
}
;;
{ .mfi
      ld8                rLnSinDataPtr    = [rLnSinDataPtr]
      nop.f              0
      // Get bits 30-15 of X_0 * Z_1
      pmpyshr2.u         GR_X_1 = GR_X_0,GR_Z_1,15
}
{ .mfi
      ldfe               FR_log2_hi = [GR_ad_q],16 // Load log2_hi
      nop.f              0
      sub                GR_N = r0, GR_N
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe               FR_log2_lo = [GR_ad_q], 16 // Load log2_lo
      nop.f              0
      add                rTmpPtr2 = 320, rPolDataPtr
}
{ .mfi
      add                rTmpPtr = 32, rPolDataPtr
      nop.f              0
      // exponent of 0.25
      adds               rExp2 = -1, rExpHalf
}
;;
{ .mfi
      ldfpd              fA3, fA3L = [rPolDataPtr], 16 // A3
      fma.s1             fA5L = fA4L, fA4L, f0 // x^4
      nop.i              0
}
{ .mfi
      ldfpd              fA1, fA1L = [rTmpPtr], 16 // A1
      fms.s1             fB8 = f8, f8, fA4L // x^2 - <x^2>
      // set p6 if -0.5 < x <= -0.25
(p15) cmp.eq.unc         p6, p0 = rExpX, rExp2
}
;;
{ .mfi
      ldfpd              fA2, fA2L = [rPolDataPtr], 16 // A2
      nop.f              0
      // set p6 if -0.5 < x <= -0.40625
(p6)  cmp.le.unc         p6, p0 = 10, GR_Index1
}
{ .mfi
      ldfe               fA21 = [rTmpPtr2], -16 // A21
      // Put integer N into rightmost significand
      nop.f              0
      adds               rTmpPtr = 240, rTmpPtr
}
;;
{ .mfi
      setf.sig           fFloatN = GR_N
      nop.f              0
      extr.u             GR_Index2 = GR_X_1, 6, 4 // Extract bits 6-9 of X_1
}
{ .mfi
      ldfe               FR_Q4 = [GR_ad_q], 16 // Load Q4
      nop.f              0
      adds               rPolDataPtr = 304, rPolDataPtr
}
;;
{ .mfi
      ldfe               fA20 = [rTmpPtr2], -32 // A20
      nop.f              0
      shladd             GR_ad_z_2 = GR_Index2, 2, GR_ad_z_2  // Point to Z_2
}
{ .mfi
      ldfe               fA19 = [rTmpPtr], -32 // A19
      nop.f              0
      shladd             GR_ad_tbl_2 = GR_Index2, 4, GR_ad_tbl_2// Point to G_2
}
;;
{ .mfi
      ldfe               fA17 = [rTmpPtr], -32 // A17
      nop.f              0
      adds               rTmpPtr3 = 8, GR_ad_tbl_2
}
{ .mfb
      ldfe               fA18 = [rTmpPtr2], -32 // A18
      nop.f              0
      // branch to special path for -0.5 < x <= 0.40625
(p6)  br.cond.spnt       lgammal_near_neg_half
}
;;
{ .mmf
      ld4                GR_Z_2 = [GR_ad_z_2] // Load Z_2
      ldfe               fA15 = [rTmpPtr], -32 // A15
      fma.s1             fB20 = fA5L, fA5L, f0 // x^8
}
;;
{ .mmf
      ldfe               fA16 = [rTmpPtr2], -32 // A16
      ldfe               fA13 = [rTmpPtr], -32 // A13
      fms.s1             fB16 = fA4L, fA4L, fA5L
}
;;
{ .mmf
      ldfps              FR_G2, FR_H2 = [GR_ad_tbl_2], 8 // Load G_2, H_2
      ldfd               FR_h2 = [rTmpPtr3] // Load h_2
      fmerge.s           fB10 = f8, fA5L // sign(x) * x^4
}
;;
{ .mmi
      ldfe               fA14 = [rTmpPtr2], -32 // A14
      ldfe               fA11 = [rTmpPtr], -32 // A11
      // Get bits 30-15 of X_1 * Z_2
      pmpyshr2.u         GR_X_2 = GR_X_1,GR_Z_2,15
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe               fA12 = [rTmpPtr2], -32 // A12
      fma.s1             fRes4H = fA3, fAbsX, f0
      adds               rTmpPtr3 = 16, GR_ad_q
}
{ .mfi
      ldfe               fA9 = [rTmpPtr], -32 // A9
      nop.f              0
      nop.i              0
}
;;
{ .mmf
      ldfe               fA10 = [rTmpPtr2], -32 // A10
      ldfe               fA7 = [rTmpPtr], -32 // A7
      fma.s1             fB18 = fB20, fB20, f0 // x^16
}
;;
{ .mmf
      ldfe               fA8 = [rTmpPtr2], -32 // A8
      ldfe               fA22 = [rPolDataPtr], 16 // A22
      fcvt.xf            fFloatN = fFloatN
}
;;
{ .mfi
      ldfe               fA5 = [rTmpPtr], -32 // A5
      fma.s1             fA21 = fA21, fAbsX, fA20 // v16
      extr.u             GR_Index3 = GR_X_2, 1, 5 // Extract bits 1-5 of X_2
}
{ .mfi
      ldfe               fA6 = [rTmpPtr2], -32 // A6
      nop.f              0
      nop.i              0
}
;;
{ .mmf
      // Point to G_3
      shladd             GR_ad_tbl_3 = GR_Index3, 4, GR_ad_tbl_3
      ldfe               fA4 = [rTmpPtr2], -32 // A4
      fma.s1             fA19 = fA19, fAbsX, fA18 // v13
}
;;
.pred.rel "mutex",p14,p15
{ .mfi
      ldfps              FR_G3, FR_H3 = [GR_ad_tbl_3],8 // Load G_3, H_3
      fms.s1             fRes4L = fA3, fAbsX, fRes4H
(p14) adds               rSgnGam = 1, r0
}
{ .mfi
      cmp.eq             p6, p7 = 4, rSgnGamSize
      fadd.s1            fRes2H = fRes4H, fA2
(p15) adds               rSgnGam = -1, r0
}
;;

{ .mfi
      ldfd               FR_h3 = [GR_ad_tbl_3] // Load h_3
      fma.s1             fA17 = fA17, fAbsX, fA16 // v12
      nop.i              0
}
;;
{ .mfi
      ldfe               FR_Q3 = [GR_ad_q], 32 // Load Q3
      fmpy.s1            FR_G = FR_G, FR_G2 // G = G_1 * G_2
      nop.i              0
}
{ .mfi
      ldfe               FR_Q2 = [rTmpPtr3], 16 // Load Q2
      fadd.s1            FR_H = FR_H, FR_H2 // H = H_1 + H_2
      nop.i              0
}
;;
{ .mfi
      ldfe               FR_Q1 = [GR_ad_q] // Load Q1
      fma.s1             fA15 = fA15, fAbsX, fA14 // v8
      nop.i              0
}
{ .mfi
      adds               rTmpPtr3 = 32, rLnSinDataPtr
      fadd.s1            FR_h = FR_h, FR_h2 // h = h_1 + h_2
      nop.i              0
}
;;
{ .mmf
      ldfpd              fLnSin2, fLnSin2L = [rLnSinDataPtr], 16
      ldfe               fLnSin6 = [rTmpPtr3], 32
      fma.s1             fA13 = fA13, fAbsX, fA12 // v7

}
;;
{ .mfi
      ldfe               fLnSin4 = [rLnSinDataPtr], 32
      fma.s1             fRes4L = fA3L, fAbsX, fRes4L
      nop.i              0
}
{ .mfi
      ldfe               fLnSin10 = [rTmpPtr3], 32
      fsub.s1            fRes2L = fA2, fRes2H
      nop.i              0
}
;;
{ .mfi
      ldfe               fLnSin8 = [rLnSinDataPtr], 32
      fma.s1             fResH = fRes2H, fAbsX, f0
      nop.i              0
}
{ .mfi
      ldfe               fLnSin14 = [rTmpPtr3], 32
      fma.s1             fA22 = fA22, fA4L, fA21 // v15
      nop.i              0
}
;;
{ .mfi
      ldfe               fLnSin12 = [rLnSinDataPtr], 32
      fma.s1             fA9 = fA9, fAbsX, fA8 // v4
      nop.i              0
}
{ .mfi
      ldfd               fLnSin18 = [rTmpPtr3], 16
      fma.s1             fA11 = fA11, fAbsX, fA10 // v5
      nop.i              0
}
;;
{ .mfi
      ldfe               fLnSin16 = [rLnSinDataPtr], 24
      fma.s1             fA19 = fA19, fA4L, fA17 // v11
      nop.i              0
}
{ .mfi
      ldfd               fLnSin22 = [rTmpPtr3], 16
      fma.s1             fPolL = fA7, fAbsX, fA6
      nop.i              0
}
;;
{ .mfi
      ldfd               fLnSin20 = [rLnSinDataPtr], 16
      fmpy.s1            FR_G = FR_G, FR_G3 // G = (G_1 * G_2) * G_3
      nop.i              0
}
{ .mfi
      ldfd               fLnSin26 = [rTmpPtr3], 16
      fadd.s1            FR_H = FR_H, FR_H3 // H = (H_1 + H_2) + H_3
      nop.i              0
}
;;
{ .mfi
      ldfd               fLnSin24 = [rLnSinDataPtr], 16
      fadd.s1            fRes2L = fRes2L, fRes4H
      nop.i              0
}
{ .mfi
      ldfd               fLnSin30 = [rTmpPtr3], 16
      fadd.s1            fA2L = fA2L, fRes4L
      nop.i              0
}
;;
{ .mfi
      ldfd               fLnSin28 = [rLnSinDataPtr], 16
      fms.s1             fResL = fRes2H, fAbsX, fResH
      nop.i              0
}
{ .mfi
      ldfd               fLnSin34 = [rTmpPtr3], 8
      fadd.s1            fRes2H = fResH, fA1
      nop.i              0
}
;;
{ .mfi
      ldfd               fLnSin32 = [rLnSinDataPtr]
      fma.s1             fA11 = fA11, fA4L, fA9 // v3
      nop.i              0
}
{ .mfi
      ldfd               fLnSin36 = [rTmpPtr3]
      fma.s1             fA15 = fA15, fA4L, fA13 // v6
      nop.i              0
}
;;

{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fadd.s1            FR_h = FR_h, FR_h3 // h = (h_1 + h_2) + h_3
      nop.i              0
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s1             fA5 = fA5, fAbsX, fA4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             FR_r = FR_G, fSignifX, f1 // r = G * S_hi - 1
      nop.i              0
}
{ .mfi
      nop.m               0
      // High part of the log(|x|): Y_hi = N * log2_hi + H
      fms.s1             FR_log2_hi = fFloatN, FR_log2_hi, FR_H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fA3L = fRes2L, fA2L
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA22 = fA22, fA5L, fA19
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes2L = fA1, fRes2H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fRes3H = fRes2H, f8, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, fA5L, fA11 // v2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin18 = fLnSin18, fA4L, fLnSin16
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // h = N * log2_lo + h
      fms.s1             FR_h = fFloatN, FR_log2_lo, FR_h
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPolL = fPolL, fA4L, fA5
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = r * Q4 + Q3
      fma.s1             FR_poly_lo = FR_r, FR_Q4, FR_Q3
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            FR_rsq = FR_r, FR_r // rsq = r * r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResL = fA3L, fAbsX, fResL
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin30 = fLnSin30, fA4L, fLnSin28
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fResH
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fRes3L = fRes2H, f8, fRes3H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1H = fRes3H, FR_log2_hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fB20, fA22, fA15
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin34 = fLnSin34, fA4L, fLnSin32
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin14 = fLnSin14, fA4L, fLnSin12
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             FR_poly_lo = FR_poly_lo, FR_r, FR_Q2
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            FR_rcub = FR_rsq, FR_r, f0 // rcub = r^3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fA1L = fA1L, fResL
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fma.s1             fLnSin22 = fLnSin22, fA4L, fLnSin20
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin26 = fLnSin26, fA4L, fLnSin24
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fsub.s1            fRes1L = FR_log2_hi, fRes1H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, fA5L, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin34 = fLnSin36, fA5L, fLnSin34
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin18 = fLnSin18, fA5L, fLnSin14
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin6 = fLnSin6, fA4L, fLnSin4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin10 = fLnSin10, fA4L, fLnSin8
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fA1L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             FR_poly_lo = FR_poly_lo, FR_rcub, FR_h
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB2 = fLnSin2, fA4L, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fRes3H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, fB10, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin26 = fLnSin26, fA5L, fLnSin22
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin34 = fLnSin34, fA5L, fLnSin30
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin10 = fLnSin10, fA5L, fLnSin6
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin2L = fLnSin2L, fA4L, f0
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fma.s1             fRes3L = fRes2L, f8, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Y_lo = poly_hi + poly_lo
      fsub.s1            FR_log2_lo = FR_poly_lo, FR_poly_hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fB4 = fLnSin2, fA4L, fB2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2H = fRes1H, fPol
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin34 = fLnSin34, fB20, fLnSin26
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fLnSin18 = fLnSin18, fB20, fLnSin10
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fLnSin2L = fB8, fLnSin2, fLnSin2L
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fadd.s1            FR_log2_lo = FR_log2_lo, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes2L = fRes1H, fRes2H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB6 = fLnSin34, fB18, fLnSin18
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fB4 = fLnSin2L, fB4
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, FR_log2_lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fPol
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB12 = fB6, fA5L, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fRes1L
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fms.s1             fB14 = fB6, fA5L, fB12
      nop.i              0
}
{ .mfb
      nop.m              0
      fadd.s1            fLnSin30 = fB2, fB12
      // branch out if x is negative
(p15) br.cond.spnt       _O_Half_neg
}
;;
{ .mfb
      nop.m              0
      // sign(x)*Pol(|x|) - log(|x|)
      fma.s0             f8 = fRes2H, f1, fRes2L
      // it's an answer already for positive x
      // exit if 0 < x < 0.5
      br.ret.sptk        b0
}
;;

// here if x is negative and |x| < 0.5
.align 32
_O_Half_neg:
{ .mfi
      nop.m              0
      fma.s1             fB14 = fB16, fB6, fB14
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fLnSin16 = fB2, fLnSin30
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fResH = fLnSin30, fRes2H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fLnSin16 = fLnSin16, fB12
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fB4 = fB14, fB4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fLnSin16 = fB4, fLnSin16
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fResL = fRes2H, fResH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fResL = fResL, fLnSin30
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fLnSin16 = fLnSin16, fRes2L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fResL = fResL, fLnSin16
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for -0.5 < x < 0
      fma.s0             f8 = fResH, f1, fResL
      // exit for -0.5 < x < 0
      br.ret.sptk        b0
}
;;

// here if x >= 8.0
// there are two computational paths:
// 1) For x >10.0 Stirling's formula is used
// 2) Polynomial approximation for 8.0 <= x <= 10.0
.align 32
lgammal_big_positive:
{ .mfi
      addl               rPolDataPtr = @ltoff(lgammal_data), gp
      fmerge.se          fSignifX =  f1, f8
      // Get high 15 bits of significand
      extr.u             GR_X_0 = rSignifX, 49, 15
}
{.mfi
      shladd             rZ1offsett = GR_Index1, 2, GR_ad_z_1  // Point to Z_1
      fnma.s1            fInvX = f8, fRcpX, f1 // start of 1st NR iteration
      adds               rSignif1andQ = 0x5, r0
}
;;
{.mfi
      ld4                GR_Z_1 = [rZ1offsett] // Load Z_1
      nop.f              0
      shl                rSignif1andQ = rSignif1andQ, 61 // significand of 1.25
}
{  .mfi
      cmp.eq             p8, p0 = rExpX, rExp8 // p8 = 1 if 8.0 <= x < 16
      nop.f              0
      adds               rSgnGam = 1, r0 // gamma is positive at this range
}
;;
{ .mfi
      shladd             GR_ad_tbl_1 = GR_Index1, 4, rTbl1Addr// Point to G_1
      nop.f              0
      add                GR_ad_q = -0x60, GR_ad_z_1 // Point to Constants_Q
}
{ .mlx
      ld8                rPolDataPtr    = [rPolDataPtr]
      movl               rDelta = 0x3FF2000000000000
}
;;
{ .mfi
      ldfps              FR_G, FR_H = [GR_ad_tbl_1],8 // Load G_1, H_1
      nop.f              0
      add                GR_ad_z_2 = 0x140, GR_ad_z_1 // Point to Constants_Z_2
}
{ .mfi
      // Point to Constants_G_H_h2
      add                GR_ad_tbl_2 = 0x180, GR_ad_z_1
      nop.f              0
      // p8 = 1 if 8.0 <= x <= 10.0
(p8)  cmp.leu.unc        p8, p0 = rSignifX, rSignif1andQ
}
;;
{ .mfi
      ldfd               FR_h = [GR_ad_tbl_1] // Load h_1
      nop.f              0
      // Get bits 30-15 of X_0 * Z_1
      pmpyshr2.u         GR_X_1 = GR_X_0,GR_Z_1,15
}
{ .mfb
(p8)  setf.d             FR_MHalf = rDelta
      nop.f              0
(p8)  br.cond.spnt       lgammal_8_10 // branch out if 8.0 <= x <= 10.0
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe               fA1 = [rPolDataPtr], 16 // Load overflow threshold
      fma.s1             fRcpX = fInvX, fRcpX, fRcpX // end of 1st NR iteration
      // Point to Constants_G_H_h3
      add                GR_ad_tbl_3 = 0x280, GR_ad_z_1
}
{ .mlx
      nop.m              0
      movl               rDelta = 0xBFE0000000000000 // -0.5 in DP
}
;;
{ .mfi
      ldfe               FR_log2_hi = [GR_ad_q],16 // Load log2_hi
      nop.f              0
      sub                GR_N = rExpX, rExpHalf, 1 // unbiased exponent of x
}
;;
{ .mfi
      ldfe               FR_log2_lo = [GR_ad_q],16 // Load log2_lo
      nop.f              0
      nop.i              0
}
{ .mfi
      setf.d             FR_MHalf = rDelta
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      // Put integer N into rightmost significand
      setf.sig           fFloatN = GR_N
      nop.f              0
      extr.u             GR_Index2 = GR_X_1, 6, 4 // Extract bits 6-9 of X_1
}
{ .mfi
      ldfe               FR_Q4 = [GR_ad_q], 16 // Load Q4
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      shladd             GR_ad_z_2 = GR_Index2, 2, GR_ad_z_2  // Point to Z_2
      nop.f              0
      shladd             GR_ad_tbl_2 = GR_Index2, 4, GR_ad_tbl_2// Point to G_2
}
{ .mfi
      ldfe               FR_Q3 = [GR_ad_q], 16 // Load Q3
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ld4                GR_Z_2 = [GR_ad_z_2] // Load Z_2
      fnma.s1            fInvX = f8, fRcpX, f1 // start of 2nd NR iteration
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G2, FR_H2 = [GR_ad_tbl_2], 8 // Load G_2, H_2
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfd               FR_h2 = [GR_ad_tbl_2] // Load h_2
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               FR_Q2 = [GR_ad_q],16 // Load Q2
      nop.f              0
      // Get bits 30-15 of X_1 * Z_2
      pmpyshr2.u         GR_X_2 = GR_X_1,GR_Z_2,15
}
;;
//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe               FR_Q1 = [GR_ad_q] // Load Q1
      fcmp.gt.s1         p7,p0 = f8, fA1 // check if x > overflow threshold
      nop.i              0
}
;;
{.mfi
      ldfpd              fA0, fA0L = [rPolDataPtr], 16 // Load two parts of C
      fma.s1             fRcpX = fInvX, fRcpX, fRcpX // end of 2nd NR iteration
      nop.i              0
}
;;
{ .mfb
      ldfpd              fB2, fA1 = [rPolDataPtr], 16
      nop.f              0
(p7)  br.cond.spnt       lgammal_overflow // branch if x > overflow threshold
}
;;
{.mfi
      ldfe               fB4 = [rPolDataPtr], 16
      fcvt.xf            fFloatN = fFloatN
      extr.u             GR_Index3 = GR_X_2, 1, 5 // Extract bits 1-5 of X_2
}
;;
{ .mfi
      shladd             GR_ad_tbl_3 = GR_Index3, 4, GR_ad_tbl_3// Point to G_3
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfe               fB6 = [rPolDataPtr], 16
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfps              FR_G3, FR_H3 = [GR_ad_tbl_3], 8 // Load G_3, H_3
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfd               FR_h3 = [GR_ad_tbl_3] // Load h_3
      fmpy.s1            FR_G = FR_G, FR_G2 // G = G_1 * G_2
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            FR_H = FR_H, FR_H2 // H = H_1 + H_2
      nop.i              0
}
;;

{ .mfi
      ldfe               fB8 = [rPolDataPtr], 16
      fadd.s1            FR_h = FR_h, FR_h2 // h = h_1 + h_2
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            fInvX = f8, fRcpX, f1 // start of 3rd NR iteration
      nop.i              0
}
;;
{ .mfi
      ldfe               fB10 = [rPolDataPtr], 16
      nop.f              0
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
;;
{ .mfi
      ldfe               fB12 = [rPolDataPtr], 16
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               fB14 = [rPolDataPtr], 16
      nop.f              0
      nop.i              0
}
;;

{ .mfi
      ldfe               fB16 = [rPolDataPtr], 16
      // get double extended coefficients from two doubles
      // two doubles are needed in Stitling's formula for negative x
      fadd.s1            fB2 = fB2, fA1
      nop.i              0
}
;;
{ .mfi
      ldfe               fB18 = [rPolDataPtr], 16
      fma.s1             fInvX = fInvX, fRcpX, fRcpX // end of 3rd NR iteration
      nop.i              0
}
;;
{ .mfi
      ldfe               fB20 = [rPolDataPtr], 16
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fmpy.s1            FR_G = FR_G, FR_G3 // G = (G_1 * G_2) * G_3
      nop.i              0
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fadd.s1            FR_H = FR_H, FR_H3 // H = (H_1 + H_2) + H_3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            FR_h = FR_h, FR_h3 // h = (h_1 + h_2) + h_3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRcpX = fInvX, fInvX, f0 // 1/x^2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA0L = fB2, fInvX, fA0L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             FR_r = fSignifX, FR_G, f1 // r = G * S_hi - 1
      nop.i              0
}
{ .mfi
      nop.m              0
      // High part of the log(x): Y_hi = N * log2_hi + H
      fma.s1             fRes2H = fFloatN, FR_log2_hi, FR_H
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      // h = N * log2_lo + h
      fma.s1             FR_h = fFloatN, FR_log2_lo, FR_h
      nop.i              0
}
{ .mfi
      nop.m              0
      // High part of the log(x): Y_hi = N * log2_hi + H
      fma.s1             fRes1H = fFloatN, FR_log2_hi, FR_H
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fPol = fB18, fRcpX, fB16 // v9
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA2L = fRcpX, fRcpX, f0   // v10
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fA3 = fB6, fRcpX, fB4     // v3
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA4 = fB10, fRcpX, fB8    // v4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes2H =fRes2H, f1, f1 //  log_Hi(x) -1
      nop.i              0
}
{ .mfi
      nop.m              0
      // poly_lo = r * Q4 + Q3
      fma.s1             FR_poly_lo = FR_r, FR_Q4, FR_Q3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes1H = fRes1H, FR_MHalf, f0 // -0.5*log_Hi(x)
      nop.i              0
}
{ .mfi
      nop.m              0
      fmpy.s1            FR_rsq = FR_r, FR_r // rsq = r * r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA7 = fB14, fRcpX, fB12  // v7
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA8 = fA2L, fB20, fPol   // v8
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA2 = fA4, fA2L, fA3    // v2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA4L = fA2L, fA2L, f0    // v5
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResH = fRes2H, f8, f0 // (x*(ln(x)-1))hi
      nop.i              0
}
{ .mfi
      nop.m              0
      // poly_lo = poly_lo * r + Q2
      fma.s1             FR_poly_lo = FR_poly_lo, FR_r, FR_Q2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             FR_rcub = FR_rsq, FR_r, f0 // rcub = r^3
      nop.i              0
}
{ .mfi
      nop.m              0
      // poly_hi = Q1 * rsq + r
      fma.s1             FR_poly_hi = FR_Q1, FR_rsq, FR_r
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA11 = fRcpX, fInvX, f0 // 1/x^3
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA6 = fA8, fA2L, fA7   // v6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fResL = fRes2H, f8, fResH // d(x*(ln(x)-1))
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes3H = fResH, fRes1H // (x*(ln(x)-1) -0.5ln(x))hi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // poly_lo = poly_lo*r^3 + h
      fma.s1             FR_poly_lo = FR_poly_lo, FR_rcub, FR_h
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPol = fA4L, fA6, fA2   // v1
      nop.i              0
}
{ .mfi
      nop.m              0
      // raise inexact exception
      fma.s0             FR_log2_lo = FR_log2_lo, FR_log2_lo, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4H = fRes3H, fA0 // (x*(ln(x)-1) -0.5ln(x))hi + Chi
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes3L = fResH, fRes3H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Y_lo = poly_hi + poly_lo
      fadd.s1            fRes2L = FR_poly_hi, FR_poly_lo
      nop.i              0
}
;;

{ .mfi
      nop.m              0
      fma.s1             fA0L = fPol, fA11, fA0L // S(1/x) + Clo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fRes1H
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes4L = fRes3H, fRes4H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResL = fRes2L, f8 , fResL // lo part of x*(ln(x)-1)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Clo + S(1/x) - 0.5*logLo(x)
      fma.s1             fA0L = fRes2L, FR_MHalf, fA0L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fA0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Clo + S(1/x) - 0.5*logLo(x) + (x*(ln(x)-1))lo
      fadd.s1            fA0L = fA0L, fResL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fA0L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      fma.s0             f8 = fRes4H, f1, fRes4L
      // exit for x > 10.0
      br.ret.sptk        b0
}
;;
// here if 8.0 <= x <= 10.0
// Result = P15(y), where y = x/8.0 - 1.5
.align 32
lgammal_8_10:
{ .mfi
      addl               rPolDataPtr    = @ltoff(lgammal_8_10_data), gp
      fms.s1             FR_FracX = fSignifX, f1, FR_MHalf // y = x/8.0 - 1.5
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
;;
{ .mfi
      ld8                rLnSinDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      adds               rZ1offsett = 32, rLnSinDataPtr
      nop.f              0
      nop.i              0
}
{ .mfi
      adds               rLnSinDataPtr = 48, rLnSinDataPtr
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA1, fA1L = [rPolDataPtr], 16 // A1
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfe               fA2 = [rZ1offsett], 32 // A5
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA0, fA0L = [rPolDataPtr], 16 // A0
      fma.s1             FR_rsq = FR_FracX, FR_FracX, f0 // y^2
      nop.i              0
}
{ .mfi
      ldfe               fA3 = [rLnSinDataPtr],32 // A5
      nop.f              0
      nop.i              0
}
;;
{ .mmf
      ldfe               fA4 = [rZ1offsett], 32 // A4
      ldfe               fA5 = [rLnSinDataPtr], 32 // A5
      nop.f              0
}
;;
{ .mmf
      ldfe               fA6 = [rZ1offsett], 32 // A6
      ldfe               fA7 = [rLnSinDataPtr], 32 // A7
      nop.f              0
}
;;
{ .mmf
      ldfe               fA8 = [rZ1offsett], 32 // A8
      ldfe               fA9 = [rLnSinDataPtr], 32 // A9
      nop.f              0
}
;;
{ .mmf
      ldfe               fA10 = [rZ1offsett], 32 // A10
      ldfe               fA11 = [rLnSinDataPtr], 32 // A11
      nop.f              0
}
;;
{ .mmf
      ldfe               fA12 = [rZ1offsett], 32 // A12
      ldfe               fA13 = [rLnSinDataPtr], 32 // A13
      fma.s1             FR_Q4 = FR_rsq, FR_rsq, f0 // y^4
}
;;
{ .mmf
      ldfe               fA14 = [rZ1offsett], 32 // A14
      ldfe               fA15 = [rLnSinDataPtr], 32 // A15
      nop.f              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes1H = FR_FracX, fA1, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA3 = fA3, FR_FracX, fA2 // v4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA5 = fA5, FR_FracX, fA4 // v5
      nop.i              0
}
;;
{ .mfi
      // store sign of GAMMA(x) if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s1             fA3L = FR_Q4, FR_Q4, f0 // v9 = y^8
      nop.i              0
}
{ .mfi
      // store sign of GAMMA(x) if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s1             fA7 = fA7, FR_FracX, fA6 // v7
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, FR_FracX, fA8 // v8
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes1L = FR_FracX, fA1, fRes1H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, FR_FracX, fA10 // v12
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA13 = fA13, FR_FracX, fA12 // v13
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fRes2H = fRes1H, f1, fA0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, FR_FracX, fA14 // v16
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA5 = fA5, FR_rsq, fA3 // v3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, FR_rsq, fA7 // v6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes1L = FR_FracX, fA1L, fRes1L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes2L = fA0, f1, fRes2H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA13 = fA13, FR_rsq, fA11 // v11
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, FR_Q4, fA5 // v2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes1L = fRes1L, f1, fA0L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes2L = fRes2L, f1, fRes1H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, FR_Q4, fA13 // v10
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes2L = fRes1L, f1, fRes2L
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fA3L, fA15, fA9
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             f8 = FR_rsq , fPol, fRes2H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, FR_rsq, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes1L = fRes2H, f1, f8
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes1L = fRes1L, f1, fPol
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fRes1L = fRes1L, f1, fRes2L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      fma.s0             f8 = f8, f1, fRes1L
      // exit for 8.0 <= x <= 10.0
      br.ret.sptk        b0
}
;;

// here if 4.0 <=x < 8.0
.align 32
lgammal_4_8:
{ .mfi
      addl               rPolDataPtr= @ltoff(lgammal_4_8_data),gp
      fms.s1             FR_FracX = fSignifX, f1, FR_MHalf
      adds               rSgnGam = 1, r0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
;;

{ .mfb
      adds               rTmpPtr = 160, rPolDataPtr
      nop.f              0
      // branch to special path which computes polynomial of 25th degree
      br.sptk            lgamma_polynom25
}
;;

// here if 2.25 <=x < 4.0
.align 32
lgammal_2Q_4:
{ .mfi
      addl               rPolDataPtr= @ltoff(lgammal_2Q_4_data),gp
      fms.s1             FR_FracX = fSignifX, f1, FR_MHalf
      adds               rSgnGam = 1, r0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
;;

{ .mfb
      adds               rTmpPtr = 160, rPolDataPtr
      nop.f              0
      // branch to special path which computes polynomial of 25th degree
      br.sptk            lgamma_polynom25
}
;;

// here if 0.5 <= |x| < 0.75
.align 32
lgammal_half_3Q:
.pred.rel "mutex", p14, p15
{ .mfi
(p14) addl               rPolDataPtr= @ltoff(lgammal_half_3Q_data),gp
     // FR_FracX = x - 0.625 for positive x
(p14) fms.s1             FR_FracX = f8, f1, FR_FracX
(p14) adds               rSgnGam = 1, r0
}
{ .mfi
(p15) addl               rPolDataPtr= @ltoff(lgammal_half_3Q_neg_data),gp
     // FR_FracX = x + 0.625 for negative x
(p15) fma.s1             FR_FracX = f8, f1, FR_FracX
(p15) adds               rSgnGam = -1, r0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
       nop.f              0
       nop.i              0
}
;;
{ .mfb
      adds               rTmpPtr = 160, rPolDataPtr
      nop.f              0
      // branch to special path which computes polynomial of 25th degree
      br.sptk            lgamma_polynom25
}
;;
// here if 1.3125 <= x < 1.5625
.align 32
lgammal_loc_min:
{ .mfi
      adds               rSgnGam = 1, r0
      nop.f              0
      nop.i              0
}
{ .mfb
      adds               rTmpPtr = 160, rPolDataPtr
      fms.s1             FR_FracX = f8, f1, fA5L
      br.sptk            lgamma_polynom25
}
;;
// here if -2.605859375 <= x < -2.5
// special polynomial approximation used since neither "near root"
// approximation nor reflection formula give satisfactory accuracy on
// this range
.align 32
_neg2andHalf:
{ .mfi
      addl               rPolDataPtr= @ltoff(lgammal_neg2andHalf_data),gp
      fma.s1             FR_FracX = fB20, f1, f8 // 2.5 + x
      adds               rSgnGam = -1, r0
}
;;
{.mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
;;
{ .mfb
      adds               rTmpPtr = 160, rPolDataPtr
      nop.f              0
      // branch to special path which computes polynomial of 25th degree
      br.sptk            lgamma_polynom25
}
;;

// here if -0.5 < x <= -0.40625
.align 32
lgammal_near_neg_half:
{ .mmf
      addl               rPolDataPtr= @ltoff(lgammal_near_neg_half_data),gp
      setf.exp           FR_FracX = rExpHalf
      nop.f              0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      adds               rSgnGam = -1, r0
}
;;
{ .mfb
      adds               rTmpPtr = 160, rPolDataPtr
      fma.s1             FR_FracX = FR_FracX, f1, f8
      // branch to special path which computes polynomial of 25th degree
      br.sptk            lgamma_polynom25
}
;;

// here if there an answer is P25(x)
// rPolDataPtr, rTmpPtr point to coefficients
// x is in FR_FracX register
.align 32
lgamma_polynom25:
{ .mfi
      ldfpd              fA3, fA0L = [rPolDataPtr], 16 // A3
      nop.f              0
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
{ .mfi
      ldfpd              fA18, fA19 = [rTmpPtr], 16 // D7, D6
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA1, fA1L = [rPolDataPtr], 16 // A1
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA16, fA17 = [rTmpPtr], 16 // D4, D5
      nop.f              0
}
;;
{ .mfi
      ldfpd              fA12, fA13 = [rPolDataPtr], 16 // D0, D1
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA14, fA15 = [rTmpPtr], 16 // D2, D3
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA24, fA25 = [rPolDataPtr], 16 // C21, C20
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA22, fA23 = [rTmpPtr], 16 // C19, C18
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA2, fA2L = [rPolDataPtr], 16 // A2
      fma.s1             fA4L = FR_FracX, FR_FracX, f0 // x^2
      nop.i              0
}
{ .mfi
      ldfpd              fA20, fA21 = [rTmpPtr], 16 // C17, C16
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfe               fA11 = [rTmpPtr], 16 // E7
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA0, fA3L = [rPolDataPtr], 16 // A0
      nop.f              0
      nop.i              0
};;
{ .mfi
      ldfe               fA10 = [rPolDataPtr], 16 // E6
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfe               fA9 = [rTmpPtr], 16 // E5
      nop.f              0
      nop.i              0
}
;;
{ .mmf
      ldfe               fA8 = [rPolDataPtr], 16 // E4
      ldfe               fA7 = [rTmpPtr], 16 // E3
      nop.f              0
}
;;
{ .mmf
      ldfe               fA6 = [rPolDataPtr], 16 // E2
      ldfe               fA5 = [rTmpPtr], 16 // E1
      nop.f              0
}
;;
{ .mfi
      ldfe               fA4 = [rPolDataPtr], 16 // E0
      fma.s1             fA5L = fA4L, fA4L, f0 // x^4
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fB2 = FR_FracX, FR_FracX, fA4L // x^2 - <x^2>
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s1             fRes4H = fA3, FR_FracX, f0 // (A3*x)hi
      nop.i              0
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s1             fA19 = fA19, FR_FracX, fA18 // D7*x + D6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResH = fA1, FR_FracX, f0 // (A1*x)hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB6 = fA1L, FR_FracX, fA0L // A1L*x + A0L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, FR_FracX, fA16 // D5*x + D4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, FR_FracX, fA14 // D3*x + D2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, FR_FracX, fA24 // C21*x + C20
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA13 = fA13, FR_FracX, fA12 // D1*x + D0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA23 = fA23, FR_FracX, fA22 // C19*x + C18
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA21 = fA21, FR_FracX, fA20 // C17*x + C16
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes4L = fA3, FR_FracX, fRes4H // delta((A3*x)hi)
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes2H = fRes4H, fA2 // (A3*x + A2)hi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fResL = fA1, FR_FracX, fResH // d(A1*x)
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes1H = fResH, fA0 // (A1*x + A0)hi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA19 = fA19, fA4L, fA17 // Dhi
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, FR_FracX, fA10 // E7*x + E6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // Doing this to raise inexact flag
      fma.s0             fA10 = fA0, fA0, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, fA4L, fA13 // Dlo
      nop.i              0
}
{ .mfi
      nop.m              0
      // (C21*x + C20)*x^2 + C19*x + C18
      fma.s1             fA25 = fA25, fA4L, fA23
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, FR_FracX, fA8 // E5*x + E4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA7 = fA7, FR_FracX, fA6 // E3*x + E2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes4L = fA3L, FR_FracX, fRes4L // (A3*x)lo
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes2L = fA2, fRes2H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fResL = fResL, fB6 // (A1L*x + A0L) + d(A1*x)
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes1L = fA0, fRes1H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA5 = fA5, FR_FracX, fA4 // E1*x + E0
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB8 = fA5L, fA5L, f0  // x^8
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // ((C21*x + C20)*x^2 + C19*x + C18)*x^2 + C17*x + C16
      fma.s1             fA25 = fA25, fA4L, fA21
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA19 = fA19, fA5L, fA15 // D
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, fA4L, fA9 // Ehi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fRes4H
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fA2L // (A3*x)lo + A2L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes3H = fRes2H, fA4L, f0 //  ((A3*x + A2)*x^2)hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fResH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes3L = fRes2H, fB2, f0 // (A3*x + A2)hi*d(x^2)
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA7 = fA7, fA4L, fA5 // Elo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fB8, fA19 // C*x^8 + D
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fRes4L // (A3*x + A2)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fB4 = fRes2H, fA4L, fRes3H // d((A3*x + A2)*x^2))
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fResL // (A1*x + A0)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fB20 = fRes3H, fRes1H // Phi
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA11 = fA11, fA5L, fA7 // E
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      //  ( (A3*x + A2)lo*<x^2> + (A3*x + A2)hi*d(x^2))
      fma.s1             fRes3L = fRes2L, fA4L, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // d((A3*x + A2)*x^2)) + (A1*x + A0)lo
      fadd.s1            fRes1L = fRes1L, fB4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fB18 = fRes1H, fB20
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fA25, fB8, fA11
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fB18 = fB18, fRes3H
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fRes4H = fPol, fA5L, fB20
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPolL = fPol, fA5L, f0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fB18 = fB18, fRes1L // Plo
      nop.i              0
}
{ .mfi
      nop.m              0
      fsub.s1            fRes4L = fB20, fRes4H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fB18 = fB18, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fB18
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      fma.s0             f8 = fRes4H, f1, fRes4L
      // P25(x) computed, exit here
      br.ret.sptk        b0
}
;;


// here if 0.75 <= x < 1.3125
.align 32
lgammal_03Q_1Q:
{ .mfi
      addl               rPolDataPtr= @ltoff(lgammal_03Q_1Q_data),gp
      fma.s1             FR_FracX = fA5L, f1, f0 // x
      adds               rSgnGam = 1, r0
}
{ .mfi
      nop.m              0
      fma.s1             fB4 = fA5L, fA5L, f0 // x^2
      nop.i              0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
;;
{ .mfb
      adds               rTmpPtr = 144, rPolDataPtr
      nop.f              0
      br.sptk            lgamma_polynom24x
}
;;

// here if 1.5625 <= x < 2.25
.align 32
lgammal_13Q_2Q:
{ .mfi
      addl               rPolDataPtr= @ltoff(lgammal_13Q_2Q_data),gp
      fma.s1             FR_FracX = fB4, f1, f0 // x
      adds               rSgnGam = 1, r0
}
{ .mfi
      nop.m              0
      fma.s1             fB4 = fB4, fB4, f0 // x^2
      nop.i              0
}
;;
{ .mfi
      ld8                rPolDataPtr = [rPolDataPtr]
      nop.f              0
      nop.i              0
}
;;
{ .mfb
      adds               rTmpPtr = 144, rPolDataPtr
      nop.f              0
      br.sptk            lgamma_polynom24x
}
;;

// here if result is Pol24(x)
// x is in FR_FracX,
// rPolDataPtr, rTmpPtr point to coefficients
.align 32
lgamma_polynom24x:
{ .mfi
      ldfpd              fA4, fA2L = [rPolDataPtr], 16
      nop.f              0
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
{ .mfi
      ldfpd              fA23, fA24 = [rTmpPtr], 16 // C18, C19
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA3, fA1L = [rPolDataPtr], 16
      fma.s1             fA5L = fB4, fB4, f0 // x^4
      nop.i              0
}
{ .mfi
      ldfpd              fA19, fA20 = [rTmpPtr], 16 // D6, D7
      fms.s1             fB2 = FR_FracX, FR_FracX, fB4 // x^2 - <x^2>
      nop.i              0
}
;;
{ .mmf
      ldfpd              fA15, fA16 = [rPolDataPtr], 16 // D2, D3
      ldfpd              fA17, fA18 = [rTmpPtr], 16 // D4, D5
      nop.f              0
}
;;
{ .mmf
      ldfpd              fA13, fA14 = [rPolDataPtr], 16 // D0, D1
      ldfpd              fA12, fA21 = [rTmpPtr], 16 // E7, C16
      nop.f              0
}
;;
{ .mfi
      ldfe               fA11 = [rPolDataPtr], 16 // E6
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfe               fA10 = [rTmpPtr], 16 // E5
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA2, fA4L = [rPolDataPtr], 16
      nop.f              0
      nop.i              0
}
{ .mfi
      ldfpd              fA1, fA3L = [rTmpPtr], 16
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      ldfpd              fA22, fA25 = [rPolDataPtr], 16 // C17, C20
      fma.s1             fA0 = fA5L, fA5L, f0 // x^8
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA0L = fA5L, FR_FracX, f0 // x^5
      nop.i              0
}
;;
{ .mmf
      ldfe               fA9 = [rPolDataPtr], 16 // E4
      ldfe               fA8 = [rTmpPtr], 16 // E3
      nop.f              0
}
;;
{ .mmf
      ldfe               fA7 = [rPolDataPtr], 16 // E2
      ldfe               fA6 = [rTmpPtr], 16 // E1
      nop.f              0
}
;;
{ .mfi
      ldfe               fA5 = [rTmpPtr], 16 // E0
      fma.s1             fRes4H = fA4, fB4, f0 // A4*<x^2>
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fA24, FR_FracX, fA23 // C19*x + C18
      nop.i              0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s1             fRes1H = fA3, fB4, f0 // A3*<x^2>
      nop.i              0
}
{ .mfi
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s1             fA1L = fA3, fB2,fA1L //  A3*d(x^2) + A1L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA20 = fA20, FR_FracX, fA19 // D7*x + D6
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA18 = fA18, FR_FracX, fA17 // D5*x + D4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA16 = fA16, FR_FracX, fA15 // D3*x + D2
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA14 = fA14, FR_FracX, fA13 // D1*x + D0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA2L = fA4, fB2,fA2L //  A4*d(x^2) + A2L
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA12 = fA12, FR_FracX, fA11 // E7*x + E6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes2L = fA4, fB4, fRes4H  // delta(A4*<x^2>)
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes2H = fRes4H, fA2       // A4*<x^2> + A2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fms.s1             fRes3L = fA3, fB4, fRes1H  // delta(A3*<x^2>)
      nop.i              0
}
{ .mfi
      nop.m              0
      fadd.s1            fRes3H = fRes1H, fA1       // A3*<x^2> + A1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA20 = fA20, fB4, fA18 // (D7*x + D6)*x^2 + D5*x + D4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA22 = fA22, FR_FracX, fA21 // C17*x + C16
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA16 = fA16, fB4, fA14 // (D3*x + D2)*x^2 + D1*x + D0
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPol = fA25, fB4, fPol // C20*x^2 + C19*x + C18
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA2L = fA4L, fB4, fA2L //  A4L*<x^2> + A4*d(x^2) + A2L
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA1L = fA3L, fB4, fA1L //  A3L*<x^2> + A3*d(x^2) + A1L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes4L = fA2, fRes2H // d1
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fResH = fRes2H, fB4, f0 // (A4*<x^2> + A2)*x^2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes1L = fA1, fRes3H // d1
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB6 = fRes3H, FR_FracX, f0 // (A3*<x^2> + A1)*x
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA10 = fA10, FR_FracX, fA9  // E5*x + E4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA8 = fA8, FR_FracX, fA7 // E3*x + E2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // (C20*x^2 + C19*x + C18)*x^2 + C17*x + C16
      fma.s1             fPol = fPol, fB4, fA22
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA6 = fA6, FR_FracX, fA5 // E1*x + E0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // A4L*<x^2> + A4*d(x^2) + A2L + delta(A4*<x^2>)
      fadd.s1            fRes2L = fA2L, fRes2L
      nop.i              0
}
{ .mfi
      nop.m              0
      // A3L*<x^2> + A3*d(x^2) + A1L + delta(A3*<x^2>)
      fadd.s1            fRes3L = fA1L, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fRes4H // d2
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fResL = fRes2H, fB4, fResH  // d(A4*<x^2> + A2)*x^2)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes1L = fRes1L, fRes1H // d2
      nop.i              0
}
{ .mfi
      nop.m              0
      fms.s1             fB8 = fRes3H, FR_FracX, fB6 // d((A3*<x^2> + A1)*x)
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fB10 = fResH, fB6 // (A4*x^4 + .. + A1*x)hi
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA12 = fA12, fB4, fA10 // Ehi
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // ((D7*x + D6)*x^2 + D5*x + D4)*x^4 + (D3*x + D2)*x^2 + D1*x + D0
      fma.s1             fA20 = fA20, fA5L, fA16
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA8 = fA8, fB4, fA6 // Elo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes2L = fRes2L, fRes4L // (A4*<x^2> + A2)lo
      nop.i              0
}
{ .mfi
      nop.m              0
      // d(A4*<x^2> + A2)*x^2) + A4*<x^2> + A2)*d(x^2)
      fma.s1             fResL = fRes2H, fB2, fResL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fRes1L   // (A4*<x^2> + A2)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fB12 = fB6, fB10
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, fA0, fA20 // PolC*x^8 + PolD
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fPolL = fA12, fA5L, fA8 // E
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fResL = fB4, fRes2L, fResL // ((A4*<x^2> + A2)*x^2)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes3L = fRes3L, FR_FracX, fB8 // ((A3*<x^2> + A1)*x)lo
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fB12 = fB12, fResH
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fPol = fPol, fA0, fPolL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fRes3L, fResL
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes2H = fPol, fA0L, fB10
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes3L = fB12, fRes3L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fsub.s1            fRes4L = fB10, fRes2H
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRes4L = fPol, fA0L, fRes4L
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fadd.s1            fRes4L = fRes4L, fRes3L
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for all paths for which the result is Pol24(x)
      fma.s0             f8 = fRes2H, f1, fRes4L
      // here is the exit for all paths for which the result is Pol24(x)
      br.ret.sptk        b0
}
;;


//  here if x is natval, nan, +/-inf, +/-0, or denormal
.align 32
lgammal_spec:
{ .mfi
      nop.m              0
      fclass.m           p9, p0 =  f8, 0xB // +/-denormals
      nop.i              0
};;
{ .mfi
      nop.m              0
      fclass.m           p6, p0 =  f8, 0x1E1 // Test x for natval, nan, +inf
      nop.i              0
};;
{ .mfb
      nop.m              0
      fclass.m           p7, p0 =  f8, 0x7 // +/-0
(p9)  br.cond.sptk       lgammal_denormal_input
};;
{ .mfb
      nop.m              0
      nop.f              0
      // branch out if x is natval, nan, +inf
(p6)  br.cond.spnt       lgammal_nan_pinf
};;
{ .mfb
      nop.m              0
      nop.f              0
(p7)  br.cond.spnt       lgammal_singularity
};;
// if we are still here then x = -inf
{ .mfi
      cmp.eq             p6, p7 = 4, rSgnGamSize
      nop.f              0
      adds               rSgnGam = 1, r0
};;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      nop.f              0
      nop.i              0
}
{ .mfb
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s0             f8 = f8,f8,f0 // return +inf, no call to error support
      br.ret.spnt        b0
};;

// here if x is NaN, NatVal or +INF
.align 32
lgammal_nan_pinf:
{ .mfi
      cmp.eq             p6, p7 = 4, rSgnGamSize
      nop.f              0
      adds               rSgnGam = 1, r0
}
;;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      fma.s0             f8 = f8,f1,f8 // return x+x if x is natval, nan, +inf
      nop.i              0
}
{ .mfb
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      nop.f              0
      br.ret.sptk        b0
}
;;

// here if x denormal or unnormal
.align 32
lgammal_denormal_input:
{ .mfi
      nop.m              0
      fma.s0             fResH = f1, f1, f8 // raise denormal exception
      nop.i              0
}
{ .mfi
      nop.m              0
      fnorm.s1           f8 = f8 // normalize input value
      nop.i              0
}
;;
{ .mfi
      getf.sig           rSignifX = f8
      fmerge.se          fSignifX =  f1, f8
      nop.i              0
}
{ .mfi
      getf.exp           rSignExpX = f8
      fcvt.fx.s1         fXint = f8 // Convert arg to int (int repres. in FR)
      nop.i              0
}
;;
{ .mfi
      getf.exp           rSignExpX = f8
      fcmp.lt.s1         p15, p14 = f8, f0
      nop.i              0
}
;;
{ .mfb
      and                rExpX = rSignExpX, r17Ones
      fmerge.s           fAbsX = f1, f8 // |x|
      br.cond.sptk       _deno_back_to_main_path
}
;;


// here if overflow (x > overflow_bound)
.align 32
lgammal_overflow:
{ .mfi
      addl               r8 = 0x1FFFE, r0
      nop.f              0
      cmp.eq             p6, p7 = 4, rSgnGamSize
}
{ .mfi
      adds               rSgnGam = 1, r0
      nop.f              0
      nop.i              0
}
;;
{ .mfi
      setf.exp           f9 = r8
      fmerge.s           FR_X = f8,f8
      mov                GR_Parameter_TAG = 102 // overflow
};;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      nop.f              0
      nop.i              0
}
{ .mfb
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      fma.s0             FR_RESULT = f9,f9,f0 // Set I,O and +INF result
      br.cond.sptk       __libm_error_region
};;

// here if x is negative integer or +/-0 (SINGULARITY)
.align 32
lgammal_singularity:
{ .mfi
      adds               rSgnGam = 1, r0
      fclass.m           p8,p0 = f8,0x6 // is x -0?
      mov                GR_Parameter_TAG = 103 // negative
}
{ .mfi
      cmp.eq             p6, p7 = 4, rSgnGamSize
      fma.s1             FR_X = f0,f0,f8
      nop.i              0
};;
{ .mfi
(p8)  sub                rSgnGam = r0, rSgnGam
      nop.f              0
      nop.i              0
}
{ .mfi
      nop.m              0
      nop.f              0
      nop.i              0
};;
{ .mfi
      // store signgam if size of variable is 4 bytes
(p6)  st4                [rSgnGamAddr] = rSgnGam
      nop.f              0
      nop.i              0
}
{ .mfb
      // store signgam if size of variable is 8 bytes
(p7)  st8                [rSgnGamAddr] = rSgnGam
      frcpa.s0           FR_RESULT, p0 = f1, f0
      br.cond.sptk       __libm_error_region
};;

GLOBAL_LIBM_END(__libm_lgammal)



LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                           // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                       // Save gp
};;
{ .mmi
        stfe [GR_Parameter_Y] = FR_Y,16         // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfe [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfe [GR_Parameter_Y] = FR_RESULT      // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 999
        nop.i 999
};;
{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]       // Get return result off stack
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
        mov   b0 = GR_SAVE_B0                  // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
};;

LOCAL_LIBM_END(__libm_error_region#)

.type   __libm_error_support#,@function
.global __libm_error_support#
