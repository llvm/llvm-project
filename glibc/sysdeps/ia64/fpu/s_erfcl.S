.file "erfcl.s"


// Copyright (c) 2001 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2001 by the Intel Numerics Group, Intel Corporation
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
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//
// History
//==============================================================
// 11/12/01  Initial version
// 02/08/02  Added missing }
// 05/20/02  Cleaned up namespace and sf0 syntax
// 02/10/03  Reordered header: .section, .global, .proc, .align;
//           used data8 for long double table values
// 03/31/05  Reformatted delimiters between data tables
//
// API
//==============================================================
// long double erfcl(long double)
//
// Implementation and Algorithm Notes:
//==============================================================
// 1. 0 <= x <= 107.0
//
//    erfcl(x) ~=~ P15(z) * expl( -x^2 )/(dx + x), z = x - xc(i).
//
//    Comment:
//
//    Let x(i) = -1.0 + 2^(i/4),i=0,...27. So we have 28 unequal
//    argument intervals [x(i),x(i+1)] with length ratio q = 2^(1/4).
//    Values xc(i) we have in the table erfc_xc_table,xc(i)=x(i)for i = 0
//    and xc(i)= 0.5*( x(i)+x(i+1) ) for i>0.
//
//    Let x(i)<= x < x(i+1).
//    We can find i as exponent of number (x + 1)^4.
//
//    Let P15(z)= a0+ a1*z +..+a15*z^15 - polynomial approximation of degree 15
//    for function      erfcl(z+xc(i)) * expl( (z+xc(i))^2)* (dx+z+xc(i)) and
//    -0.5*[x(i+1)-x(i)] <= z <= 0.5*[x(i+1)-x(i)].
//
//    Let  Q(z)= (P(z)- S)/S, S = a0, rounded to 16 bits.
//    Polynomial coefficients for Q(z) we have in the table erfc_Q_table as
//    long double values
//
//    We use multi precision to calculate input argument -x^2 for expl and
//    for u = 1/(dx + x).
//
//    Algorithm description for expl function see below. In accordance with
//    denotation of this algorithm we have for expl:
//
//    expl(X) ~=~ 2^K*T_1*(1+W_1)*T_2*(1+W_2)*(1+ poly(r)), X = -x^2.
//
//    Final calculations for erfcl:
//
//    erfcl(x) ~=~
//
//         2^K*T_1*(1+W_1)*T_2*(1+W_2)*(1+ poly(r))*(1-dy)*S*(1+Q(z))*u*(1+du),
//
//    where dy - low bits of x^2 and u, u*du - hi and low bits of 1/(dx + x).
//
//    The order of calculations is the next:
//
//    1)  M = 2^K*T_1*T_2*S          without rounding error,
//    2)  W = W_1 + (W_2 + W_1*W_2), where 1+W  ~=~ (1+W_1)(1+W_2),
//    3)  H = W - dy,                where 1+H  ~=~ (1+W )(1-dy),
//    4)  R = poly(r)*H + poly(r),
//    5)  R = H + R              ,   where 1+R  ~=~ (1+H )(1+poly(r)),
//    6)  G = Q(z)*R + Q(z),
//    7)  R1 = R + du,               where 1+R1 ~=~ (1+R)(1+du),
//    8)  G1 = R1 + G,               where 1+G1 ~=~ (1+R1)(1+Q(z)),
//    9)  V  = G1*M*u,
//    10) erfcl(x) ~=~ M*u + V
//
// 2. -6.5 <= x < 0
//
//    erfcl(x)  = 2.0 - erfl(-x)
//
// 3. x > 107.0
//    erfcl(x)  ~=~ 0.0
//
// 4. x < -6.5
//    erfcl(x)  ~=~ 2.0

// Special values
//==============================================================
// erfcl(+0)    = 1.0
// erfcl(-0)    = 1.0

// erfcl(+qnan) = +qnan
// erfcl(-qnan) = -qnan
// erfcl(+snan) = +qnan
// erfcl(-snan) = -qnan

// erfcl(-inf)  = 2.0
// erfcl(+inf)  = +0

//==============================================================
// Algorithm description of used expl function.
//
// Implementation and Algorithm Notes:
//
//  ker_exp_64( in_FR  : X,
//            out_FR : Y_hi,
//            out_FR : Y_lo,
//            out_FR : scale,
//            out_PR : Safe )
//
// On input, X is in register format
//
// On output,
//
//   scale*(Y_hi + Y_lo)  approximates  exp(X)
//
// The accuracy is sufficient for a highly accurate 64 sig.
// bit implementation.  Safe is set if there is no danger of
// overflow/underflow when the result is composed from scale,
// Y_hi and Y_lo. Thus, we can have a fast return if Safe is set.
// Otherwise, one must prepare to handle the possible exception
// appropriately.  Note that SAFE not set (false) does not mean
// that overflow/underflow will occur; only the setting of SAFE
// guarantees the opposite.
//
// **** High Level Overview ****
//
// The method consists of three cases.
//
// If           |X| < Tiny  use case exp_tiny;
// else if  |X| < 2^(-6)    use case exp_small;
// else     use case exp_regular;
//
// Case exp_tiny:
//
//   1 + X     can be used to approximate exp(X)
//   X + X^2/2 can be used to approximate exp(X) - 1
//
// Case exp_small:
//
//   Here, exp(X) and exp(X) - 1 can all be
//   approximated by a relatively simple polynomial.
//
//   This polynomial resembles the truncated Taylor series
//
//  exp(w) = 1 + w + w^2/2! + w^3/3! + ... + w^n/n!
//
// Case exp_regular:
//
//   Here we use a table lookup method. The basic idea is that in
//   order to compute exp(X), we accurately decompose X into
//
//   X = N * log(2)/(2^12)  + r,    |r| <= log(2)/2^13.
//
//   Hence
//
//   exp(X) = 2^( N / 2^12 ) * exp(r).
//
//   The value 2^( N / 2^12 ) is obtained by simple combinations
//   of values calculated beforehand and stored in table; exp(r)
//   is approximated by a short polynomial because |r| is small.
//
//   We elaborate this method in 4 steps.
//
//   Step 1: Reduction
//
//   The value 2^12/log(2) is stored as a double-extended number
//   L_Inv.
//
//   N := round_to_nearest_integer( X * L_Inv )
//
//   The value log(2)/2^12 is stored as two numbers L_hi and L_lo so
//   that r can be computed accurately via
//
//   r := (X - N*L_hi) - N*L_lo
//
//   We pick L_hi such that N*L_hi is representable in 64 sig. bits
//   and thus the FMA   X - N*L_hi   is error free. So r is the
//   1 rounding error from an exact reduction with respect to
//
//   L_hi + L_lo.
//
//   In particular, L_hi has 30 significant bit and can be stored
//   as a double-precision number; L_lo has 64 significant bits and
//   stored as a double-extended number.
//
//   Step 2: Approximation
//
//   exp(r) - 1 is approximated by a short polynomial of the form
//
//   r + A_1 r^2 + A_2 r^3 + A_3 r^4 .
//
//   Step 3: Composition from Table Values
//
//   The value 2^( N / 2^12 ) can be composed from a couple of tables
//   of precalculated values. First, express N as three integers
//   K, M_1, and M_2 as
//
//     N  =  K * 2^12  + M_1 * 2^6 + M_2
//
//   Where 0 <= M_1, M_2 < 2^6; and K can be positive or negative.
//   When N is represented in 2's complement, M_2 is simply the 6
//   lsb's, M_1 is the next 6, and K is simply N shifted right
//   arithmetically (sign extended) by 12 bits.
//
//   Now, 2^( N / 2^12 ) is simply
//
//      2^K * 2^( M_1 / 2^6 ) * 2^( M_2 / 2^12 )
//
//   Clearly, 2^K needs no tabulation. The other two values are less
//   trivial because if we store each accurately to more than working
//   precision, than its product is too expensive to calculate. We
//   use the following method.
//
//   Define two mathematical values, delta_1 and delta_2, implicitly
//   such that
//
//     T_1 = exp( [M_1 log(2)/2^6]  -  delta_1 )
//     T_2 = exp( [M_2 log(2)/2^12] -  delta_2 )
//
//   are representable as 24 significant bits. To illustrate the idea,
//   we show how we define delta_1:
//
//     T_1     := round_to_24_bits( exp( M_1 log(2)/2^6 ) )
//     delta_1  = (M_1 log(2)/2^6) - log( T_1 )
//
//   The last equality means mathematical equality. We then tabulate
//
//     W_1 := exp(delta_1) - 1
//     W_2 := exp(delta_2) - 1
//
//   Both in double precision.
//
//   From the tabulated values T_1, T_2, W_1, W_2, we compose the values
//   T and W via
//
//     T := T_1 * T_2           ...exactly
//     W := W_1 + (1 + W_1)*W_2
//
//   W approximates exp( delta ) - 1  where delta = delta_1 + delta_2.
//   The mathematical product of T and (W+1) is an accurate representation
//   of 2^(M_1/2^6) * 2^(M_2/2^12).
//
//   Step 4. Reconstruction
//
//   Finally, we can reconstruct exp(X), exp(X) - 1.
//   Because
//
//  X = K * log(2) + (M_1*log(2)/2^6  - delta_1)
//             + (M_2*log(2)/2^12 - delta_2)
//             + delta_1 + delta_2 + r      ...accurately
//   We have
//
//  exp(X) ~=~ 2^K * ( T + T*[exp(delta_1+delta_2+r) - 1] )
//         ~=~ 2^K * ( T + T*[exp(delta + r) - 1]         )
//         ~=~ 2^K * ( T + T*[(exp(delta)-1)
//               + exp(delta)*(exp(r)-1)]   )
//             ~=~ 2^K * ( T + T*( W + (1+W)*poly(r) ) )
//             ~=~ 2^K * ( Y_hi  +  Y_lo )
//
//   where Y_hi = T  and Y_lo = T*(W + (1+W)*poly(r))
//
//   For exp(X)-1, we have
//
//  exp(X)-1 ~=~ 2^K * ( Y_hi + Y_lo ) - 1
//       ~=~ 2^K * ( Y_hi + Y_lo - 2^(-K) )
//
//   and we combine Y_hi + Y_lo - 2^(-N)  into the form of two
//   numbers  Y_hi + Y_lo carefully.
//
//   **** Algorithm Details ****
//
//   A careful algorithm must be used to realize the mathematical ideas
//   accurately. We describe each of the three cases. We assume SAFE
//   is preset to be TRUE.
//
//   Case exp_tiny:
//
//   The important points are to ensure an accurate result under
//   different rounding directions and a correct setting of the SAFE
//   flag.
//
//   If expm1 is 1, then
//      SAFE  := False  ...possibility of underflow
//      Scale := 1.0
//      Y_hi  := X
//      Y_lo  := 2^(-17000)
//   Else
//      Scale := 1.0
//      Y_hi  := 1.0
//      Y_lo  := X  ...for different rounding modes
//   Endif
//
//   Case exp_small:
//
//   Here we compute a simple polynomial. To exploit parallelism, we split
//   the polynomial into several portions.
//
//   Let r = X
//
//   If exp     ...i.e. exp( argument )
//
//      rsq := r * r;
//      r4  := rsq*rsq
//      poly_lo := P_3 + r*(P_4 + r*(P_5 + r*P_6))
//      poly_hi := r + rsq*(P_1 + r*P_2)
//      Y_lo    := poly_hi + r4 * poly_lo
//      Y_hi    := 1.0
//      Scale   := 1.0
//
//   Else           ...i.e. exp( argument ) - 1
//
//      rsq := r * r
//      r4  := rsq * rsq
//      r6  := rsq * r4
//      poly_lo := r6*(Q_5 + r*(Q_6 + r*Q_7))
//      poly_hi := Q_1 + r*(Q_2 + r*(Q_3 + r*Q_4))
//      Y_lo    := rsq*poly_hi +  poly_lo
//      Y_hi    := X
//      Scale   := 1.0
//
//   Endif
//
//  Case exp_regular:
//
//  The previous description contain enough information except the
//  computation of poly and the final Y_hi and Y_lo in the case for
//  exp(X)-1.
//
//  The computation of poly for Step 2:
//
//   rsq := r*r
//   poly := r + rsq*(A_1 + r*(A_2 + r*A_3))
//
//  For the case exp(X) - 1, we need to incorporate 2^(-K) into
//  Y_hi and Y_lo at the end of Step 4.
//
//   If K > 10 then
//      Y_lo := Y_lo - 2^(-K)
//   Else
//      If K < -10 then
//   Y_lo := Y_hi + Y_lo
//   Y_hi := -2^(-K)
//      Else
//   Y_hi := Y_hi - 2^(-K)
//      End If
//   End If
//

// Overview of operation
//==============================================================

// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f9 -> f14,  f36 -> f126

// General registers used:
// r32 -> r71

// Predicate registers used:
// p6 -> p15

// Assembly macros
//==============================================================
// GR for exp(X)
GR_ad_Arg           = r33
GR_ad_C             = r34
GR_ERFC_S_TB        = r35
GR_signexp_x        = r36
GR_exp_x            = r36
GR_exp_mask         = r37
GR_ad_W1            = r38
GR_ad_W2            = r39
GR_M2               = r40
GR_M1               = r41
GR_K                = r42
GR_exp_2_k          = r43
GR_ad_T1            = r44
GR_ad_T2            = r45
GR_N_fix            = r46
GR_ad_P             = r47
GR_exp_bias         = r48
GR_BIAS             = r48
GR_exp_half         = r49
GR_sig_inv_ln2      = r50
GR_rshf_2to51       = r51
GR_exp_2tom51       = r52
GR_rshf             = r53

// GR for erfcl(x)
//==============================================================

GR_ERFC_XC_TB       = r54
GR_ERFC_P_TB        = r55
GR_IndxPlusBias     = r56
GR_P_POINT_1        = r57
GR_P_POINT_2        = r58
GR_AbsArg           = r59
GR_ShftXBi          = r60
GR_ShftPi           = r61
GR_mBIAS            = r62
GR_ShftPi_bias      = r63
GR_ShftXBi_bias     = r64
GR_ShftA14          = r65
GR_ShftA15          = r66
GR_EpsNorm          = r67
GR_0x1              = r68
GR_ShftPi_8         = r69
GR_26PlusBias       = r70
GR_27PlusBias       = r71

// GR for __libm_support call
//==============================================================
GR_SAVE_B0          = r64
GR_SAVE_PFS         = r65
GR_SAVE_GP          = r66
GR_SAVE_SP          = r67

GR_Parameter_X      = r68
GR_Parameter_Y      = r69
GR_Parameter_RESULT = r70
GR_Parameter_TAG    = r71

//==============================================================
// Floating Point Registers
//
FR_RSHF_2TO51       = f10
FR_INV_LN2_2TO63    = f11
FR_W_2TO51_RSH      = f12
FR_2TOM51           = f13
FR_RSHF             = f14

FR_scale            = f36
FR_float_N          = f37
FR_N_signif         = f38
FR_L_hi             = f39
FR_L_lo             = f40
FR_r                = f41
FR_W1               = f42
FR_T1               = f43
FR_W2               = f44
FR_T2               = f45
FR_rsq              = f46
FR_C2               = f47
FR_C3               = f48
FR_poly             = f49
FR_P6               = f49
FR_T                = f50
FR_P5               = f50
FR_P4               = f51
FR_W                = f51
FR_P3               = f52
FR_Wp1              = f52
FR_P2               = f53
FR_P1               = f54
FR_Q7               = f56
FR_Q6               = f57
FR_Q5               = f58
FR_Q4               = f59
FR_Q3               = f60
FR_Q2               = f61
FR_Q1               = f62
FR_C1               = f63
FR_A15              = f64
FR_ch_dx            = f65
FR_T_scale          = f66
FR_norm_x           = f67
FR_AbsArg           = f68
FR_POS_ARG_ASYMP    = f69
FR_NEG_ARG_ASYMP    = f70
FR_Tmp              = f71
FR_Xc               = f72
FR_A0               = f73
FR_A1               = f74
FR_A2               = f75
FR_A3               = f76
FR_A4               = f77
FR_A5               = f78
FR_A6               = f79
FR_A7               = f80
FR_A8               = f81
FR_A9               = f82
FR_A10              = f83
FR_A11              = f84
FR_A12              = f85
FR_A13              = f86
FR_A14              = f87
FR_P15_0_1          = f88
FR_P15_8_1          = f88
FR_P15_1_1          = f89
FR_P15_8_2          = f89
FR_P15_1_2          = f90
FR_P15_2_1          = f91
FR_P15_2_2          = f92
FR_P15_3_1          = f93
FR_P15_3_2          = f94
FR_P15_4_2          = f95
FR_P15_7_1          = f96
FR_P15_7_2          = f97
FR_P15_9_1          = f98
FR_P15_9_2          = f99
FR_P15_13_1         = f100
FR_P15_14_1         = f101
FR_P15_14_2         = f102
FR_Tmp2             = f103
FR_Xpdx_lo          = f104
FR_2                = f105
FR_xsq_lo           = f106
FR_LocArg           = f107
FR_Tmpf             = f108
FR_Tmp1             = f109
FR_EpsNorm          = f110
FR_UnfBound         = f111
FR_NormX            = f112
FR_Xpdx_hi          = f113
FR_dU               = f114
FR_H                = f115
FR_G                = f116
FR_V                = f117
FR_M                = f118
FR_U                = f119
FR_Q                = f120
FR_S                = f121
FR_R                = f122
FR_res_pos_x_hi     = f123
FR_res_pos_x_lo     = f124
FR_dx               = f125
FR_dx1              = f126

// for error handler routine
FR_X                = f9
FR_Y                = f0
FR_RESULT           = f8

// Data tables
//==============================================================
RODATA
.align 16

// ************* DO NOT CHANGE ORDER OF THESE TABLES ********************
LOCAL_OBJECT_START(exp_table_1)

data8 0xae89f995ad3ad5ea , 0x00003ffe      // x = 0.681..,bound for dx = 0.875
data8 0x405AC00000000000 , 0x401A000000000000        //ARG_ASYMP,NEG_ARG_ASYMP
data8 0x3FE4000000000000 , 0x3FEC000000000000                    //0.625,0.875
data8 0xD5126065B720A4e9 , 0x00004005                    // underflow boundary
data8 0x8000000000000000 , 0x00000001                             //FR_EpsNorm
LOCAL_OBJECT_END(exp_table_1)

LOCAL_OBJECT_START(Constants_exp_64_Arg)
data8 0xB17217F400000000,0x00003FF2 //L_hi = hi part log(2)/2^12
data8 0xF473DE6AF278ECE6,0x00003FD4 //L_lo = lo part log(2)/2^12
LOCAL_OBJECT_END(Constants_exp_64_Arg)

LOCAL_OBJECT_START(Constants_exp_64_C)
data8 0xAAAAAAABB1B736A0,0x00003FFA // C3
data8 0xAAAAAAAB90CD6327,0x00003FFC // C2
data8 0xFFFFFFFFFFFFFFFF,0x00003FFD // C1
LOCAL_OBJECT_END(Constants_exp_64_C)

LOCAL_OBJECT_START(Constants_exp_64_T1)
data4 0x3F800000,0x3F8164D2,0x3F82CD87,0x3F843A29
data4 0x3F85AAC3,0x3F871F62,0x3F88980F,0x3F8A14D5
data4 0x3F8B95C2,0x3F8D1ADF,0x3F8EA43A,0x3F9031DC
data4 0x3F91C3D3,0x3F935A2B,0x3F94F4F0,0x3F96942D
data4 0x3F9837F0,0x3F99E046,0x3F9B8D3A,0x3F9D3EDA
data4 0x3F9EF532,0x3FA0B051,0x3FA27043,0x3FA43516
data4 0x3FA5FED7,0x3FA7CD94,0x3FA9A15B,0x3FAB7A3A
data4 0x3FAD583F,0x3FAF3B79,0x3FB123F6,0x3FB311C4
data4 0x3FB504F3,0x3FB6FD92,0x3FB8FBAF,0x3FBAFF5B
data4 0x3FBD08A4,0x3FBF179A,0x3FC12C4D,0x3FC346CD
data4 0x3FC5672A,0x3FC78D75,0x3FC9B9BE,0x3FCBEC15
data4 0x3FCE248C,0x3FD06334,0x3FD2A81E,0x3FD4F35B
data4 0x3FD744FD,0x3FD99D16,0x3FDBFBB8,0x3FDE60F5
data4 0x3FE0CCDF,0x3FE33F89,0x3FE5B907,0x3FE8396A
data4 0x3FEAC0C7,0x3FED4F30,0x3FEFE4BA,0x3FF28177
data4 0x3FF5257D,0x3FF7D0DF,0x3FFA83B3,0x3FFD3E0C
LOCAL_OBJECT_END(Constants_exp_64_T1)

LOCAL_OBJECT_START(Constants_exp_64_T2)
data4 0x3F800000,0x3F80058C,0x3F800B18,0x3F8010A4
data4 0x3F801630,0x3F801BBD,0x3F80214A,0x3F8026D7
data4 0x3F802C64,0x3F8031F2,0x3F803780,0x3F803D0E
data4 0x3F80429C,0x3F80482B,0x3F804DB9,0x3F805349
data4 0x3F8058D8,0x3F805E67,0x3F8063F7,0x3F806987
data4 0x3F806F17,0x3F8074A8,0x3F807A39,0x3F807FCA
data4 0x3F80855B,0x3F808AEC,0x3F80907E,0x3F809610
data4 0x3F809BA2,0x3F80A135,0x3F80A6C7,0x3F80AC5A
data4 0x3F80B1ED,0x3F80B781,0x3F80BD14,0x3F80C2A8
data4 0x3F80C83C,0x3F80CDD1,0x3F80D365,0x3F80D8FA
data4 0x3F80DE8F,0x3F80E425,0x3F80E9BA,0x3F80EF50
data4 0x3F80F4E6,0x3F80FA7C,0x3F810013,0x3F8105AA
data4 0x3F810B41,0x3F8110D8,0x3F81166F,0x3F811C07
data4 0x3F81219F,0x3F812737,0x3F812CD0,0x3F813269
data4 0x3F813802,0x3F813D9B,0x3F814334,0x3F8148CE
data4 0x3F814E68,0x3F815402,0x3F81599C,0x3F815F37
LOCAL_OBJECT_END(Constants_exp_64_T2)

LOCAL_OBJECT_START(Constants_exp_64_W1)
data8 0x0000000000000000, 0xBE384454171EC4B4
data8 0xBE6947414AA72766, 0xBE5D32B6D42518F8
data8 0x3E68D96D3A319149, 0xBE68F4DA62415F36
data8 0xBE6DDA2FC9C86A3B, 0x3E6B2E50F49228FE
data8 0xBE49C0C21188B886, 0x3E64BFC21A4C2F1F
data8 0xBE6A2FBB2CB98B54, 0x3E5DC5DE9A55D329
data8 0x3E69649039A7AACE, 0x3E54728B5C66DBA5
data8 0xBE62B0DBBA1C7D7D, 0x3E576E0409F1AF5F
data8 0x3E6125001A0DD6A1, 0xBE66A419795FBDEF
data8 0xBE5CDE8CE1BD41FC, 0xBE621376EA54964F
data8 0x3E6370BE476E76EE, 0x3E390D1A3427EB92
data8 0x3E1336DE2BF82BF8, 0xBE5FF1CBD0F7BD9E
data8 0xBE60A3550CEB09DD, 0xBE5CA37E0980F30D
data8 0xBE5C541B4C082D25, 0xBE5BBECA3B467D29
data8 0xBE400D8AB9D946C5, 0xBE5E2A0807ED374A
data8 0xBE66CB28365C8B0A, 0x3E3AAD5BD3403BCA
data8 0x3E526055C7EA21E0, 0xBE442C75E72880D6
data8 0x3E58B2BB85222A43, 0xBE5AAB79522C42BF
data8 0xBE605CB4469DC2BC, 0xBE589FA7A48C40DC
data8 0xBE51C2141AA42614, 0xBE48D087C37293F4
data8 0x3E367A1CA2D673E0, 0xBE51BEBB114F7A38
data8 0xBE6348E5661A4B48, 0xBDF526431D3B9962
data8 0x3E3A3B5E35A78A53, 0xBE46C46C1CECD788
data8 0xBE60B7EC7857D689, 0xBE594D3DD14F1AD7
data8 0xBE4F9C304C9A8F60, 0xBE52187302DFF9D2
data8 0xBE5E4C8855E6D68F, 0xBE62140F667F3DC4
data8 0xBE36961B3BF88747, 0x3E602861C96EC6AA
data8 0xBE3B5151D57FD718, 0x3E561CD0FC4A627B
data8 0xBE3A5217CA913FEA, 0x3E40A3CC9A5D193A
data8 0xBE5AB71310A9C312, 0x3E4FDADBC5F57719
data8 0x3E361428DBDF59D5, 0x3E5DB5DB61B4180D
data8 0xBE42AD5F7408D856, 0x3E2A314831B2B707
LOCAL_OBJECT_END(Constants_exp_64_W1)

LOCAL_OBJECT_START(Constants_exp_64_W2)
data8 0x0000000000000000, 0xBE641F2537A3D7A2
data8 0xBE68DD57AD028C40, 0xBE5C77D8F212B1B6
data8 0x3E57878F1BA5B070, 0xBE55A36A2ECAE6FE
data8 0xBE620608569DFA3B, 0xBE53B50EA6D300A3
data8 0x3E5B5EF2223F8F2C, 0xBE56A0D9D6DE0DF4
data8 0xBE64EEF3EAE28F51, 0xBE5E5AE2367EA80B
data8 0x3E47CB1A5FCBC02D, 0xBE656BA09BDAFEB7
data8 0x3E6E70C6805AFEE7, 0xBE6E0509A3415EBA
data8 0xBE56856B49BFF529, 0x3E66DD3300508651
data8 0x3E51165FC114BC13, 0x3E53333DC453290F
data8 0x3E6A072B05539FDA, 0xBE47CD877C0A7696
data8 0xBE668BF4EB05C6D9, 0xBE67C3E36AE86C93
data8 0xBE533904D0B3E84B, 0x3E63E8D9556B53CE
data8 0x3E212C8963A98DC8, 0xBE33138F032A7A22
data8 0x3E530FA9BC584008, 0xBE6ADF82CCB93C97
data8 0x3E5F91138370EA39, 0x3E5443A4FB6A05D8
data8 0x3E63DACD181FEE7A, 0xBE62B29DF0F67DEC
data8 0x3E65C4833DDE6307, 0x3E5BF030D40A24C1
data8 0x3E658B8F14E437BE, 0xBE631C29ED98B6C7
data8 0x3E6335D204CF7C71, 0x3E529EEDE954A79D
data8 0x3E5D9257F64A2FB8, 0xBE6BED1B854ED06C
data8 0x3E5096F6D71405CB, 0xBE3D4893ACB9FDF5
data8 0xBDFEB15801B68349, 0x3E628D35C6A463B9
data8 0xBE559725ADE45917, 0xBE68C29C042FC476
data8 0xBE67593B01E511FA, 0xBE4A4313398801ED
data8 0x3E699571DA7C3300, 0x3E5349BE08062A9E
data8 0x3E5229C4755BB28E, 0x3E67E42677A1F80D
data8 0xBE52B33F6B69C352, 0xBE6B3550084DA57F
data8 0xBE6DB03FD1D09A20, 0xBE60CBC42161B2C1
data8 0x3E56ED9C78A2B771, 0xBE508E319D0FA795
data8 0xBE59482AFD1A54E9, 0xBE2A17CEB07FD23E
data8 0x3E68BF5C17365712, 0x3E3956F9B3785569
LOCAL_OBJECT_END(Constants_exp_64_W2)


LOCAL_OBJECT_START(erfc_xc_table)

data8 0x0000000000000000, 0x00000000 //XC[0] = +0.00000000000000000000e-01L
data8 0x9A79C70000000000, 0x00003FFD //XC[1] = +3.01710337400436401367e-01L
data8 0x8C49EF0000000000, 0x00003FFE //XC[2] = +5.48003137111663818359e-01L
data8 0xD744FC0000000000, 0x00003FFE //XC[3] = +8.40896368026733398438e-01L
data8 0x9837F00000000000, 0x00003FFF //XC[4] = +1.18920707702636718750e+00L
data8 0xCD3CE30000000000, 0x00003FFF //XC[5] = +1.60342061519622802734e+00L
data8 0x8624F70000000000, 0x00004000 //XC[6] = +2.09600615501403808594e+00L
data8 0xABA27E0000000000, 0x00004000 //XC[7] = +2.68179273605346679688e+00L
data8 0xD837F00000000000, 0x00004000 //XC[8] = +3.37841415405273437500e+00L
data8 0x869E710000000000, 0x00004001 //XC[9] = +4.20684099197387695313e+00L
data8 0xA624F70000000000, 0x00004001 //XC[10] = +5.19201231002807617188e+00L
data8 0xCBA27E0000000000, 0x00004001 //XC[11] = +6.36358547210693359375e+00L
data8 0xF837F00000000000, 0x00004001 //XC[12] = +7.75682830810546875000e+00L
data8 0x969E710000000000, 0x00004002 //XC[13] = +9.41368198394775390625e+00L
data8 0xB624F70000000000, 0x00004002 //XC[14] = +1.13840246200561523438e+01L
data8 0xDBA27E0000000000, 0x00004002 //XC[15] = +1.37271709442138671875e+01L
data8 0x841BF80000000000, 0x00004003 //XC[16] = +1.65136566162109375000e+01L
data8 0x9E9E710000000000, 0x00004003 //XC[17] = +1.98273639678955078125e+01L
data8 0xBE24F70000000000, 0x00004003 //XC[18] = +2.37680492401123046875e+01L
data8 0xE3A27E0000000000, 0x00004003 //XC[19] = +2.84543418884277343750e+01L
data8 0x881BF80000000000, 0x00004004 //XC[20] = +3.40273132324218750000e+01L
data8 0xA29E710000000000, 0x00004004 //XC[21] = +4.06547279357910156250e+01L
data8 0xC224F70000000000, 0x00004004 //XC[22] = +4.85360984802246093750e+01L
data8 0xE7A27E0000000000, 0x00004004 //XC[23] = +5.79086837768554687500e+01L
data8 0x8A1BF80000000000, 0x00004005 //XC[24] = +6.90546264648437500000e+01L
data8 0xA49E710000000000, 0x00004005 //XC[25] = +8.23094558715820312500e+01L
data8 0xC424F70000000000, 0x00004005 //XC[26] = +9.80721969604492187500e+01L
data8 0xD5A27E0000000000, 0x00004005 //XC[27] = +1.06817367553710937500e+02L
LOCAL_OBJECT_END(erfc_xc_table)

LOCAL_OBJECT_START(erfc_s_table)

data8 0xE000000000000000, 0x00003FFE //s[0] = +8.75000000000000000000e-01L
data8 0xDCEF000000000000, 0x00003FFE //s[1] = +8.63021850585937500000e-01L
data8 0xD79D000000000000, 0x00003FFE //s[2] = +8.42239379882812500000e-01L
data8 0xB25E000000000000, 0x00003FFE //s[3] = +6.96746826171875000000e-01L
data8 0xB0EA000000000000, 0x00003FFE //s[4] = +6.91070556640625000000e-01L
data8 0xAE3F000000000000, 0x00003FFE //s[5] = +6.80648803710937500000e-01L
data8 0xAB05000000000000, 0x00003FFE //s[6] = +6.68045043945312500000e-01L
data8 0xA7AC000000000000, 0x00003FFE //s[7] = +6.54968261718750000000e-01L
data8 0xA478000000000000, 0x00003FFE //s[8] = +6.42456054687500000000e-01L
data8 0xA18D000000000000, 0x00003FFE //s[9] = +6.31057739257812500000e-01L
data8 0x9EF8000000000000, 0x00003FFE //s[10] = +6.20971679687500000000e-01L
data8 0x9CBA000000000000, 0x00003FFE //s[11] = +6.12213134765625000000e-01L
data8 0x9ACD000000000000, 0x00003FFE //s[12] = +6.04690551757812500000e-01L
data8 0x992A000000000000, 0x00003FFE //s[13] = +5.98297119140625000000e-01L
data8 0x97C7000000000000, 0x00003FFE //s[14] = +5.92880249023437500000e-01L
data8 0x969C000000000000, 0x00003FFE //s[15] = +5.88317871093750000000e-01L
data8 0x95A0000000000000, 0x00003FFE //s[16] = +5.84472656250000000000e-01L
data8 0x94CB000000000000, 0x00003FFE //s[17] = +5.81222534179687500000e-01L
data8 0x9419000000000000, 0x00003FFE //s[18] = +5.78506469726562500000e-01L
data8 0x9383000000000000, 0x00003FFE //s[19] = +5.76217651367187500000e-01L
data8 0x9305000000000000, 0x00003FFE //s[20] = +5.74295043945312500000e-01L
data8 0x929B000000000000, 0x00003FFE //s[21] = +5.72677612304687500000e-01L
data8 0x9242000000000000, 0x00003FFE //s[22] = +5.71319580078125000000e-01L
data8 0x91F8000000000000, 0x00003FFE //s[23] = +5.70190429687500000000e-01L
data8 0x91B9000000000000, 0x00003FFE //s[24] = +5.69229125976562500000e-01L
data8 0x9184000000000000, 0x00003FFE //s[25] = +5.68420410156250000000e-01L
data8 0x9158000000000000, 0x00003FFE //s[26] = +5.67749023437500000000e-01L
data8 0x9145000000000000, 0x00003FFE //s[27] = +5.67459106445312500000e-01L
LOCAL_OBJECT_END(erfc_s_table)

LOCAL_OBJECT_START(erfc_Q_table)
// Q(z)= (P(z)- S)/S
//
// Pol0
data8 0x98325D50F9DC3499, 0x0000BFAA //A0 = +3.07358861423101280650e-26L
data8 0xED35081A2494DDD9, 0x00003FF8 //A1 = +1.44779757616302832466e-02L
data8 0x9443549BCD0F94CE, 0x0000BFFD //A2 = -2.89576190966300084405e-01L
data8 0xC7FD4B98ECF3DBBF, 0x00003FFD //A3 = +3.90604364793467799170e-01L
data8 0xB82CE31288B49759, 0x0000BFFD //A4 = -3.59717460644199233866e-01L
data8 0x8A8293447BEF69B5, 0x00003FFD //A5 = +2.70527460203054582368e-01L
data8 0xB5793E30EE36766C, 0x0000BFFC //A6 = -1.77220317589265674647e-01L
data8 0xD6066D16BBDECE17, 0x00003FFB //A7 = +1.04504444366724593714e-01L
data8 0xE7C783CE3C997BD8, 0x0000BFFA //A8 = -5.65867565781331646771e-02L
data8 0xE9969EBC2F5B2828, 0x00003FF9 //A9 = +2.85142040533900194955e-02L
data8 0xDD31D619F29AD7BF, 0x0000BFF8 //A10 = -1.35006514390540367929e-02L
data8 0xC63A20EB59768F3A, 0x00003FF7 //A11 = +6.04940993680332271481e-03L
data8 0xA8DEC641AACEB600, 0x0000BFF6 //A12 = -2.57675495383156581601e-03L
data8 0x87F0E77BA914FBEB, 0x00003FF5 //A13 = +1.03714776726541296794e-03L
data8 0xC306C2894C5CEF2D, 0x0000BFF3 //A14 = -3.71983348634136412407e-04L
data8 0xBDAB416A989D0697, 0x00003FF1 //A15 = +9.04412111877987292294e-05L
// Pol1
data8 0x82808893DA2DD83F, 0x00003FEE //A0 = +7.77853035974467145290e-06L
data8 0xAE9CD9DCADC86113, 0x0000BFFB //A1 = -8.52601070853077921197e-02L
data8 0x9D429743E312AD9F, 0x0000BFFB //A2 = -7.67871682732076080494e-02L
data8 0x8637FC533AE805DC, 0x00003FFC //A3 = +1.31072943286859831330e-01L
data8 0xF68DBE3639ABCB6E, 0x0000BFFB //A4 = -1.20387540845703264588e-01L
data8 0xB168FFC3CFA71256, 0x00003FFB //A5 = +8.66260511047190247534e-02L
data8 0xDBC5078A7EA89236, 0x0000BFFA //A6 = -5.36546988077281230848e-02L
data8 0xF4331FEDB2CB838F, 0x00003FF9 //A7 = +2.98095344165515989564e-02L
data8 0xF909173C0E61C25D, 0x0000BFF8 //A8 = -1.51999213123642373375e-02L
data8 0xEC83560A2ACB23E9, 0x00003FF7 //A9 = +7.21780491979582106904e-03L
data8 0xD350D62C4FEAD8F5, 0x0000BFF6 //A10 = -3.22442272982896360044e-03L
data8 0xB2F44F4B3FD9B826, 0x00003FF5 //A11 = +1.36531322425499451283e-03L
data8 0x9078BC61927671C6, 0x0000BFF4 //A12 = -5.51115510818844954547e-04L
data8 0xDF67AC6287A63B03, 0x00003FF2 //A13 = +2.13055585989529858265e-04L
data8 0xA719CFEE67FCE1CE, 0x0000BFF1 //A14 = -7.96798844477905965933e-05L
data8 0xEF926367BABBB029, 0x00003FEF //A15 = +2.85591875675765038065e-05L
// Pol2
data8 0x82B5E5A93B059C50, 0x00003FEF //A0 = +1.55819100856330860049e-05L
data8 0xDC856BC2542B1938, 0x0000BFFB //A1 = -1.07676355235999875911e-01L
data8 0xDF225EF5694F14AE, 0x0000BFF8 //A2 = -1.36190345125628043277e-02L
data8 0xDAF66A954ED22428, 0x00003FFA //A3 = +5.34576571853233908886e-02L
data8 0xD28AE4F21A392EC6, 0x0000BFFA //A4 = -5.14019911949062230820e-02L
data8 0x9441A95713F0DB5B, 0x00003FFA //A5 = +3.61954321717769771045e-02L
data8 0xB0957B5C483C7A04, 0x0000BFF9 //A6 = -2.15556535133667988704e-02L
data8 0xBB9260E812814F71, 0x00003FF8 //A7 = +1.14484735825400480057e-02L
data8 0xB68AB17287ABAB04, 0x0000BFF7 //A8 = -5.57073273108465072470e-03L
data8 0xA56A95E0BC0EF01B, 0x00003FF6 //A9 = +2.52405318381952650677e-03L
data8 0x8D19C7D286839C00, 0x0000BFF5 //A10 = -1.07651294935087466892e-03L
data8 0xE45DB3766711A0D3, 0x00003FF3 //A11 = +4.35573615323234291196e-04L
data8 0xB05949F947FA7AEF, 0x0000BFF2 //A12 = -1.68179306983868501372e-04L
data8 0x82901D055A0D5CB6, 0x00003FF1 //A13 = +6.22572626227726684168e-05L
data8 0xBB957698542D6FD0, 0x0000BFEF //A14 = -2.23617364009159182821e-05L
data8 0x810740E1DF572394, 0x00003FEE //A15 = +7.69068800065192940487e-06L
// Pol3
data8 0x9526D1C87655AFA8, 0x00003FEC //A0 = +2.22253260814242012255e-06L
data8 0xA47E21EBFE73F72F, 0x0000BFF8 //A1 = -1.00398379581527733314e-02L
data8 0xDE65685FCDF7A913, 0x0000BFFA //A2 = -5.42959286802879105148e-02L
data8 0xED289CB8F97D4860, 0x00003FFA //A3 = +5.79000589346770417248e-02L
data8 0xAA3100D5A7D870F1, 0x0000BFFA //A4 = -4.15506394006027604387e-02L
data8 0xCA0567032C5308C0, 0x00003FF9 //A5 = +2.46607791863290331169e-02L
data8 0xD3E1794A50F31BEB, 0x0000BFF8 //A6 = -1.29321751094401754013e-02L
data8 0xCAA02CB4C87CC1F0, 0x00003FF7 //A7 = +6.18364508551740736863e-03L
data8 0xB3F126AF16B121F2, 0x0000BFF6 //A8 = -2.74569696838501870748e-03L
data8 0x962B2D64D3900510, 0x00003FF5 //A9 = +1.14569596409019883022e-03L
data8 0xED8785714A9A00FB, 0x0000BFF3 //A10 = -4.53051338046340380512e-04L
data8 0xB325DA4515D8B54C, 0x00003FF2 //A11 = +1.70848714622328427290e-04L
data8 0x8179C36354571747, 0x0000BFF1 //A12 = -6.17387951061077132522e-05L
data8 0xB40F241C01C907E9, 0x00003FEF //A13 = +2.14647227210702861416e-05L
data8 0xF436D84AD7D4D316, 0x0000BFED //A14 = -7.27815144835213913238e-06L
data8 0x9EB432503FB0B7BC, 0x00003FEC //A15 = +2.36487228755136968792e-06L
// Pol4
data8 0xE0BA539E4AFC4741, 0x00003FED //A0 = +6.69741148991838024429e-06L
data8 0x8583BF71139452CF, 0x0000BFFA //A1 = -3.25963476363756051657e-02L
data8 0x8384FEF6D08AD6CE, 0x0000BFF9 //A2 = -1.60546283500634200479e-02L
data8 0xB1E67DFB84C97036, 0x00003FF9 //A3 = +2.17163525195697635702e-02L
data8 0xFB6ACEE6899E360D, 0x0000BFF8 //A4 = -1.53452892792759316229e-02L
data8 0x8D2B869EB9149905, 0x00003FF8 //A5 = +8.61633440480716870830e-03L
data8 0x8A90BFE0FD869A41, 0x0000BFF7 //A6 = -4.22868126950622376530e-03L
data8 0xF7536A76E59F54D2, 0x00003FF5 //A7 = +1.88694643606912107006e-03L
data8 0xCCF6FE58C16E1CC7, 0x0000BFF4 //A8 = -7.81878732767742447339e-04L
data8 0x9FCC6ED9914FAA24, 0x00003FF3 //A9 = +3.04791577214885118730e-04L
data8 0xEC7F5AAACAE593E8, 0x0000BFF1 //A10 = -1.12770784960291779798e-04L
data8 0xA72CE628A114C940, 0x00003FF0 //A11 = +3.98577182157456408782e-05L
data8 0xE2DCC5750FD769BA, 0x0000BFEE //A12 = -1.35220520471857266339e-05L
data8 0x9459160B1E6F1F8D, 0x00003FED //A13 = +4.42111470121432700283e-06L
data8 0xBE0A05701BD0DD42, 0x0000BFEB //A14 = -1.41590196994052764542e-06L
data8 0xE905D729105081BF, 0x00003FE9 //A15 = +4.34038814785401120999e-07L
// Pol5
data8 0xA33649C3AB459832, 0x00003FEE //A0 = +9.72819704141525206634e-06L
data8 0x9E4EA2F44C9A24BD, 0x0000BFFA //A1 = -3.86492123987296806210e-02L
data8 0xE80C0B1280F357BF, 0x0000BFF2 //A2 = -2.21297306012713370124e-04L
data8 0xDAECCE90A4D45D9A, 0x00003FF7 //A3 = +6.68106161291482829670e-03L
data8 0xA4006572071BDD4B, 0x0000BFF7 //A4 = -5.00493005170532147076e-03L
data8 0xB07FD7EB1F4D8E8E, 0x00003FF6 //A5 = +2.69316693731732554959e-03L
data8 0xA1F471D42ADD73A1, 0x0000BFF5 //A6 = -1.23561753760779610478e-03L
data8 0x8611D0ED1B4C8176, 0x00003FF4 //A7 = +5.11434914439322741260e-04L
data8 0xCDADB789B487A541, 0x0000BFF2 //A8 = -1.96150380913036018825e-04L
data8 0x9470252731687FEE, 0x00003FF1 //A9 = +7.07807859951401721129e-05L
data8 0xCB9399AD1C376D85, 0x0000BFEF //A10 = -2.42682175234436724152e-05L
data8 0x858D815F9CA0A9F7, 0x00003FEE //A11 = +7.96036454038012144300e-06L
data8 0xA878D338E6E6A079, 0x0000BFEC //A12 = -2.51042802626063073967e-06L
data8 0xCD2C2F079D2FCB36, 0x00003FEA //A13 = +7.64327468786076941271e-07L
data8 0xF5EF4A4B2EA426F2, 0x0000BFE8 //A14 = -2.29044563492386125272e-07L
data8 0x8CE52181393820FC, 0x00003FE7 //A15 = +6.56093668622712763489e-08L
// Pol6
data8 0xB2015D7F1864B7CF, 0x00003FEC //A0 = +2.65248615880090351276e-06L
data8 0x954EA7A861B4462A, 0x0000BFFA //A1 = -3.64519642954351295215e-02L
data8 0x9E46F2A4D9157E69, 0x00003FF7 //A2 = +4.83023498390681965101e-03L
data8 0xA0D12B422FFD5BAD, 0x00003FF5 //A3 = +1.22693684633643883352e-03L
data8 0xB291D16A560A740E, 0x0000BFF5 //A4 = -1.36237794246703606647e-03L
data8 0xC138941BC8AF4A9D, 0x00003FF4 //A5 = +7.37079658343628747256e-04L
data8 0xA761669D61B405CF, 0x0000BFF3 //A6 = -3.19252914480518163396e-04L
data8 0x8053680F1C84607E, 0x00003FF2 //A7 = +1.22381025852939439541e-04L
data8 0xB518F4B6F25015F9, 0x0000BFF0 //A8 = -4.31770048258291369742e-05L
data8 0xEFF526AC70B9411E, 0x00003FEE //A9 = +1.43025887824433324525e-05L
data8 0x970B2A848DF5B5C2, 0x0000BFED //A10 = -4.50145058393497252604e-06L
data8 0xB614D2E61DB86963, 0x00003FEB //A11 = +1.35661172167726780059e-06L
data8 0xD34EA4D283EC33FA, 0x0000BFE9 //A12 = -3.93590335713880681528e-07L
data8 0xED209EBD68E1145F, 0x00003FE7 //A13 = +1.10421060667544991323e-07L
data8 0x83A126E22A17568D, 0x0000BFE6 //A14 = -3.06473811074239684132e-08L
data8 0x8B778496EDE9F415, 0x00003FE4 //A15 = +8.11804009754249175736e-09L
// Pol7
data8 0x8E152F522501B7B9, 0x00003FEE //A0 = +8.46879203970927626532e-06L
data8 0xFD22F92EE21F491E, 0x0000BFF9 //A1 = -3.09004656656418947425e-02L
data8 0xAF0C41847D89EC14, 0x00003FF7 //A2 = +5.34203719233189217519e-03L
data8 0xB7C539C400445956, 0x0000BFF3 //A3 = -3.50514245383356287965e-04L
data8 0x8428C78B2B1E3622, 0x0000BFF3 //A4 = -2.52073850239006530978e-04L
data8 0xAFC0CCC7D1A05F5B, 0x00003FF2 //A5 = +1.67611241057491801028e-04L
data8 0x95DC7272C5695A5A, 0x0000BFF1 //A6 = -7.14593512262564106636e-05L
data8 0xD6FCA68A61F0E835, 0x00003FEF //A7 = +2.56284375437771117850e-05L
data8 0x8B71C74DEA936C66, 0x0000BFEE //A8 = -8.31153675277218441096e-06L
data8 0xA8AC71E2A56AA2C9, 0x00003FEC //A9 = +2.51343269277107451413e-06L
data8 0xC15DED6C44B46046, 0x0000BFEA //A10 = -7.20347851650066610771e-07L
data8 0xD42BA1DFBD1277AC, 0x00003FE8 //A11 = +1.97599119274780745741e-07L
data8 0xE03A81F2C976D11A, 0x0000BFE6 //A12 = -5.22072765405802337371e-08L
data8 0xE56A19A67DD66100, 0x00003FE4 //A13 = +1.33536787408751203998e-08L
data8 0xE964D255CB31DFFA, 0x0000BFE2 //A14 = -3.39632729387679010008e-09L
data8 0xE22E62E932B704D4, 0x00003FE0 //A15 = +8.22842400379225526299e-10L
// Pol8
data8 0xB8B835882D46A6C8, 0x00003FEF //A0 = +2.20202883282415435401e-05L
data8 0xC9D1F63F89B74E90, 0x0000BFF9 //A1 = -2.46362504515706189782e-02L
data8 0x8E376748B1274F30, 0x00003FF7 //A2 = +4.34010070001387441657e-03L
data8 0x98174C7EA49B5B37, 0x0000BFF4 //A3 = -5.80181163659971286762e-04L
data8 0x8D2C40506AE9FF97, 0x00003FEF //A4 = +1.68291159100251734927e-05L
data8 0xD9A580C115B9D150, 0x00003FEF //A5 = +2.59454841475194555896e-05L
data8 0xDB35B21F1C3F99CE, 0x0000BFEE //A6 = -1.30659192305072674545e-05L
data8 0x99FAADAE17A3050E, 0x00003FED //A7 = +4.58893813631592314881e-06L
data8 0xBA1D259BCD6987A9, 0x0000BFEB //A8 = -1.38665627771423394637e-06L
data8 0xCDD7FF5BEA0145C2, 0x00003FE9 //A9 = +3.83413844219813384124e-07L
data8 0xD60857176CE6AB9D, 0x0000BFE7 //A10 = -9.96666862214499946343e-08L
data8 0xD446A2402112DF4C, 0x00003FE5 //A11 = +2.47121687566658908126e-08L
data8 0xCA87133235F1F495, 0x0000BFE3 //A12 = -5.89433000014933371980e-09L
data8 0xBB15B0021581C8B6, 0x00003FE1 //A13 = +1.36122047057936849125e-09L
data8 0xAC9D6585D4AF505E, 0x0000BFDF //A14 = -3.13984547328132268695e-10L
data8 0x975A1439C3795183, 0x00003FDD //A15 = +6.88268624429648826457e-11L
// Pol9
data8 0x99A7676284CDC9FE, 0x00003FEF //A0 = +1.83169747921764176475e-05L
data8 0x9AD0AE249A02896C, 0x0000BFF9 //A1 = -1.88983346204739151909e-02L
data8 0xCB89B4AEC19898BE, 0x00003FF6 //A2 = +3.10574208447745576452e-03L
data8 0xEBBC47E30E1AC2C2, 0x0000BFF3 //A3 = -4.49629730048297442064e-04L
data8 0xD1E35B7FCE1CF859, 0x00003FF0 //A4 = +5.00412261289558493438e-05L
data8 0xB40743664EF24552, 0x0000BFEB //A5 = -1.34131589671166307319e-06L
data8 0xCAD2F5C596FFE1B4, 0x0000BFEB //A6 = -1.51115702599728593837e-06L
data8 0xAE42B6D069DFDDF2, 0x00003FEA //A7 = +6.49171330116787223873e-07L
data8 0xD0739A05BB43A714, 0x0000BFE8 //A8 = -1.94135651872623440782e-07L
data8 0xD745B854AB601BD7, 0x00003FE6 //A9 = +5.01219983943456578062e-08L
data8 0xCC4066E13E338B13, 0x0000BFE4 //A10 = -1.18890061172430768892e-08L
data8 0xB6EAADB55A6C3CB4, 0x00003FE2 //A11 = +2.66178850259168707794e-09L
data8 0x9CC6C178AD3F96AD, 0x0000BFE0 //A12 = -5.70349182959704086428e-10L
data8 0x81D0E2AA27DEB74A, 0x00003FDE //A13 = +1.18066926578104076645e-10L
data8 0xD75FB9049190BEFD, 0x0000BFDB //A14 = -2.44851795398843967972e-11L
data8 0xA9384A51D48C8703, 0x00003FD9 //A15 = +4.80951837368635202609e-12L
// Pol10
data8 0xD2B3482EE449C535, 0x00003FEE //A0 = +1.25587177382575655080e-05L
data8 0xE7939B2D0607DFCF, 0x0000BFF8 //A1 = -1.41343131436717436429e-02L
data8 0x8810EB4AC5F0F1CE, 0x00003FF6 //A2 = +2.07620377002350121270e-03L
data8 0x9546589602AEB955, 0x0000BFF3 //A3 = -2.84719065122144294949e-04L
data8 0x9333434342229798, 0x00003FF0 //A4 = +3.50952732796136549298e-05L
data8 0xEB36A98FD81D3DEB, 0x0000BFEC //A5 = -3.50495464815398722482e-06L
data8 0xAC370EFA025D0477, 0x00003FE8 //A6 = +1.60387784498518639254e-07L
data8 0xC8DF7F8ACA099426, 0x00003FE6 //A7 = +4.67693991699936842330e-08L
data8 0xAC694AD4921C02CF, 0x0000BFE5 //A8 = -2.00713167514877937714e-08L
data8 0xB6E29F2FDE2D8C1A, 0x00003FE3 //A9 = +5.32266106167252495164e-09L
data8 0xA41F8EEA75474358, 0x0000BFE1 //A10 = -1.19415398856537468324e-09L
data8 0x869D778A1C56D3D6, 0x00003FDF //A11 = +2.44863450057778470469e-10L
data8 0xD02658BF31411F4C, 0x0000BFDC //A12 = -4.73277831746128372261e-11L
data8 0x9A4A95EE59127779, 0x00003FDA //A13 = +8.77044784978207256260e-12L
data8 0xE518330AF013C2F6, 0x0000BFD7 //A14 = -1.62781453276882333209e-12L
data8 0xA036A9DF71BD108A, 0x00003FD5 //A15 = +2.84596398987114375607e-13L
// Pol11
data8 0x9191CFBF001F3BB3, 0x00003FEE //A0 = +8.67662287973472452343e-06L
data8 0xAA47E0CF01AE9730, 0x0000BFF8 //A1 = -1.03931136509584404513e-02L
data8 0xAEABE7F17B01D18F, 0x00003FF5 //A2 = +1.33263784731775399430e-03L
data8 0xAC0D6A309D04E5DB, 0x0000BFF2 //A3 = -1.64081956462118568288e-04L
data8 0xA08357DF458054D0, 0x00003FEF //A4 = +1.91346477952797715021e-05L
data8 0x8A1596B557440FE0, 0x0000BFEC //A5 = -2.05761687274453412571e-06L
data8 0xCDA0EAE0A5615E9A, 0x00003FE8 //A6 = +1.91506542215670149741e-07L
data8 0xD36A08FB4E104F9A, 0x0000BFE4 //A7 = -1.23059260396551086769e-08L
data8 0xD7433F91E78A7A11, 0x0000BFDF //A8 = -3.91560549815575091188e-10L
data8 0xC2F5308FD4F5CE62, 0x00003FDF //A9 = +3.54626121852421163117e-10L
data8 0xC83876915F49D630, 0x0000BFDD //A10 = -9.10497688901018285126e-11L
data8 0xA11C605DEAE1FE9C, 0x00003FDB //A11 = +1.83161825409194847892e-11L
data8 0xE7977BC1342D19BF, 0x0000BFD8 //A12 = -3.29111645807102123274e-12L
data8 0x9BC3A7D6396C6756, 0x00003FD6 //A13 = +5.53385887288503961220e-13L
data8 0xD0110D5683740B8C, 0x0000BFD3 //A14 = -9.24001363293241428519e-14L
data8 0x81786D7856A5CC92, 0x00003FD1 //A15 = +1.43741041714595023996e-14L
// Pol12
data8 0xB85654F6033B3372, 0x00003FEF //A0 = +2.19747106911869287049e-05L
data8 0xF78B40078736B406, 0x0000BFF7 //A1 = -7.55444170413862312647e-03L
data8 0xDA8FDE84D88E5D5D, 0x00003FF4 //A2 = +8.33747822263358628569e-04L
data8 0xBC2D3F3891721AA9, 0x0000BFF1 //A3 = -8.97296647669960333635e-05L
data8 0x9D15ACFD3BF50064, 0x00003FEE //A4 = +9.36297600601039610762e-06L
data8 0xFBED3D03F3C1B671, 0x0000BFEA //A5 = -9.38500137149172923985e-07L
data8 0xBEE615E3B2FA16C8, 0x00003FE7 //A6 = +8.88941676851808958175e-08L
data8 0x843D32692CF5662A, 0x0000BFE4 //A7 = -7.69732580860195238520e-09L
data8 0x99E74472FD94E22B, 0x00003FE0 //A8 = +5.59897264617128952416e-10L
data8 0xCEF63DABF4C32E15, 0x0000BFDB //A9 = -2.35288414996279313219e-11L
data8 0xA2D86C25C0991123, 0x0000BFD8 //A10 = -2.31417232327307408235e-12L
data8 0xF50C1B31D2E922BD, 0x00003FD6 //A11 = +8.70582858983364191159e-13L
data8 0xC0F093DEC2B019A1, 0x0000BFD4 //A12 = -1.71364927865227509533e-13L
data8 0xFC1441C4CD105981, 0x00003FD1 //A13 = +2.79864052545369490865e-14L
data8 0x9CC959853267F026, 0x0000BFCF //A14 = -4.35170017302700609509e-15L
data8 0xB06BA14016154F1E, 0x00003FCC //A15 = +6.12081320471295704631e-16L
// Pol13
data8 0xA59E74BF544F2422, 0x00003FEF //A0 = +1.97433196215210145261e-05L
data8 0xB2814F4EDAE15330, 0x0000BFF7 //A1 = -5.44754383528015875700e-03L
data8 0x867C249D378F0A23, 0x00003FF4 //A2 = +5.13019308804593120161e-04L
data8 0xC76644393388AB68, 0x0000BFF0 //A3 = -4.75405403392600215101e-05L
data8 0x91143AD5CCA229FE, 0x00003FED //A4 = +4.32369180778264703719e-06L
data8 0xCE6A11FB6840A974, 0x0000BFE9 //A5 = -3.84476663329551178495e-07L
data8 0x8EC29F66C59DE243, 0x00003FE6 //A6 = +3.32389596787155456596e-08L
data8 0xBE3FCDDCA94CA24E, 0x0000BFE2 //A7 = -2.76849073931513325199e-09L
data8 0xF06A84BDC70A0B0D, 0x00003FDE //A8 = +2.18657158231304988330e-10L
data8 0x8B8E6969D056D124, 0x0000BFDB //A9 = -1.58657139740906811035e-11L
data8 0x8984985AA29A0567, 0x00003FD7 //A10 = +9.77123802231106533829e-13L
data8 0xA53ABA084300137C, 0x0000BFD2 //A11 = -3.66882970952892030306e-14L
data8 0xA90EC851E91C3319, 0x0000BFCE //A12 = -2.34614750044359490986e-15L
data8 0xEC9CAF64237B5060, 0x00003FCC //A13 = +8.20912960028437475035e-16L
data8 0xA9156668FCF01479, 0x0000BFCA //A14 = -1.46656639874123613261e-16L
data8 0xBAEF58D8118DD5D4, 0x00003FC7 //A15 = +2.02675278255254907493e-17L
// Pol14
data8 0xC698952E9CEAA800, 0x00003FEF //A0 = +2.36744912073515619263e-05L
data8 0x800395F8C7B4FA00, 0x0000BFF7 //A1 = -3.90667746392883642897e-03L
data8 0xA3B2467B6B391831, 0x00003FF3 //A2 = +3.12226081793919541155e-04L
data8 0xCF2061122A69D72B, 0x0000BFEF //A3 = -2.46914006692526122176e-05L
data8 0x817FAB6B5DEB9924, 0x00003FEC //A4 = +1.92968114320180123521e-06L
data8 0x9FC190F5827740E7, 0x0000BFE8 //A5 = -1.48784479265231093475e-07L
data8 0xC1FE5C1835C8AFCD, 0x00003FE4 //A6 = +1.12919132662720380018e-08L
data8 0xE7216A9FBB204DA3, 0x0000BFE0 //A7 = -8.40847981461949000003e-10L
data8 0x867566ED95C5C64F, 0x00003FDD //A8 = +6.11446929759298780795e-11L
data8 0x97A8BFA723F0F014, 0x0000BFD9 //A9 = -4.31041298699752869577e-12L
data8 0xA3D24B7034984522, 0x00003FD5 //A10 = +2.91005377301348717042e-13L
data8 0xA5AAA371C22F3741, 0x0000BFD1 //A11 = -1.83926825395757259128e-14L
data8 0x95352E5597EACC23, 0x00003FCD //A12 = +1.03533666540077850452e-15L
data8 0xCCEBE3043B689428, 0x0000BFC8 //A13 = -4.44352525147076912166e-17L
data8 0xA779DAB4BE1F80BB, 0x0000BFBC //A14 = -8.86610526981738255206e-21L
data8 0xB171271F3517282C, 0x00003FC1 //A15 = +3.00598445879282370850e-19L
// Pol15
data8 0xB7AC727D1C3FEB05, 0x00003FEE //A0 = +1.09478009914822049780e-05L
data8 0xB6E6274485C10B0A, 0x0000BFF6 //A1 = -2.79081782038927199588e-03L
data8 0xC5CAE2122D009506, 0x00003FF2 //A2 = +1.88629638738336219173e-04L
data8 0xD466E7957D0A3362, 0x0000BFEE //A3 = -1.26601440424012313479e-05L
data8 0xE2593D798DA20E2E, 0x00003FEA //A4 = +8.43214222346512003230e-07L
data8 0xEF2D2BBA7D2882CC, 0x0000BFE6 //A5 = -5.56876064495961858535e-08L
data8 0xFA5819BB4AE974C2, 0x00003FE2 //A6 = +3.64298674151704370449e-09L
data8 0x819BB0CE825FBB28, 0x0000BFDF //A7 = -2.35755881668932259913e-10L
data8 0x84871099BF728B8F, 0x00003FDB //A8 = +1.50666434199945890414e-11L
data8 0x858188962DFEBC9F, 0x0000BFD7 //A9 = -9.48617116568458677088e-13L
data8 0x840F38FF2FBAE753, 0x00003FD3 //A10 = +5.86461827778372616657e-14L
data8 0xFF47EAF69577B213, 0x0000BFCE //A11 = -3.54273456410181081472e-15L
data8 0xEF402CCB4D29FAF8, 0x00003FCA //A12 = +2.07516888659313950588e-16L
data8 0xD6B789E01141231B, 0x0000BFC6 //A13 = -1.16398290506765191078e-17L
data8 0xB5EEE343E9CFE3EC, 0x00003FC2 //A14 = +6.16413506924643419723e-19L
data8 0x859B41A39D600346, 0x0000BFBE //A15 = -2.82922705825870414438e-20L
// Pol16
data8 0x85708B69FD184E11, 0x00003FED //A0 = +3.97681079176353356199e-06L
data8 0x824D92BC60A1F70A, 0x0000BFF6 //A1 = -1.98826630037499070532e-03L
data8 0xEDCF7D3576BB5258, 0x00003FF1 //A2 = +1.13396885054265675352e-04L
data8 0xD7FC59226A947CDF, 0x0000BFED //A3 = -6.43687650810478871875e-06L
data8 0xC32C51B574E2651E, 0x00003FE9 //A4 = +3.63538268539251809118e-07L
data8 0xAF67910F5681401F, 0x0000BFE5 //A5 = -2.04197779750247395258e-08L
data8 0x9CB3E8D7DCD1EA9D, 0x00003FE1 //A6 = +1.14016272459029850306e-09L
data8 0x8B14ECFBF7D4F114, 0x0000BFDD //A7 = -6.32470533185766848692e-11L
data8 0xF518253AE4A3AE72, 0x00003FD8 //A8 = +3.48299974583453268369e-12L
data8 0xD631A5699AA2F334, 0x0000BFD4 //A9 = -1.90242426474085078079e-13L
data8 0xB971AD4C30C56E5D, 0x00003FD0 //A10 = +1.02942127356740047925e-14L
data8 0x9ED0065A601F3160, 0x0000BFCC //A11 = -5.50991880383698965959e-16L
data8 0x863A04008E12867C, 0x00003FC8 //A12 = +2.91057593756148904838e-17L
data8 0xDF62F9F44F5C7170, 0x0000BFC3 //A13 = -1.51372666097522872780e-18L
data8 0xBA4E118E88CFDD31, 0x00003FBF //A14 = +7.89032177282079635722e-20L
data8 0x942AD897FC4D2F2A, 0x0000BFBB //A15 = -3.92195756076319409245e-21L
// Pol17
data8 0xCB8514540566C717, 0x00003FEF //A0 = +2.42614557068144130848e-05L
data8 0xB94F08D6816E0CD4, 0x0000BFF5 //A1 = -1.41379340061829929314e-03L
data8 0x8E7C342C2DABB51B, 0x00003FF1 //A2 = +6.79422240687700109911e-05L
data8 0xDA69DAFF71E30D5B, 0x0000BFEC //A3 = -3.25461473899657142468e-06L
data8 0xA6D5B2DB69B4B3F6, 0x00003FE8 //A4 = +1.55376978584082701045e-07L
data8 0xFDF4F76BC1D1BD47, 0x0000BFE3 //A5 = -7.39111857092131684572e-09L
data8 0xC08BC52C95B12C2D, 0x00003FDF //A6 = +3.50239092565793882444e-10L
data8 0x91624BF6D3A3F6C9, 0x0000BFDB //A7 = -1.65282439890232458821e-11L
data8 0xDA91F7A450DE4270, 0x00003FD6 //A8 = +7.76517285902715940501e-13L
data8 0xA380ADF55416E624, 0x0000BFD2 //A9 = -3.63048822989374426852e-14L
data8 0xF350FC0CEDEE0FD6, 0x00003FCD //A10 = +1.68834630987974622269e-15L
data8 0xB3FA19FBDC8F023C, 0x0000BFC9 //A11 = -7.80525639701804380489e-17L
data8 0x8435328C80940126, 0x00003FC5 //A12 = +3.58349966898667910204e-18L
data8 0xC0D22F655BA5EF39, 0x0000BFC0 //A13 = -1.63325770165403860181e-19L
data8 0x8F14B9EBD5A9AB25, 0x00003FBC //A14 = +7.57464305512080733773e-21L
data8 0xCD4804BBF6DC1B6F, 0x0000BFB7 //A15 = -3.39609459750208886298e-22L
// Pol18
data8 0xE251DFE45AB0C22E, 0x00003FEE //A0 = +1.34897126299700418200e-05L
data8 0x83943CC7D59D4215, 0x0000BFF5 //A1 = -1.00386850310061655307e-03L
data8 0xAA57896951134BCA, 0x00003FF0 //A2 = +4.06126834109940757047e-05L
data8 0xDC0A67051E1C4A2C, 0x0000BFEB //A3 = -1.63943048164477430317e-06L
data8 0x8DCB3C0A8CD07BBE, 0x00003FE7 //A4 = +6.60279229777753829876e-08L
data8 0xB64DE81C24F7F265, 0x0000BFE2 //A5 = -2.65287705357477481067e-09L
data8 0xE9CBB7A990DBA8B5, 0x00003FDD //A6 = +1.06318007608620426224e-10L
data8 0x9583D4B85C2ADC6F, 0x0000BFD9 //A7 = -4.24947087941505088222e-12L
data8 0xBEB0EE8114EEDF77, 0x00003FD4 //A8 = +1.69367754741562774916e-13L
data8 0xF2791BB8F06BDA93, 0x0000BFCF //A9 = -6.72997988617021128704e-15L
data8 0x99A907F6A92195B4, 0x00003FCB //A10 = +2.66558091161711891239e-16L
data8 0xC213E5E6F833BB93, 0x0000BFC6 //A11 = -1.05209746502719578617e-17L
data8 0xF41FBBA6B343960F, 0x00003FC1 //A12 = +4.13562069721140021224e-19L
data8 0x98F194AEE31D188D, 0x0000BFBD //A13 = -1.61935414722333263347e-20L
data8 0xC42F5029BB622157, 0x00003FB8 //A14 = +6.49121108201931196678e-22L
data8 0xF43BD08079E50E0F, 0x0000BFB3 //A15 = -2.52531675510242468317e-23L
// Pol19
data8 0x82557B149A04D08E, 0x00003FEF //A0 = +1.55370127331027842820e-05L
data8 0xBAAB433307CE614B, 0x0000BFF4 //A1 = -7.12085701486669872724e-04L
data8 0xCB52D9DBAC16FE82, 0x00003FEF //A2 = +2.42380662859334411743e-05L
data8 0xDD214359DBBCE7D1, 0x0000BFEA //A3 = -8.23773197624244883859e-07L
data8 0xF01E8E968139524C, 0x00003FE5 //A4 = +2.79535729459988509676e-08L
data8 0x82286A057E0916CE, 0x0000BFE1 //A5 = -9.47023128967039348510e-10L
data8 0x8CDDDC4E8D013365, 0x00003FDC //A6 = +3.20293663356974901319e-11L
data8 0x982FEEE90D4E8751, 0x0000BFD7 //A7 = -1.08135537312234452657e-12L
data8 0xA41D1E84083B8FD6, 0x00003FD2 //A8 = +3.64405720894915411836e-14L
data8 0xB0A1B6111B72E159, 0x0000BFCD //A9 = -1.22562851790685744085e-15L
data8 0xBDB77DE6B650FFA2, 0x00003FC8 //A10 = +4.11382657214908334175e-17L
data8 0xCB54E95CDB66978A, 0x0000BFC3 //A11 = -1.37782909696752432371e-18L
data8 0xD959E428A62B1B6C, 0x00003FBE //A12 = +4.60258936838597812582e-20L
data8 0xE7D49EC23F1A16A0, 0x0000BFB9 //A13 = -1.53412587409583783059e-21L
data8 0xFDE429BC9947B2BE, 0x00003FB4 //A14 = +5.25034823750902928092e-23L
data8 0x872137A062C042EF, 0x0000BFB0 //A15 = -1.74651114923000080365e-24L
// Pol20
data8 0x8B9B185C6A2659AC, 0x00003FEF //A0 = +1.66423130594825442963e-05L
data8 0x84503AD52588A1E8, 0x0000BFF4 //A1 = -5.04735556466270303549e-04L
data8 0xF26C7C2B566388E1, 0x00003FEE //A2 = +1.44495826764677427386e-05L
data8 0xDDDA15FEE262BB47, 0x0000BFE9 //A3 = -4.13231361893675488873e-07L
data8 0xCACEBC73C90C2FE0, 0x00003FE4 //A4 = +1.18049538609157282958e-08L
data8 0xB9314D00022B41DD, 0x0000BFDF //A5 = -3.36863342776746896664e-10L
data8 0xA8E9FBDC714638B9, 0x00003FDA //A6 = +9.60164921624768038366e-12L
data8 0x99E246C0CC8CA6F6, 0x0000BFD5 //A7 = -2.73352704217713596798e-13L
data8 0x8C04E7B5DF372EA1, 0x00003FD0 //A8 = +7.77262480048865685174e-15L
data8 0xFE7B90CAA0B6D5F7, 0x0000BFCA //A9 = -2.20728537958846147109e-16L
data8 0xE6F40BAD4EC6CB4F, 0x00003FC5 //A10 = +6.26000182616999972048e-18L
data8 0xD14F4E0538F0F992, 0x0000BFC0 //A11 = -1.77292283439752259258e-19L
data8 0xBD5A7FAA548CC749, 0x00003FBB //A12 = +5.01214569023722089225e-21L
data8 0xAB15D69425373A67, 0x0000BFB6 //A13 = -1.41518447770061562822e-22L
data8 0x9EF95456F75B4DF4, 0x00003FB1 //A14 = +4.10938011540250142351e-24L
data8 0x8FADCC45E81433E7, 0x0000BFAC //A15 = -1.16062889679749879834e-25L
// Pol21
data8 0xB47A917B0F7B50AE, 0x00003FEF //A0 = +2.15147474240529518138e-05L
data8 0xBB77DC3BA0C937B3, 0x0000BFF3 //A1 = -3.57567223048598672970e-04L
data8 0x90694DFF4EBF7370, 0x00003FEE //A2 = +8.60758700336677694536e-06L
data8 0xDE5379AA90A98F3F, 0x0000BFE8 //A3 = -2.07057292787309736495e-07L
data8 0xAB0322293F1F9CA0, 0x00003FE3 //A4 = +4.97711123919916694625e-09L
data8 0x837119E59D3B7AC2, 0x0000BFDE //A5 = -1.19545621970063369582e-10L
data8 0xC9E5B74A38ECF3FC, 0x00003FD8 //A6 = +2.86913359605586285967e-12L
data8 0x9AEF5110C6885352, 0x0000BFD3 //A7 = -6.88048865490621757799e-14L
data8 0xED988D52189CE6A3, 0x00003FCD //A8 = +1.64865278639132278935e-15L
data8 0xB6063CECD8012B6D, 0x0000BFC8 //A9 = -3.94702428606368525374e-17L
data8 0x8B541EB15E79CEEC, 0x00003FC3 //A10 = +9.44127272399408815784e-19L
data8 0xD51A136D8C75BC25, 0x0000BFBD //A11 = -2.25630369561137931232e-20L
data8 0xA2C1C5E19CC79E6F, 0x00003FB8 //A12 = +5.38517493921589837361e-22L
data8 0xF86F9772306F56C1, 0x0000BFB2 //A13 = -1.28438352359240135735e-23L
data8 0xC32F6FEEDE86528E, 0x00003FAD //A14 = +3.15338862172962186458e-25L
data8 0x9534ED189744D7D4, 0x0000BFA8 //A15 = -7.53301543611470014315e-27L
// Pol22
data8 0xCBA0A2DB94A2C494, 0x00003FEF //A0 = +2.42742878212752702946e-05L
data8 0x84C089154A49E0E8, 0x0000BFF3 //A1 = -2.53204520651046300034e-04L
data8 0xABF5665BD0D8B0CD, 0x00003FED //A2 = +5.12476542947092361490e-06L
data8 0xDEA1C518E3EEE872, 0x0000BFE7 //A3 = -1.03671063536324831083e-07L
data8 0x900B77F271559AE8, 0x00003FE2 //A4 = +2.09612770408581408652e-09L
data8 0xBA4C74A262BE3E4E, 0x0000BFDC //A5 = -4.23594098489216166935e-11L
data8 0xF0D1680FCC1EAF97, 0x00003FD6 //A6 = +8.55557381760467917779e-13L
data8 0x9B8F8E033BB83A24, 0x0000BFD1 //A7 = -1.72707138247091685914e-14L
data8 0xC8DCA6A691DB8335, 0x00003FCB //A8 = +3.48439884388851942939e-16L
data8 0x819A6CB9CEA5E9BD, 0x0000BFC6 //A9 = -7.02580471688245511753e-18L
data8 0xA726B4F622585BEA, 0x00003FC0 //A10 = +1.41582572516648501043e-19L
data8 0xD7727648A4095986, 0x0000BFBA //A11 = -2.85141885626054217632e-21L
data8 0x8AB627E09CF45997, 0x00003FB5 //A12 = +5.73697507862703019314e-23L
data8 0xB28C15C117CC604F, 0x0000BFAF //A13 = -1.15383428132352407085e-24L
data8 0xECB8428626DA072C, 0x00003FA9 //A14 = +2.39025879246942839796e-26L
data8 0x98B731BCFA2CE2B2, 0x0000BFA4 //A15 = -4.81885474332093262902e-28L
// Pol23
data8 0xC6D013811314D31B, 0x00003FED //A0 = +5.92508308918577687876e-06L
data8 0xBBF3057B8DBACBCF, 0x0000BFF2 //A1 = -1.79242422493281965934e-04L
data8 0xCCADECA501162313, 0x00003FEC //A2 = +3.04996061562356504918e-06L
data8 0xDED1FDBE8CCAF3DB, 0x0000BFE6 //A3 = -5.18793887648024117154e-08L
data8 0xF27B74EDDCA65859, 0x00003FE0 //A4 = +8.82145297317787820675e-10L
data8 0x83E4415687F01A0C, 0x0000BFDB //A5 = -1.49943414247603665601e-11L
data8 0x8F6CB350861CE446, 0x00003FD5 //A6 = +2.54773288906376920377e-13L
data8 0x9BE8456A30CBFC02, 0x0000BFCF //A7 = -4.32729710913845745148e-15L
data8 0xA9694F7E1033977D, 0x00003FC9 //A8 = +7.34704698157502347441e-17L
data8 0xB8035A3D5AF82D85, 0x0000BFC3 //A9 = -1.24692123826025468001e-18L
data8 0xC7CB4B3ACB905FDA, 0x00003FBD //A10 = +2.11540249352095943317e-20L
data8 0xD8D70AEB2E58D729, 0x0000BFB7 //A11 = -3.58731705184186608576e-22L
data8 0xEB27A61B1D5C7697, 0x00003FB1 //A12 = +6.07861113430709162243e-24L
data8 0xFEF9ED74D4F4C9B0, 0x0000BFAB //A13 = -1.02984099170876754831e-25L
data8 0x8E6F410068C12043, 0x00003FA6 //A14 = +1.79777721804459361762e-27L
data8 0x9AE2F6705481630E, 0x0000BFA0 //A15 = -3.05459905177379058768e-29L
// Pol24
data8 0xD2D858D5B01C9434, 0x00003FEE //A0 = +1.25673476165670766128e-05L
data8 0x8505330F8B4FDE49, 0x0000BFF2 //A1 = -1.26858053564784963985e-04L
data8 0xF39171C8B1D418C2, 0x00003FEB //A2 = +1.81472407620770441249e-06L
data8 0xDEF065C3D7BFD26E, 0x0000BFE5 //A3 = -2.59535215807652675043e-08L
data8 0xCC0199EA6ACA630C, 0x00003FDF //A4 = +3.71085215769339916703e-10L
data8 0xBAA25319F01ED248, 0x0000BFD9 //A5 = -5.30445960650683029105e-12L
data8 0xAAB28A84F8CFE4D1, 0x00003FD3 //A6 = +7.58048850973457592162e-14L
data8 0x9C14B931AEB311A8, 0x0000BFCD //A7 = -1.08302915828084288776e-15L
data8 0x8EADA745715A0714, 0x00003FC7 //A8 = +1.54692159263197000533e-17L
data8 0x82643F3F722CE6B5, 0x0000BFC1 //A9 = -2.20891945694400066611e-19L
data8 0xEE42ECDE465A99E4, 0x00003FBA //A10 = +3.15336372779307614198e-21L
data8 0xD99FC74326ACBFC0, 0x0000BFB4 //A11 = -4.50036161691276556269e-23L
data8 0xC6A4DCACC554911E, 0x00003FAE //A12 = +6.41853356148678957077e-25L
data8 0xB550CEA09DA96F44, 0x0000BFA8 //A13 = -9.15410112414783078242e-27L
data8 0xAA9149317996F32F, 0x00003FA2 //A14 = +1.34554050666508391264e-28L
data8 0x9C3008EFE3F52F19, 0x0000BF9C //A15 = -1.92516125328592532359e-30L
// Pol25
data8 0xA68E78218806283F, 0x00003FEF //A0 = +1.98550844852103406280e-05L
data8 0xBC41423996DC8A37, 0x0000BFF1 //A1 = -8.97669395268764751516e-05L
data8 0x90E55AE31A2F8271, 0x00003FEB //A2 = +1.07955871580069359702e-06L
data8 0xDF022272DA4A3BEF, 0x0000BFE4 //A3 = -1.29807937275957214439e-08L
data8 0xAB95DCBFFB0BAAB8, 0x00003FDE //A4 = +1.56056011861921437794e-10L
data8 0x83FF2547BA9011FF, 0x0000BFD8 //A5 = -1.87578539510813332135e-12L
data8 0xCB0C353560EEDC45, 0x00003FD1 //A6 = +2.25428217090412574481e-14L
data8 0x9C24CEB86E76D2C5, 0x0000BFCB //A7 = -2.70866279585559299821e-16L
data8 0xF01AFA23DDFDAE0E, 0x00003FC4 //A8 = +3.25403467375734083376e-18L
data8 0xB892BDFBCF1D9740, 0x0000BFBE //A9 = -3.90848978133441513662e-20L
data8 0x8DDBBF34415AAECA, 0x00003FB8 //A10 = +4.69370027479731756829e-22L
data8 0xDA04170D07458C3B, 0x0000BFB1 //A11 = -5.63558091177482043435e-24L
data8 0xA76F391095A9563A, 0x00003FAB //A12 = +6.76262416498584003290e-26L
data8 0x8098FA125C18D8DB, 0x0000BFA5 //A13 = -8.11564737276592661642e-28L
data8 0xCB9E4D5C08923227, 0x00003F9E //A14 = +1.00391606269366059664e-29L
data8 0x9CEC3BF7A0BE2CAF, 0x0000BF98 //A15 = -1.20888920108938909316e-31L
// Pol26
data8 0xC17AB25E269272F7, 0x00003FEE //A0 = +1.15322640047234590651e-05L
data8 0x85310509E633FEF2, 0x0000BFF1 //A1 = -6.35106483144690768696e-05L
data8 0xAC5E4C4DCB2D940C, 0x00003FEA //A2 = +6.42122148740412561597e-07L
data8 0xDF0AAD0571FFDD48, 0x0000BFE3 //A3 = -6.49136789710824396482e-09L
data8 0x9049D8440AFD180F, 0x00003FDD //A4 = +6.56147932223174570008e-11L
data8 0xBAA936477C5FA9D7, 0x0000BFD6 //A5 = -6.63153032879993841863e-13L
data8 0xF17261294EAB1443, 0x00003FCF //A6 = +6.70149477756803680009e-15L
data8 0x9C22F87C31DB007A, 0x0000BFC9 //A7 = -6.77134581402030645534e-17L
data8 0xC9E98E633942AC12, 0x00003FC2 //A8 = +6.84105580182052870823e-19L
data8 0x828998181309642C, 0x0000BFBC //A9 = -6.91059649300859944955e-21L
data8 0xA8C3D4DCE1ECBAB6, 0x00003FB5 //A10 = +6.97995542988331257517e-23L
data8 0xDA288D52CC4C351A, 0x0000BFAE //A11 = -7.04907829139578377009e-25L
data8 0x8CEEACB790B5F374, 0x00003FA8 //A12 = +7.11526399101774993883e-27L
data8 0xB61C8A29D98F24C0, 0x0000BFA1 //A13 = -7.18303147470398859453e-29L
data8 0xF296F69FE45BDA7D, 0x00003F9A //A14 = +7.47537230021540031251e-31L
data8 0x9D4B25BF6FB7234B, 0x0000BF94 //A15 = -7.57340869663212138051e-33L
// Pol27
data8 0xC7772CC326D6FBB8, 0x00003FEE //A0 = +1.18890718679826004395e-05L
data8 0xE0F9D5410565D55D, 0x0000BFF0 //A1 = -5.36384368533203585378e-05L
data8 0x85C0BE825680E148, 0x00003FEA //A2 = +4.98268406609692971520e-07L
data8 0x9F058A389D7BA177, 0x0000BFE3 //A3 = -4.62813885933188677790e-09L
data8 0xBD0B751F0A6BAC7A, 0x00003FDC //A4 = +4.29838009673609430305e-11L
data8 0xE0B6823570502E9D, 0x0000BFD5 //A5 = -3.99170340031272728535e-13L
data8 0x858A9C52FC426D86, 0x00003FCF //A6 = +3.70651975271664045723e-15L
data8 0x9EB4438BFDF1928D, 0x0000BFC8 //A7 = -3.44134780748056488222e-17L
data8 0xBC968DCD8C06D74E, 0x00003FC1 //A8 = +3.19480670422195579127e-19L
data8 0xE0133A405F782125, 0x0000BFBA //A9 = -2.96560935615546392028e-21L
data8 0x851AFEBB70D07E79, 0x00003FB4 //A10 = +2.75255617931932536111e-23L
data8 0x9E1E21A841BF8738, 0x0000BFAD //A11 = -2.55452923487640676799e-25L
data8 0xBBCF2EF1C6E72327, 0x00003FA6 //A12 = +2.37048675755308004410e-27L
data8 0xDF0D320CF12B8BCB, 0x0000BF9F //A13 = -2.19945804585962185550e-29L
data8 0x8470A76DE5FCADD8, 0x00003F99 //A14 = +2.04056213851532266258e-31L
data8 0x9D41C15F6A6FBB04, 0x0000BF92 //A15 = -1.89291056020108587823e-33L
LOCAL_OBJECT_END(erfc_Q_table)


.section .text
GLOBAL_LIBM_ENTRY(erfcl)

{ .mfi
      alloc          r32 = ar.pfs, 0, 36, 4, 0
      fma.s1         FR_Tmp = f1, f1, f8                   // |x|+1, if x >= 0
      nop.i          0
}
{ .mfi
      addl           GR_ad_Arg    = @ltoff(exp_table_1), gp
      fms.s1         FR_Tmp1 = f1, f1, f8                   // |x|+1, if x < 0
      mov            GR_rshf_2to51 = 0x4718         // begin 1.10000 2^(63+51)
}
;;

{ .mfi
      ld8            GR_ad_Arg = [GR_ad_Arg]             // Point to Arg table
      fcmp.ge.s1     p6,p7 = f8, f0                     // p6: x >= 0 ,p7: x<0
      shl            GR_rshf_2to51 = GR_rshf_2to51,48 // end 1.10000 2^(63+51)
}
{ .mlx
      mov            GR_rshf = 0x43e8     // begin 1.1000 2^63 for right shift
      movl           GR_sig_inv_ln2 = 0xb8aa3b295c17f0bc   // signif. of 1/ln2
}
;;

{ .mfi
      mov            GR_exp_2tom51 = 0xffff-51
      fclass.m       p8,p0 = f8,0x07                            // p8:   x = 0
      shl            GR_rshf = GR_rshf,48   // end 1.1000 2^63 for right shift
}
{ .mfi
      nop.m          0
      fnma.s1        FR_norm_x   = f8, f8, f0             //high bits for -x^2
      nop.i          0
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      setf.sig       FR_INV_LN2_2TO63 = GR_sig_inv_ln2    // form 1/ln2 * 2^63
(p6)  fma.s1         FR_AbsArg = f1, f0, f8                  // |x|, if x >= 0
      nop.i          0
}
{ .mfi
      setf.d         FR_RSHF_2TO51 = GR_rshf_2to51    //const 1.10 * 2^(63+51)
(p7)  fms.s1         FR_AbsArg = f1, f0, f8                   // |x|, if x < 0
      mov            GR_exp_mask = 0x1FFFF               // Form exponent mask
}
;;

{ .mfi
      ldfe           FR_ch_dx = [GR_ad_Arg], 16
      fclass.m       p10,p0 = f8, 0x21                        // p10: x = +inf
      mov            GR_exp_bias = 0x0FFFF                // Set exponent bias
}
{ .mlx
      setf.d         FR_RSHF = GR_rshf      // Right shift const 1.1000 * 2^63
      movl           GR_ERFC_XC_TB = 0x650
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      setf.exp       FR_2TOM51 = GR_exp_2tom51    // 2^-51 for scaling float_N
(p6)  fma.s1         FR_Tmp = FR_Tmp, FR_Tmp, f0            // (|x|+1)^2,x >=0
      nop.i          0
}
{ .mfi
      ldfpd          FR_POS_ARG_ASYMP,FR_NEG_ARG_ASYMP = [GR_ad_Arg], 16
(p7)  fma.s1         FR_Tmp = FR_Tmp1, FR_Tmp1, f0           // (|x|+1)^2, x<0
      mov            GR_0x1 = 0x1
}
;;

//p8: y = 1.0, x = 0.0,quick exit
{ .mfi
      ldfpd          FR_dx,FR_dx1 = [GR_ad_Arg], 16
      fclass.m       p9,p0 = f8, 0x22                          // p9: x = -inf
      nop.i          0

}
{ .mfb
      nop.m          0
(p8)  fma.s0         f8 = f1, f1, f0
(p8)  br.ret.spnt    b0
}
;;

{ .mfi
      ldfe           FR_UnfBound = [GR_ad_Arg], 16
      fclass.m       p11,p0 = f8, 0xc3                        // p11: x = nan
      mov            GR_BIAS = 0x0FFFF
}
{ .mfi
      nop.m          0
      fma.s1         FR_NormX = f8,f1,f0
      nop.i          0
}
;;

{ .mfi
      ldfe           FR_EpsNorm = [GR_ad_Arg], 16
      fmerge.s       FR_X = f8,f8
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_xsq_lo = f8, f8, FR_norm_x        // low bits for -x^2
      nop.i          0
}
;;

{ .mfi
      add            GR_ad_C = 0x20, GR_ad_Arg             // Point to C table
      nop.f          0
      add            GR_ad_T1 = 0x50, GR_ad_Arg           // Point to T1 table
}
{ .mfi
      add            GR_ad_T2 = 0x150, GR_ad_Arg          // Point to T2 table
      nop.f          0
      add            GR_ERFC_XC_TB = GR_ERFC_XC_TB, GR_ad_Arg //poin.to XB_TBL
}
;;

{ .mfi
      getf.exp       GR_signexp_x = FR_norm_x  // Extr. sign and exponent of x
      fma.s1         FR_Tmp = FR_Tmp, FR_Tmp, f0                  // (|x|+1)^4
      add            GR_ad_W1 = 0x100, GR_ad_T2           // Point to W1 table
}
{ .mfi
      ldfe           FR_L_hi = [GR_ad_Arg],16                      // Get L_hi
      nop.f          0
      add            GR_ad_W2 = 0x300, GR_ad_T2           // Point to W2 table
}
;;

// p9: y = 2.0, x = -inf, quick exit
{ .mfi
      sub            GR_mBIAS = r0, GR_BIAS
      fma.s1         FR_2 = f1, f1, f1
      nop.i          0
}
{ .mfb
      ldfe           FR_L_lo = [GR_ad_Arg],16                      // Get L_lo
(p9)  fma.s0         f8 = f1, f1, f1
(p9)  br.ret.spnt    b0
}
;;

// p10: y = 0.0, x = +inf, quick exit
{ .mfi
      adds           GR_ERFC_P_TB = 0x380, GR_ERFC_XC_TB   // pointer to P_TBL
      fma.s1         FR_N_signif = FR_norm_x, FR_INV_LN2_2TO63, FR_RSHF_2TO51
      and            GR_exp_x = GR_signexp_x, GR_exp_mask
}
{ .mfb
      adds           GR_ERFC_S_TB = 0x1C0, GR_ERFC_XC_TB   // pointer to S_TBL
(p10) fma.s0         f8 = f0, f1, f0
(p10) br.ret.spnt    b0
}
;;

// p12: |x| < 0.681...  ->  dx = 0.875 (else dx = 0.625 )
// p11: y = x, x = nan, quick exit
{ .mfi
      ldfe           FR_C3 = [GR_ad_C],16           // Get C3 for normal path
      fcmp.lt.s1     p12,p0 = FR_AbsArg, FR_ch_dx
      shl            GR_ShftPi_bias = GR_BIAS, 8                //  BIAS * 256
}
{ .mfb
      sub            GR_exp_x = GR_exp_x, GR_exp_bias          // Get exponent
(p11) fma.s0         f8 = f8, f1, f0
(p11) br.ret.spnt    b0

}
;;

{ .mfi
      ldfe           FR_C2 = [GR_ad_C],16              // Get A2 for main path
      nop.f          0
      nop.i          0
}
;;

//p15: x > POS_ARG_ASYMP = 107.0 -> erfcl(x) ~=~ 0.0
{ .mfi
      ldfe           FR_C1 = [GR_ad_C],16              // Get C1 for main path
(p6)  fcmp.gt.unc.s1 p15,p0 = FR_AbsArg, FR_POS_ARG_ASYMP        // p6: x >= 0
      nop.i          0
}
{ .mfb
      nop.m          0
(p12) fma.s1         FR_dx = FR_dx1, f1, f0   //p12: dx = 0.875 for x < 0.681
      nop.b          0
}
;;

//p14: x < - NEG_ARG_ASYMP = -6.5 -> erfcl(x) ~=~ 2.0
{ .mfi
      nop.m          0
(p7)  fcmp.gt.unc.s1 p14,p0 = FR_AbsArg,FR_NEG_ARG_ASYMP          // p7: x < 0
      shladd         GR_ShftXBi_bias = GR_mBIAS, 4, r0
}
;;

{ .mfi
      nop.m          0
      fma.s0         FR_Tmpf = f1, f1, FR_EpsNorm                    // flag i
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         FR_float_N = FR_N_signif, FR_2TOM51, FR_RSHF
      nop.i          0
}
;;

// p8: x < UnfBound ~=~ 106.53... -> result without underflow error
// p14: y ~=~ 2, x < -6.5,quick exit
{ .mfi
      getf.exp       GR_IndxPlusBias = FR_Tmp      // exp + bias for (|x|+1)^4
      fcmp.lt.s1     p8,p0 = FR_NormX,FR_UnfBound
      nop.i          0
}
{ .mfb
      nop.m          0
(p14) fnma.s0        FR_RESULT = FR_EpsNorm,FR_EpsNorm,FR_2
(p14) br.ret.spnt    b0

}
;;

// p15: y ~=~ 0.0 (result with underflow error), x > POS_ARG_ASYMP = 107.0,
// call __libm_error_region
{ .mfb
(p15) mov            GR_Parameter_TAG = 207
(p15) fma.s0         FR_RESULT = FR_EpsNorm,FR_EpsNorm,f0
(p15) br.cond.spnt   __libm_error_region
}
;;

{ .mfi
      getf.sig       GR_N_fix = FR_N_signif          // Get N from significand
      nop.f          0
      shl            GR_ShftPi = GR_IndxPlusBias, 8

}
{ .mfi
      shladd         GR_ShftXBi = GR_IndxPlusBias, 4, GR_ShftXBi_bias
      nop.f          0
      nop.i          0
}
;;

{ .mmi
      add            GR_ERFC_S_TB = GR_ERFC_S_TB, GR_ShftXBi    //poin.to S[i]
      add            GR_ERFC_XC_TB = GR_ERFC_XC_TB, GR_ShftXBi //poin.to XC[i]
      sub            GR_ShftPi = GR_ShftPi, GR_ShftPi_bias            // 256*i
}
;;

{ .mfi
      ldfe           FR_Xc  = [GR_ERFC_XC_TB]
      fma.s1         FR_Xpdx_hi = FR_AbsArg, f1, FR_dx              //  x + dx
      add            GR_ShftA14 = 0xE0, GR_ShftPi     // pointer shift for A14


}
{ .mfi
      ldfe           FR_S  = [GR_ERFC_S_TB]
      fnma.s1        FR_r = FR_L_hi, FR_float_N, FR_norm_x//r= -L_hi*float_N+x
      add            GR_ShftA15 = 0xF0, GR_ShftPi     // pointer shift for A15
}
;;

{ .mfi
      add            GR_P_POINT_1 = GR_ERFC_P_TB, GR_ShftA14 // pointer to A14
      fcmp.gt.s1     p9,p10 = FR_AbsArg, FR_dx      //p9: x > dx, p10: x <= dx
      extr.u         GR_M1 = GR_N_fix, 6, 6               // Extract index M_1
}
{ .mfi
      add            GR_P_POINT_2 = GR_ERFC_P_TB, GR_ShftA15 // pointer to A15
      nop.f          0
      nop.i          0

}
;;

{ .mfi
      ldfe           FR_A14 = [GR_P_POINT_1], -32
      nop.f          0
      extr.u         GR_M2 = GR_N_fix, 0, 6               // Extract index M_2
}
{ .mfi
      ldfe           FR_A15 = [GR_P_POINT_2], -32
      nop.f          0
      shladd         GR_ad_W1 = GR_M1,3,GR_ad_W1                // Point to W1
}
;;

{ .mfi
      ldfe           FR_A12 = [GR_P_POINT_1], -64
      nop.f          0
      extr           GR_K = GR_N_fix, 12, 32         // Extract limite range K
}
{ .mfi
      ldfe           FR_A13 = [GR_P_POINT_2], -64
      nop.f          0
      shladd         GR_ad_T1 = GR_M1,2,GR_ad_T1                // Point to T1
}
;;

{ .mfi
      ldfe           FR_A8 = [GR_P_POINT_1], 32
      nop.f          0
      add            GR_exp_2_k = GR_exp_bias, GR_K    // Form exponent of 2^k
}
{ .mfi
      ldfe           FR_A9 = [GR_P_POINT_2], 32
      nop.f          0
      shladd         GR_ad_W2 = GR_M2,3,GR_ad_W2                // Point to W2
}
;;

{ .mfi
      ldfe           FR_A10 = [GR_P_POINT_1], -96
      nop.f          0
      shladd         GR_ad_T2 = GR_M2,2,GR_ad_T2                // Point to T2
}
{ .mfi
      ldfe           FR_A11 = [GR_P_POINT_2], -96
      fnma.s1        FR_r = FR_L_lo, FR_float_N, FR_r //r = -L_lo*float_N + r
      nop.i          0
}
;;

{ .mfi
      ldfe           FR_A4 = [GR_P_POINT_1], 32
(p10) fms.s1         FR_Tmp = FR_dx,f1, FR_Xpdx_hi   //for lo  of  x+dx, x<=dx
      nop.i          0
}
{ .mfi
      ldfe           FR_A5 = [GR_P_POINT_2], 32
(p9)  fms.s1         FR_Tmp = FR_AbsArg, f1, FR_Xpdx_hi //for lo of x+dx, x>dx
      nop.i          0
}
;;

{ .mfi
      ldfe           FR_A6 = [GR_P_POINT_1], -64
      frcpa.s1       FR_U,p11 = f1, FR_Xpdx_hi          //  hi of  1 /(x + dx)
      nop.i          0
}
{ .mfi
      ldfe           FR_A7 = [GR_P_POINT_2], -64
      nop.f          0
      nop.i          0
}
;;

{ .mfi
      ldfe           FR_A2 = [GR_P_POINT_1], -32
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfe           FR_A3 = [GR_P_POINT_2], -32
      nop.f          0
      nop.i          0
}
;;

{ .mfi
      ldfe           FR_A0 = [GR_P_POINT_1], 224
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfe           FR_A1 = [GR_P_POINT_2]
      fms.s1         FR_LocArg = FR_AbsArg, f1, FR_Xc       // xloc = x - x[i]
      nop.i          0
}
;;

{ .mfi
      ldfd           FR_W1 = [GR_ad_W1],0                            // Get W1
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfd           FR_W2 = [GR_ad_W2],0                            // Get W2
      fma.s1         FR_poly = FR_r, FR_C3, FR_C2        // poly = r * A3 + A2
      nop.i          0
}
;;

{ .mfi
      ldfs           FR_T1 = [GR_ad_T1],0                            // Get T1
(p10) fma.s1         FR_Xpdx_lo = FR_AbsArg,f1, FR_Tmp//lo of x + dx , x <= dx
      nop.i          0
}
{ .mfi
      ldfs           FR_T2 = [GR_ad_T2],0                            // Get T2
(p9)  fma.s1         FR_Xpdx_lo = FR_dx,f1, FR_Tmp   // lo  of  x + dx, x > dx
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fnma.s1        FR_Tmp1 = FR_Xpdx_hi, FR_U, FR_2        //  N-R, iter. N1
      nop.i          0
}
{ .mfi
      nop.m          0
      fmpy.s1        FR_rsq = FR_r, FR_r                        // rsq = r * r
      nop.i          0
}
;;

{ .mfi
      setf.exp       FR_scale = GR_exp_2_k                  // Set scale = 2^k
      fma.s1         FR_P15_1_1 = FR_LocArg, FR_LocArg, f0          // xloc ^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_0_1 = FR_A15, FR_LocArg, FR_A14
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_1_2 = FR_A13, FR_LocArg, FR_A12
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_poly = FR_r, FR_poly, FR_C1    // poly = r * poly + A1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_2_1 = FR_A9, FR_LocArg, FR_A8
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_2_2 = FR_A11, FR_LocArg, FR_A10
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_U = FR_U, FR_Tmp1, f0                //  N-R, iter. N1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_3_1 = FR_A5, FR_LocArg, FR_A4
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_3_2 = FR_A7, FR_LocArg, FR_A6
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_4_2 = FR_A3, FR_LocArg, FR_A2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_W = FR_W1, FR_W2, FR_W2            // W = W1 * W2 + W2
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fmpy.s1        FR_T = FR_T1, FR_T2                        // T = T1 * T2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_7_1 = FR_P15_0_1, FR_P15_1_1, FR_P15_1_2
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_7_2 = FR_P15_1_1, FR_P15_1_1, f0         // xloc^4
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_8_1 = FR_P15_1_1, FR_P15_2_2, FR_P15_2_1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fnma.s1        FR_Tmp = FR_Xpdx_hi, FR_U, FR_2         //  N-R, iter. N2
      nop.i          0
}

{ .mfi
      nop.m          0
      fma.s1         FR_poly = FR_rsq, FR_poly, FR_r  // poly = rsq * poly + r
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_9_1 = FR_P15_1_1, FR_P15_4_2, FR_A0
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_9_2 = FR_P15_1_1, FR_P15_3_2, FR_P15_3_1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_W = FR_W, f1, FR_W1                      // W = W + W1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_T_scale = FR_T, FR_scale, f0    // T_scale = T * scale
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_13_1 = FR_P15_7_2, FR_P15_7_1, FR_P15_8_1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_U = FR_U, FR_Tmp, f0                 //  N-R, iter. N2
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_14_1 = FR_P15_7_2, FR_P15_9_2, FR_P15_9_1
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_14_2 = FR_P15_7_2, FR_P15_7_2, f0        // xloc^8
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_M =  FR_T_scale, FR_S, f0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fnma.s1        FR_Tmp = FR_Xpdx_hi, FR_U, FR_2         //  N-R, iter. N3
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_Q = FR_P15_14_2, FR_P15_13_1, FR_P15_14_1
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fms.s1         FR_H = FR_W, f1, FR_xsq_lo              // H = W - xsq_lo
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_U = FR_U, FR_Tmp, f0                 //  N-R, iter. N3
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_Q = FR_A1, FR_LocArg, FR_Q
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fnma.s1        FR_Tmp = FR_Xpdx_hi, FR_U, f1                   // for du
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_R = FR_H, FR_poly, FR_poly
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_res_pos_x_hi = FR_M, FR_U, f0                 //  M *U
      nop.i          0

}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_R = FR_R, f1, FR_H            // R = H + P(r) + H*P(r)
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s0         FR_Tmpf = f8, f1, f0                          //  flag  d
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fnma.s1        FR_dU = FR_Xpdx_lo, FR_U, FR_Tmp
      nop.i          0
}
;;

// p7: we begin to calculate y(x) = 2 - erfcl(-x) in multi precision
// for -6.5 <= x < 0
{ .mfi
      nop.m          0
      fms.s1         FR_res_pos_x_lo = FR_M, FR_U, FR_res_pos_x_hi
      nop.i          0

}
{ .mfi
      nop.m          0
(p7)  fnma.s1        FR_Tmp1 = FR_res_pos_x_hi, f1, FR_2           //p7: x < 0
      nop.i          0

}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_G = FR_R, FR_Q, FR_Q
      nop.i          0

}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_Tmp = FR_R, f1, FR_dU                       //  R + du
      nop.i          0

}
;;

{ .mfi
      nop.m          0
(p7)  fnma.s1        FR_Tmp2 = FR_Tmp1, f1, FR_2                   //p7: x < 0
      nop.i          0

}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_G = FR_G, f1, FR_Tmp
      nop.i          0

}
;;

{ .mfi
      nop.m          0
(p7)  fnma.s1        FR_Tmp2 = FR_res_pos_x_hi, f1, FR_Tmp2        //p7: x < 0
      nop.i          0

}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_V = FR_G, FR_res_pos_x_hi, f0          // V = G * M *U
      nop.i          0

}
;;

{ .mfi
      nop.m          0
(p7)  fma.s1         FR_res_pos_x_lo = FR_res_pos_x_lo, f1, FR_V   //p7: x < 0
      nop.i          0

}
;;

{ .mfi
      nop.m          0
(p7)  fnma.s1        FR_Tmp2 = FR_res_pos_x_lo, f1, FR_Tmp2        //p7: x < 0
      nop.i          0

}
;;


//p6: result for     0 < x < = POS_ARG_ASYMP
//p7: result for   - NEG_ARG_ASYMP  <= x < 0
//p8: exit   for   - NEG_ARG_ASYMP  <=   x < UnfBound

ERFC_RESULT:
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m          0
(p6)  fma.s0         f8 = FR_M, FR_U, FR_V                       // p6: x >= 0
      nop.i          0
}
{ .mfb
      mov            GR_Parameter_TAG = 207
(p7)  fma.s0         f8 = FR_Tmp2, f1, FR_Tmp1                    // p7: x < 0
(p8)  br.ret.sptk    b0
};;

GLOBAL_LIBM_END(erfcl)
libm_alias_ldouble_other (erfc, erfc)

// call via (p15) br.cond.spnt   __libm_error_region
//          for  x > POS_ARG_ASYMP
// or
//
// after .endp erfcl for UnfBound < = x < = POS_ARG_ASYMP

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp                       // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                                 // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                                      // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                                           // Save gp
};;
{ .mmi
        stfe [GR_Parameter_Y] = FR_Y,16          // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp                      // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                                           // Save b0
};;
.body
{ .mib
        stfe [GR_Parameter_X] = FR_X             // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y    // Parameter 3 address
        nop.b 0
}
{ .mib
        stfe [GR_Parameter_Y] = FR_RESULT        // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]        // Get return result off stack
.restore sp
        add   sp = 64,sp                              // Restore stack pointer
        mov   b0 = GR_SAVE_B0                        // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                                    // Restore gp
        mov   ar.pfs = GR_SAVE_PFS                           // Restore ar.pfs
        br.ret.sptk     b0                                           // Return
};;

LOCAL_LIBM_END(__libm_error_region)
.type   __libm_error_support#,@function
.global __libm_error_support#
