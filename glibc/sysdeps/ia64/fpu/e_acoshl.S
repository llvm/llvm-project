.file "acoshl.s"


// Copyright (c) 2000 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2000 by the Intel Numerics Group, Intel Corporation
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
//*********************************************************************
//
// History:
// 10/01/01 Initial version
// 10/10/01 Performance inproved
// 12/11/01 Changed huges_logp to not be global
// 01/02/02 Corrected .restore syntax
// 05/20/02 Cleaned up namespace and sf0 syntax
// 08/14/02 Changed mli templates to mlx
// 02/06/03 Reorganized data tables
// 03/31/05 Reformatted delimiters between data tables
//
//*********************************************************************
//
// API
//==============================================================
// long double acoshl(long double);
//
// Overview of operation
//==============================================================
//
// There are 6 paths:
// 1. x = 1
//    Return acoshl(x) = 0;
//
// 2. x < 1
//    Return acoshl(x) = Nan (Domain error, error handler call with tag 135);
//
// 3. x = [S,Q]Nan or +INF
//    Return acoshl(x) = x + x;
//
// 4. 'Near 1': 1 < x < 1+1/8
//    Return acoshl(x) = sqrtl(2*y)*(1-P(y)/Q(y)),
//                   where y = 1, P(y)/Q(y) - rational approximation
//
// 5. 'Huges': x > 0.5*2^64
//    Return acoshl(x) = (logl(2*x-1));
//
// 6. 'Main path': 1+1/8 < x < 0.5*2^64
//    b_hi + b_lo = x + sqrt(x^2 - 1);
//    acoshl(x) = logl_special(b_hi, b_lo);
//
// Algorithm description
//==============================================================
//
// I. Near 1 path algorithm
// **************************************************************
// The formula is acoshl(x) = sqrtl(2*y)*(1-P(y)/Q(y)),
//                 where y = 1, P(y)/Q(y) - rational approximation
//
// 1) y = x - 1, y2 = 2 * y
//
// 2) Compute in parallel sqrtl(2*y) and P(y)/Q(y)
//    a) sqrtl computation method described below (main path algorithm, item 2))
//       As result we obtain (gg+gl) - multiprecision result
//       as pair of double extended values
//    b) P(y) and Q(y) calculated without any extra precision manipulations
//    c) P/Q division:
//       y = frcpa(Q)         initial approximation of 1/Q
//       z = P*y              initial approximation of P/Q
//
//       e = 1 - b*y
//       e2 = e + e^2
//       e1 = e^2
//       y1 = y + y*e2 = y + y*(e+e^2)
//
//       e3 = e + e1^2
//       y2 = y + y1*e3 = y + y*(e+e^2+..+e^6)
//
//       r = P - Q*z
//       e = 1 - Q*y2
//       xx = z + r*y2         high part of a/b
//
//       y3 = y2 + y2*e4
//       r1 = P  - Q*xx
//       xl = r1*y3            low part of a/b
//
// 3) res = sqrt(2*y) - sqrt(2*y)*(P(y)/Q(y)) =
//        = (gg+gl) - (gg + gl)*(xx+xl);
//
//    a) hh = gg*xx; hl = gg*xl; lh = gl*xx; ll = gl*xl;
//    b) res = ((((gl + ll) + lh) + hl) + hh) + gg;
//       (exactly in this order)
//
// II. Main path algorithm
// ( thanks to Peter Markstein for the idea of sqrt(x^2+1) computation! )
// **********************************************************************
//
// There are 3 parts of x+sqrt(x^2-1) computation:
//
//  1) m2 = (m2_hi+m2_lo) = x^2-1 obtaining
//     ------------------------------------
//     m2_hi = x2_hi - 1, where x2_hi = x * x;
//     m2_lo = x2_lo + p1_lo, where
//                            x2_lo = FMS(x*x-x2_hi),
//                            p1_lo = (1 + m2_hi) - x2_hi;
//
//  2) g = (g_hi+g_lo) = sqrt(m2) = sqrt(m2_hi+m2_lo)
//     ----------------------------------------------
//     r = invsqrt(m2_hi) (8-bit reciprocal square root approximation);
//     g = m2_hi * r (first 8 bit-approximation of sqrt);
//
//     h = 0.5 * r;
//     e = 0.5 - g * h;
//     g = g * e + g (second 16 bit-approximation of sqrt);
//
//     h = h * e + h;
//     e = 0.5 - g * h;
//     g = g * e + g (third 32 bit-approximation of sqrt);
//
//     h = h * e + h;
//     e = 0.5 - g * h;
//     g_hi = g * e + g (fourth 64 bit-approximation of sqrt);
//
//     Remainder computation:
//     h = h * e + h;
//     d = (m2_hi - g_hi * g_hi) + m2_lo;
//     g_lo = d * h;
//
//  3) b = (b_hi + b_lo) = x + g, where g = (g_hi + g_lo) = sqrt(x^2-1)
//     -------------------------------------------------------------------
//     b_hi = (g_hi + x) + gl;
//     b_lo = (x - b_hi) + g_hi + gl;
//
//  Now we pass b presented as sum b_hi + b_lo to special version
//  of logl function which accept a pair of arguments as
//  mutiprecision value.
//
//  Special log algorithm overview
//  ================================
//   Here we use a table lookup method. The basic idea is that in
//   order to compute logl(Arg) for an argument Arg in [1,2),
//   we construct a value G such that G*Arg is close to 1 and that
//   logl(1/G) is obtainable easily from a table of values calculated
//   beforehand. Thus
//
//      logl(Arg) = logl(1/G) + logl((G*Arg - 1))
//
//   Because |G*Arg - 1| is small, the second term on the right hand
//   side can be approximated by a short polynomial. We elaborate
//   this method in four steps.
//
//   Step 0: Initialization
//
//   We need to calculate logl( X+1 ). Obtain N, S_hi such that
//
//      X = 2^N * ( S_hi + S_lo )   exactly
//
//   where S_hi in [1,2) and S_lo is a correction to S_hi in the sense
//   that |S_lo| <= ulp(S_hi).
//
//   For the special version of logl: S_lo = b_lo
//   !-----------------------------------------------!
//
//   Step 1: Argument Reduction
//
//   Based on S_hi, obtain G_1, G_2, G_3 from a table and calculate
//
//      G := G_1 * G_2 * G_3
//      r := (G * S_hi - 1) + G * S_lo
//
//   These G_j's have the property that the product is exactly
//   representable and that |r| < 2^(-12) as a result.
//
//   Step 2: Approximation
//
//   logl(1 + r) is approximated by a short polynomial poly(r).
//
//   Step 3: Reconstruction
//
//   Finally, logl( X ) = logl( X+1 ) is given by
//
//   logl( X )   =   logl( 2^N * (S_hi + S_lo) )
//                 ~=~  N*logl(2) + logl(1/G) + logl(1 + r)
//                 ~=~  N*logl(2) + logl(1/G) + poly(r).
//
//   For detailed description see logl or log1pl function, regular path.
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f32 -> f95 (64 registers)

// General registers used:
// r32 -> r67 (36 registers)

// Predicate registers used:
// p7 -> p11
// p7  for 'NaNs, Inf' path
// p8  for 'near 1' path
// p9  for 'huges' path
// p10 for x = 1
// p11 for x < 1
//
//*********************************************************************
// IEEE Special Conditions:
//
//    acoshl(+inf)  = +inf
//    acoshl(-inf) = QNaN
//    acoshl(1)    = 0
//    acoshl(x<1)  = QNaN
//    acoshl(SNaN) = QNaN
//    acoshl(QNaN) = QNaN
//

// Data tables
//==============================================================

RODATA
.align 64

// Near 1 path rational approximation coefficients
LOCAL_OBJECT_START(Poly_P)
data8 0xB0978143F695D40F, 0x3FF1  // .84205539791447100108478906277453574946e-4
data8 0xB9800D841A8CAD29, 0x3FF6  // .28305085180397409672905983082168721069e-2
data8 0xC889F455758C1725, 0x3FF9  // .24479844297887530847660233111267222945e-1
data8 0x9BE1DFF006F45F12, 0x3FFB  // .76114415657565879842941751209926938306e-1
data8 0x9E34AF4D372861E0, 0x3FFB  // .77248925727776366270605984806795850504e-1
data8 0xF3DC502AEE14C4AE, 0x3FA6  // .3077953476682583606615438814166025592e-26
LOCAL_OBJECT_END(Poly_P)

//
LOCAL_OBJECT_START(Poly_Q)
data8 0xF76E3FD3C7680357, 0x3FF1  // .11798413344703621030038719253730708525e-3
data8 0xD107D2E7273263AE, 0x3FF7  // .63791065024872525660782716786703188820e-2
data8 0xB609BE5CDE206AEF, 0x3FFB  // .88885771950814004376363335821980079985e-1
data8 0xF7DEACAC28067C8A, 0x3FFD  // .48412074662702495416825113623936037072302
data8 0x8F9BE5890CEC7E38, 0x3FFF  // 1.1219450873557867470217771071068369729526
data8 0xED4F06F3D2BC92D1, 0x3FFE  // .92698710873331639524734537734804056798748
LOCAL_OBJECT_END(Poly_Q)

// Q coeffs
LOCAL_OBJECT_START(Constants_Q)
data4  0x00000000,0xB1721800,0x00003FFE,0x00000000
data4  0x4361C4C6,0x82E30865,0x0000BFE2,0x00000000
data4  0x328833CB,0xCCCCCAF2,0x00003FFC,0x00000000
data4  0xA9D4BAFB,0x80000077,0x0000BFFD,0x00000000
data4  0xAAABE3D2,0xAAAAAAAA,0x00003FFD,0x00000000
data4  0xFFFFDAB7,0xFFFFFFFF,0x0000BFFD,0x00000000
LOCAL_OBJECT_END(Constants_Q)

// Z1 - 16 bit fixed
LOCAL_OBJECT_START(Constants_Z_1)
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

// G1 and H1 - IEEE single and h1 - IEEE double
LOCAL_OBJECT_START(Constants_G_H_h1)
data4  0x3F800000,0x00000000
data8  0x0000000000000000
data4  0x3F70F0F0,0x3D785196
data8  0x3DA163A6617D741C
data4  0x3F638E38,0x3DF13843
data8  0x3E2C55E6CBD3D5BB
data4  0x3F579430,0x3E2FF9A0
data8  0xBE3EB0BFD86EA5E7
data4  0x3F4CCCC8,0x3E647FD6
data8  0x3E2E6A8C86B12760
data4  0x3F430C30,0x3E8B3AE7
data8  0x3E47574C5C0739BA
data4  0x3F3A2E88,0x3EA30C68
data8  0x3E20E30F13E8AF2F
data4  0x3F321640,0x3EB9CEC8
data8  0xBE42885BF2C630BD
data4  0x3F2AAAA8,0x3ECF9927
data8  0x3E497F3497E577C6
data4  0x3F23D708,0x3EE47FC5
data8  0x3E3E6A6EA6B0A5AB
data4  0x3F1D89D8,0x3EF8947D
data8  0xBDF43E3CD328D9BE
data4  0x3F17B420,0x3F05F3A1
data8  0x3E4094C30ADB090A
data4  0x3F124920,0x3F0F4303
data8  0xBE28FBB2FC1FE510
data4  0x3F0D3DC8,0x3F183EBF
data8  0x3E3A789510FDE3FA
data4  0x3F088888,0x3F20EC80
data8  0x3E508CE57CC8C98F
data4  0x3F042108,0x3F29516A
data8  0xBE534874A223106C
LOCAL_OBJECT_END(Constants_G_H_h1)

// Z2 - 16 bit fixed
LOCAL_OBJECT_START(Constants_Z_2)
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

// G2 and H2 - IEEE single and h2 - IEEE double
LOCAL_OBJECT_START(Constants_G_H_h2)
data4  0x3F800000,0x00000000
data8  0x0000000000000000
data4  0x3F7F00F8,0x3B7F875D
data8  0x3DB5A11622C42273
data4  0x3F7E03F8,0x3BFF015B
data8  0x3DE620CF21F86ED3
data4  0x3F7D08E0,0x3C3EE393
data8  0xBDAFA07E484F34ED
data4  0x3F7C0FC0,0x3C7E0586
data8  0xBDFE07F03860BCF6
data4  0x3F7B1880,0x3C9E75D2
data8  0x3DEA370FA78093D6
data4  0x3F7A2328,0x3CBDC97A
data8  0x3DFF579172A753D0
data4  0x3F792FB0,0x3CDCFE47
data8  0x3DFEBE6CA7EF896B
data4  0x3F783E08,0x3CFC15D0
data8  0x3E0CF156409ECB43
data4  0x3F774E38,0x3D0D874D
data8  0xBE0B6F97FFEF71DF
data4  0x3F766038,0x3D1CF49B
data8  0xBE0804835D59EEE8
data4  0x3F757400,0x3D2C531D
data8  0x3E1F91E9A9192A74
data4  0x3F748988,0x3D3BA322
data8  0xBE139A06BF72A8CD
data4  0x3F73A0D0,0x3D4AE46F
data8  0x3E1D9202F8FBA6CF
data4  0x3F72B9D0,0x3D5A1756
data8  0xBE1DCCC4BA796223
data4  0x3F71D488,0x3D693B9D
data8  0xBE049391B6B7C239
LOCAL_OBJECT_END(Constants_G_H_h2)

// G3 and H3 - IEEE single and h3 - IEEE double
LOCAL_OBJECT_START(Constants_G_H_h3)
data4  0x3F7FFC00,0x38800100
data8  0x3D355595562224CD
data4  0x3F7FF400,0x39400480
data8  0x3D8200A206136FF6
data4  0x3F7FEC00,0x39A00640
data8  0x3DA4D68DE8DE9AF0
data4  0x3F7FE400,0x39E00C41
data8  0xBD8B4291B10238DC
data4  0x3F7FDC00,0x3A100A21
data8  0xBD89CCB83B1952CA
data4  0x3F7FD400,0x3A300F22
data8  0xBDB107071DC46826
data4  0x3F7FCC08,0x3A4FF51C
data8  0x3DB6FCB9F43307DB
data4  0x3F7FC408,0x3A6FFC1D
data8  0xBD9B7C4762DC7872
data4  0x3F7FBC10,0x3A87F20B
data8  0xBDC3725E3F89154A
data4  0x3F7FB410,0x3A97F68B
data8  0xBD93519D62B9D392
data4  0x3F7FAC18,0x3AA7EB86
data8  0x3DC184410F21BD9D
data4  0x3F7FA420,0x3AB7E101
data8  0xBDA64B952245E0A6
data4  0x3F7F9C20,0x3AC7E701
data8  0x3DB4B0ECAABB34B8
data4  0x3F7F9428,0x3AD7DD7B
data8  0x3D9923376DC40A7E
data4  0x3F7F8C30,0x3AE7D474
data8  0x3DC6E17B4F2083D3
data4  0x3F7F8438,0x3AF7CBED
data8  0x3DAE314B811D4394
data4  0x3F7F7C40,0x3B03E1F3
data8  0xBDD46F21B08F2DB1
data4  0x3F7F7448,0x3B0BDE2F
data8  0xBDDC30A46D34522B
data4  0x3F7F6C50,0x3B13DAAA
data8  0x3DCB0070B1F473DB
data4  0x3F7F6458,0x3B1BD766
data8  0xBDD65DDC6AD282FD
data4  0x3F7F5C68,0x3B23CC5C
data8  0xBDCDAB83F153761A
data4  0x3F7F5470,0x3B2BC997
data8  0xBDDADA40341D0F8F
data4  0x3F7F4C78,0x3B33C711
data8  0x3DCD1BD7EBC394E8
data4  0x3F7F4488,0x3B3BBCC6
data8  0xBDC3532B52E3E695
data4  0x3F7F3C90,0x3B43BAC0
data8  0xBDA3961EE846B3DE
data4  0x3F7F34A0,0x3B4BB0F4
data8  0xBDDADF06785778D4
data4  0x3F7F2CA8,0x3B53AF6D
data8  0x3DCC3ED1E55CE212
data4  0x3F7F24B8,0x3B5BA620
data8  0xBDBA31039E382C15
data4  0x3F7F1CC8,0x3B639D12
data8  0x3D635A0B5C5AF197
data4  0x3F7F14D8,0x3B6B9444
data8  0xBDDCCB1971D34EFC
data4  0x3F7F0CE0,0x3B7393BC
data8  0x3DC7450252CD7ADA
data4  0x3F7F04F0,0x3B7B8B6D
data8  0xBDB68F177D7F2A42
LOCAL_OBJECT_END(Constants_G_H_h3)

// Assembly macros
//==============================================================

// Floating Point Registers

FR_Arg          = f8
FR_Res          = f8


FR_PP0          = f32
FR_PP1          = f33
FR_PP2          = f34
FR_PP3          = f35
FR_PP4          = f36
FR_PP5          = f37
FR_QQ0          = f38
FR_QQ1          = f39
FR_QQ2          = f40
FR_QQ3          = f41
FR_QQ4          = f42
FR_QQ5          = f43

FR_Q1           = f44
FR_Q2           = f45
FR_Q3           = f46
FR_Q4           = f47

FR_Half         = f48
FR_Two          = f49

FR_log2_hi      = f50
FR_log2_lo      = f51


FR_X2           = f52
FR_M2           = f53
FR_M2L          = f54
FR_Rcp          = f55
FR_GG           = f56
FR_HH           = f57
FR_EE           = f58
FR_DD           = f59
FR_GL           = f60
FR_Tmp          = f61


FR_XM1          = f62
FR_2XM1         = f63
FR_XM12         = f64



    // Special logl registers
FR_XLog_Hi      = f65
FR_XLog_Lo      = f66

FR_Y_hi         = f67
FR_Y_lo         = f68

FR_S_hi         = f69
FR_S_lo         = f70

FR_poly_lo      = f71
FR_poly_hi      = f72

FR_G            = f73
FR_H            = f74
FR_h            = f75

FR_G2           = f76
FR_H2           = f77
FR_h2           = f78

FR_r            = f79
FR_rsq          = f80
FR_rcub         = f81

FR_float_N      = f82

FR_G3           = f83
FR_H3           = f84
FR_h3           = f85

FR_2_to_minus_N = f86


   // Near 1  registers
FR_PP           = f65
FR_QQ           = f66


FR_PV6          = f69
FR_PV4          = f70
FR_PV3          = f71
FR_PV2          = f72

FR_QV6          = f73
FR_QV4          = f74
FR_QV3          = f75
FR_QV2          = f76

FR_Y0           = f77
FR_Q0           = f78
FR_E0           = f79
FR_E2           = f80
FR_E1           = f81
FR_Y1           = f82
FR_E3           = f83
FR_Y2           = f84
FR_R0           = f85
FR_E4           = f86
FR_Y3           = f87
FR_R1           = f88
FR_X_Hi         = f89
FR_X_lo         = f90

FR_HH           = f91
FR_LL           = f92
FR_HL           = f93
FR_LH           = f94



	// Error handler registers
FR_Arg_X        = f95
FR_Arg_Y        = f0


// General Purpose Registers

    // General prolog registers
GR_PFS          = r32
GR_OneP125      = r33
GR_TwoP63       = r34
GR_Arg          = r35
GR_Half         = r36

    // Near 1 path registers
GR_Poly_P       = r37
GR_Poly_Q       = r38

    // Special logl registers
GR_Index1       = r39
GR_Index2       = r40
GR_signif       = r41
GR_X_0          = r42
GR_X_1          = r43
GR_X_2          = r44
GR_minus_N      = r45
GR_Z_1          = r46
GR_Z_2          = r47
GR_N            = r48
GR_Bias         = r49
GR_M            = r50
GR_Index3       = r51
GR_exp_2tom80   = r52
GR_exp_mask     = r53
GR_exp_2tom7    = r54
GR_ad_ln10      = r55
GR_ad_tbl_1     = r56
GR_ad_tbl_2     = r57
GR_ad_tbl_3     = r58
GR_ad_q         = r59
GR_ad_z_1       = r60
GR_ad_z_2       = r61
GR_ad_z_3       = r62

//
// Added for unwind support
//
GR_SAVE_PFS         = r32
GR_SAVE_B0          = r33
GR_SAVE_GP          = r34

GR_Parameter_X      = r64
GR_Parameter_Y      = r65
GR_Parameter_RESULT = r66
GR_Parameter_TAG    = r67



.section .text
GLOBAL_LIBM_ENTRY(acoshl)

{ .mfi
      alloc      GR_PFS       = ar.pfs,0,32,4,0     // Local frame allocation
      fcmp.lt.s1 p11, p0      = FR_Arg, f1          // if arg is less than 1
      mov	     GR_Half      = 0xfffe              // 0.5's exp
}
{ .mfi
      addl       GR_Poly_Q    = @ltoff(Poly_Q), gp  // Address of Q-coeff table
      fma.s1     FR_X2        = FR_Arg, FR_Arg, f0  // Obtain x^2
      addl       GR_Poly_P    = @ltoff(Poly_P), gp  // Address of P-coeff table
};;

{ .mfi
      getf.d     GR_Arg       = FR_Arg        // get argument as double (int64)
      fma.s0        FR_Two       = f1, f1, f1    // construct 2.0
      addl       GR_ad_z_1    = @ltoff(Constants_Z_1#),gp // logl tables
}
{ .mlx
      nop.m 0
      movl       GR_TwoP63    = 0x43E8000000000000 // 0.5*2^63 (huge arguments)
};;

{ .mfi
      ld8        GR_Poly_P    = [GR_Poly_P]  // get actual P-coeff table address
      fcmp.eq.s1 p10, p0      = FR_Arg, f1   // if arg == 1 (return 0)
      nop.i 0
}
{ .mlx
      ld8        GR_Poly_Q    = [GR_Poly_Q]  // get actual Q-coeff table address
      movl       GR_OneP125   = 0x3FF2000000000000  // 1.125 (near 1 path bound)
};;

{ .mfi
      ld8        GR_ad_z_1    = [GR_ad_z_1]      // Get pointer to Constants_Z_1
      fclass.m   p7,p0        = FR_Arg, 0xe3       // if arg NaN inf
      cmp.le     p9, p0       = GR_TwoP63, GR_Arg // if arg > 0.5*2^63 ('huges')
}
{ .mfb
      cmp.ge     p8, p0       = GR_OneP125, GR_Arg // if arg<1.125 -near 1 path
	  fms.s1     FR_XM1       = FR_Arg, f1, f1     // X0 = X-1 (for near 1 path)
(p11) br.cond.spnt acoshl_lt_pone                  // error branch (less than 1)
};;

{ .mmi
      setf.exp	FR_Half       = GR_Half     // construct 0.5
(p9)  setf.s    FR_XLog_Lo    = r0          // Low of logl arg=0 (Huges path)
      mov        GR_exp_mask  = 0x1FFFF         // Create exponent mask
};;

{ .mmf
(p8)  ldfe       FR_PP5       = [GR_Poly_P],16     // Load P5
(p8)  ldfe       FR_QQ5       = [GR_Poly_Q],16     // Load Q5
      fms.s1     FR_M2        = FR_X2, f1, f1      // m2 = x^2 - 1
};;

{ .mfi
(p8)  ldfe       FR_QQ4       = [GR_Poly_Q],16         // Load Q4
      fms.s1     FR_M2L       = FR_Arg, FR_Arg, FR_X2  // low part of
	                                                   //    m2 = fma(X*X - m2)
      add        GR_ad_tbl_1  = 0x040, GR_ad_z_1    // Point to Constants_G_H_h1
}
{ .mfb
(p8)  ldfe       FR_PP4       = [GR_Poly_P],16     // Load P4
(p7)  fma.s0     FR_Res       = FR_Arg,f1,FR_Arg   // r = a + a (Nan, Inf)
(p7)  br.ret.spnt b0                               // return    (Nan, Inf)
};;

{ .mfi
(p8)  ldfe       FR_PP3       = [GR_Poly_P],16      // Load P3
      nop.f 0
      add        GR_ad_q      = -0x60, GR_ad_z_1    // Point to Constants_P
}
{ .mfb
(p8)  ldfe       FR_QQ3       = [GR_Poly_Q],16      // Load Q3
(p9)  fms.s1 FR_XLog_Hi       = FR_Two, FR_Arg, f1  // Hi  of log arg = 2*X-1
(p9)  br.cond.spnt huges_logl                       // special version of log
}
;;

{ .mfi
(p8)  ldfe       FR_PP2       = [GR_Poly_P],16       // Load P2
(p8)  fma.s1     FR_2XM1      = FR_Two, FR_XM1, f0   // 2X0 = 2 * X0
      add        GR_ad_z_2    = 0x140, GR_ad_z_1    // Point to Constants_Z_2
}
{ .mfb
(p8)  ldfe       FR_QQ2       = [GR_Poly_Q],16       // Load Q2
(p10) fma.s0   FR_Res         = f0,f1,f0             // r = 0  (arg = 1)
(p10) br.ret.spnt b0                                 // return (arg = 1)
};;

{ .mmi
(p8)  ldfe       FR_PP1       = [GR_Poly_P],16       // Load P1
(p8)  ldfe       FR_QQ1       = [GR_Poly_Q],16       // Load Q1
      add        GR_ad_tbl_2  = 0x180, GR_ad_z_1    // Point to Constants_G_H_h2
}
;;

{ .mfi
(p8)  ldfe       FR_PP0       = [GR_Poly_P]          // Load P0
      fma.s1     FR_Tmp       = f1, f1, FR_M2        // Tmp = 1 + m2
      add        GR_ad_tbl_3  = 0x280, GR_ad_z_1    // Point to Constants_G_H_h3
}
{ .mfb
(p8)  ldfe       FR_QQ0       = [GR_Poly_Q]
      nop.f 0
(p8)  br.cond.spnt near_1                            // near 1 path
};;
{ .mfi
      ldfe       FR_log2_hi   = [GR_ad_q],16      // Load log2_hi
      nop.f 0
      mov        GR_Bias      = 0x0FFFF                  // Create exponent bias
};;
{ .mfi
      nop.m 0
      frsqrta.s1 FR_Rcp, p0   = FR_M2           // Rcp = 1/m2 reciprocal appr.
      nop.i 0
};;

{ .mfi
      ldfe       FR_log2_lo   = [GR_ad_q],16     // Load log2_lo
      fms.s1     FR_Tmp       = FR_X2, f1, FR_Tmp  // Tmp =  x^2 - Tmp
      nop.i 0
};;

{ .mfi
      ldfe       FR_Q4        = [GR_ad_q],16          // Load Q4
      fma.s1     FR_GG        = FR_Rcp, FR_M2, f0   // g = Rcp * m2
                                               // 8 bit Newton Raphson iteration
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HH 		  = FR_Half, FR_Rcp, f0      // h = 0.5 * Rcp
      nop.i 0
};;
{ .mfi
      ldfe       FR_Q3        = [GR_ad_q],16   // Load Q3
      fnma.s1    FR_EE        = FR_GG, FR_HH, FR_Half   // e = 0.5 - g * h
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_M2L       = FR_Tmp, f1, FR_M2L  // low part of m2 = Tmp+m2l
      nop.i 0
};;

{ .mfi
      ldfe       FR_Q2        = [GR_ad_q],16      // Load Q2
      fma.s1     FR_GG        = FR_GG, FR_EE, FR_GG     // g = g * e + g
                                              // 16 bit Newton Raphson iteration
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HH        = FR_HH, FR_EE, FR_HH     // h = h * e + h
      nop.i 0
};;

{ .mfi
      ldfe       FR_Q1        = [GR_ad_q]                // Load Q1
      fnma.s1    FR_EE        = FR_GG, FR_HH, FR_Half   // e = 0.5 - g * h
      nop.i 0
};;
{ .mfi
      nop.m 0
      fma.s1    FR_GG         = FR_GG, FR_EE, FR_GG     // g = g * e + g
                                              // 32 bit Newton Raphson iteration
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_HH         = FR_HH, FR_EE, FR_HH     // h = h * e + h
      nop.i 0
};;

{ .mfi
      nop.m 0
      fnma.s1   FR_EE         = FR_GG, FR_HH, FR_Half   // e = 0.5 - g * h
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_GG         = FR_GG, FR_EE, FR_GG     // g = g * e + g
                                              // 64 bit Newton Raphson iteration
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_HH         = FR_HH, FR_EE, FR_HH     // h = h * e + h
      nop.i 0
};;

{ .mfi
      nop.m 0
      fnma.s1   FR_DD         = FR_GG, FR_GG, FR_M2  // Remainder d = g * g - p2
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_XLog_Hi     = FR_Arg, f1, FR_GG // bh = z + gh
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_DD         = FR_DD, f1, FR_M2L       // add p2l: d = d + p2l
      nop.i 0
};;

{ .mfi
      getf.sig  GR_signif     = FR_XLog_Hi     // Get significand of x+1
      nop.f 0
      mov       GR_exp_2tom7  = 0x0fff8        // Exponent of 2^-7
};;

{ .mfi
      nop.m 0
      fma.s1    FR_GL         = FR_DD, FR_HH, f0        // gl = d * h
      extr.u    GR_Index1     = GR_signif, 59, 4    // Get high 4 bits of signif
}
{ .mfi
      nop.m 0
      fma.s1    FR_XLog_Hi     = FR_DD,  FR_HH, FR_XLog_Hi // bh = bh + gl
      nop.i 0
};;



{ .mmi
      shladd    GR_ad_z_1     = GR_Index1, 2, GR_ad_z_1  // Point to Z_1
      shladd    GR_ad_tbl_1   = GR_Index1, 4, GR_ad_tbl_1  // Point to G_1
      extr.u    GR_X_0        = GR_signif, 49, 15 // Get high 15 bits of signif.
};;

{ .mmi
      ld4       GR_Z_1        = [GR_ad_z_1]    // Load Z_1
      nop.m 0
      nop.i 0
};;

{ .mmi
      ldfps     FR_G, FR_H    = [GR_ad_tbl_1],8     // Load G_1, H_1
      nop.m 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      fms.s1    FR_XLog_Lo     = FR_Arg,  f1,   FR_XLog_Hi // bl = x - bh
      pmpyshr2.u GR_X_1       = GR_X_0,GR_Z_1,15  // Get bits 30-15 of X_0 * Z_1
};;

// WE CANNOT USE GR_X_1 IN NEXT 3 CYCLES BECAUSE OF POSSIBLE 10 CLOCKS STALL!
// "DEAD" ZONE!

{ .mfi
      nop.m 0
      nop.f 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      fmerge.se FR_S_hi       =  f1,FR_XLog_Hi            // Form |x+1|
      nop.i 0
};;


{ .mmi
      getf.exp  GR_N          =  FR_XLog_Hi    // Get N = exponent of x+1
      ldfd      FR_h          = [GR_ad_tbl_1]        // Load h_1
      nop.i 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      extr.u    GR_Index2     = GR_X_1, 6, 4      // Extract bits 6-9 of X_1
};;

{ .mfi
      shladd    GR_ad_tbl_2   = GR_Index2, 4, GR_ad_tbl_2  // Point to G_2
      fma.s1    FR_XLog_Lo    = FR_XLog_Lo, f1, FR_GG // bl = bl + gg
      mov       GR_exp_2tom80 = 0x0ffaf           // Exponent of 2^-80
}
{ .mfi
      shladd    GR_ad_z_2     = GR_Index2, 2, GR_ad_z_2  // Point to Z_2
      nop.f 0
      sub       GR_N          = GR_N, GR_Bias // sub bias from exp
};;

{ .mmi
      ldfps     FR_G2, FR_H2  = [GR_ad_tbl_2],8       // Load G_2, H_2
      ld4       GR_Z_2        = [GR_ad_z_2]                // Load Z_2
      sub       GR_minus_N    = GR_Bias, GR_N         // Form exponent of 2^(-N)
};;

{ .mmi
      ldfd      FR_h2         = [GR_ad_tbl_2]             // Load h_2
      nop.m 0
      nop.i 0
};;

{ .mmi
      setf.sig  FR_float_N    = GR_N        // Put integer N into rightmost sign
      setf.exp  FR_2_to_minus_N = GR_minus_N   // Form 2^(-N)
      pmpyshr2.u GR_X_2       = GR_X_1,GR_Z_2,15 // Get bits 30-15 of X_1 * Z_2
};;

// WE CANNOT USE GR_X_2 IN NEXT 3 CYCLES ("DEAD" ZONE!)
// BECAUSE OF POSSIBLE 10 CLOCKS STALL!
// (Just nops added - nothing to do here)

{ .mfi
      nop.m 0
      fma.s1    FR_XLog_Lo     = FR_XLog_Lo, f1, FR_GL // bl = bl + gl
      nop.i 0
};;
{ .mfi
      nop.m 0
      nop.f 0
      nop.i 0
};;
{ .mfi
      nop.m 0
      nop.f 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      extr.u    GR_Index3     = GR_X_2, 1, 5         // Extract bits 1-5 of X_2
};;

{ .mfi
      shladd    GR_ad_tbl_3   = GR_Index3, 4, GR_ad_tbl_3  // Point to G_3
      nop.f 0
      nop.i 0
};;

{ .mfi
      ldfps     FR_G3, FR_H3  = [GR_ad_tbl_3],8   // Load G_3, H_3
      nop.f 0
      nop.i 0
};;

{ .mfi
      ldfd      FR_h3         = [GR_ad_tbl_3]            // Load h_3
	  fcvt.xf   FR_float_N    = FR_float_N
      nop.i 0
};;

{ .mfi
      nop.m 0
      fmpy.s1   FR_G          = FR_G, FR_G2              // G = G_1 * G_2
      nop.i 0
}
{ .mfi
      nop.m 0
      fadd.s1   FR_H          = FR_H, FR_H2              // H = H_1 + H_2
      nop.i 0
};;

{ .mfi
      nop.m 0
      fadd.s1   FR_h          = FR_h, FR_h2              // h = h_1 + h_2
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_S_lo     = FR_XLog_Lo, FR_2_to_minus_N, f0 //S_lo=S_lo*2^(-N)
      nop.i 0
};;

{ .mfi
      nop.m 0
      fmpy.s1   FR_G          = FR_G, FR_G3             // G = (G_1 * G_2) * G_3
      nop.i 0
}
{ .mfi
      nop.m 0
      fadd.s1   FR_H          = FR_H, FR_H3             // H = (H_1 + H_2) + H_3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fadd.s1   FR_h          = FR_h, FR_h3             // h = (h_1 + h_2) + h_3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fms.s1    FR_r          = FR_G, FR_S_hi, f1           // r = G * S_hi - 1
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_Y_hi       = FR_float_N, FR_log2_hi, FR_H // Y_hi=N*log2_hi+H
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_h          = FR_float_N, FR_log2_lo, FR_h  // h=N*log2_lo+h
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_r          = FR_G, FR_S_lo, FR_r  // r=G*S_lo+(G*S_hi-1)
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_poly_lo    = FR_r, FR_Q4, FR_Q3      // poly_lo = r * Q4 + Q3
      nop.i 0
}
{ .mfi
      nop.m 0
      fmpy.s1   FR_rsq        = FR_r, FR_r              // rsq = r * r
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_poly_lo    = FR_poly_lo, FR_r, FR_Q2 // poly_lo=poly_lo*r+Q2
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_rcub       = FR_rsq, FR_r, f0        // rcub = r^3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_poly_hi    = FR_Q1, FR_rsq, FR_r // poly_hi = Q1*rsq + r
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_poly_lo    = FR_poly_lo, FR_rcub, FR_h//poly_lo=poly_lo*r^3+h
      nop.i 0
};;

{ .mfi
      nop.m 0
      fadd.s0   FR_Y_lo       = FR_poly_hi, FR_poly_lo
	                                                     // Y_lo=poly_hi+poly_lo
      nop.i 0
};;

{ .mfb
      nop.m 0
      fadd.s0   FR_Res        = FR_Y_lo,FR_Y_hi    // Result=Y_lo+Y_hi
      br.ret.sptk   b0                         // Common exit for 2^-7 < x < inf
};;


huges_logl:
{ .mmi
      getf.sig   GR_signif    = FR_XLog_Hi               // Get significand of x+1
      mov        GR_exp_2tom7 = 0x0fff8            // Exponent of 2^-7
      nop.i 0
};;

{ .mfi
      add        GR_ad_tbl_1  = 0x040, GR_ad_z_1    // Point to Constants_G_H_h1
      nop.f 0
      add        GR_ad_q      = -0x60, GR_ad_z_1    // Point to Constants_P
}
{ .mfi
      add        GR_ad_z_2    = 0x140, GR_ad_z_1    // Point to Constants_Z_2
      nop.f 0
      add        GR_ad_tbl_2  = 0x180, GR_ad_z_1    // Point to Constants_G_H_h2
};;

{ .mfi
      add        GR_ad_tbl_3  = 0x280, GR_ad_z_1    // Point to Constants_G_H_h3
      nop.f 0
      extr.u     GR_Index1    = GR_signif, 59, 4    // Get high 4 bits of signif
};;

{ .mfi
      shladd     GR_ad_z_1    = GR_Index1, 2, GR_ad_z_1  // Point to Z_1
      nop.f 0
      extr.u     GR_X_0       = GR_signif, 49, 15 // Get high 15 bits of signif.
};;

{ .mfi
      ld4        GR_Z_1       = [GR_ad_z_1]     // Load Z_1
      nop.f 0
      mov        GR_exp_mask  = 0x1FFFF         // Create exponent mask
}
{ .mfi
      shladd     GR_ad_tbl_1  = GR_Index1, 4, GR_ad_tbl_1 // Point to G_1
      nop.f 0
      mov        GR_Bias      = 0x0FFFF                  // Create exponent bias
};;

{ .mfi
      ldfps      FR_G, FR_H   = [GR_ad_tbl_1],8     // Load G_1, H_1
      fmerge.se  FR_S_hi      =  f1,FR_XLog_Hi            // Form |x|
      nop.i 0
};;

{ .mmi
      getf.exp   GR_N         =  FR_XLog_Hi         // Get N = exponent of x+1
      ldfd       FR_h         = [GR_ad_tbl_1] // Load h_1
      nop.i 0
};;

{ .mfi
      ldfe       FR_log2_hi   = [GR_ad_q],16      // Load log2_hi
      nop.f 0
      pmpyshr2.u GR_X_1       = GR_X_0,GR_Z_1,15  // Get bits 30-15 of X_0 * Z_1
};;

{ .mmi
      ldfe       FR_log2_lo   = [GR_ad_q],16     // Load log2_lo
      sub        GR_N         = GR_N, GR_Bias
      mov        GR_exp_2tom80 = 0x0ffaf         // Exponent of 2^-80
};;

{ .mfi
      ldfe       FR_Q4        = [GR_ad_q],16          // Load Q4
      nop.f 0
      sub        GR_minus_N   = GR_Bias, GR_N         // Form exponent of 2^(-N)
};;

{ .mmf
      ldfe       FR_Q3        = [GR_ad_q],16   // Load Q3
      setf.sig   FR_float_N   = GR_N        // Put integer N into rightmost sign
      nop.f 0
};;

{ .mmi
      ldfe       FR_Q2        = [GR_ad_q],16      // Load Q2
	  nop.m 0
      extr.u     GR_Index2    = GR_X_1, 6, 4      // Extract bits 6-9 of X_1
};;

{ .mmi
      ldfe       FR_Q1        = [GR_ad_q]                // Load Q1
      shladd     GR_ad_z_2    = GR_Index2, 2, GR_ad_z_2  // Point to Z_2
      nop.i 0
};;

{ .mmi
      ld4        GR_Z_2       = [GR_ad_z_2]                // Load Z_2
      shladd     GR_ad_tbl_2  = GR_Index2, 4, GR_ad_tbl_2  // Point to G_2
	  nop.i 0
};;

{ .mmi
      ldfps      FR_G2, FR_H2 = [GR_ad_tbl_2],8       // Load G_2, H_2
      nop.m 0
      nop.i 0
};;

{ .mmf
      ldfd       FR_h2        = [GR_ad_tbl_2]         // Load h_2
      setf.exp FR_2_to_minus_N = GR_minus_N   // Form 2^(-N)
      nop.f 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      pmpyshr2.u GR_X_2       = GR_X_1,GR_Z_2,15   // Get bits 30-15 of X_1*Z_2
};;

// WE CANNOT USE GR_X_2 IN NEXT 3 CYCLES ("DEAD" ZONE!)
// BECAUSE OF POSSIBLE 10 CLOCKS STALL!
// (Just nops added - nothing to do here)

{ .mfi
      nop.m 0
      nop.f 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      extr.u     GR_Index3    = GR_X_2, 1, 5          // Extract bits 1-5 of X_2
};;

{ .mfi
      shladd     GR_ad_tbl_3  = GR_Index3, 4, GR_ad_tbl_3  // Point to G_3
	  fcvt.xf    FR_float_N   = FR_float_N
      nop.i 0
};;

{ .mfi
      ldfps      FR_G3, FR_H3 = [GR_ad_tbl_3],8   // Load G_3, H_3
      nop.f 0
      nop.i 0
};;

{ .mfi
      ldfd       FR_h3        = [GR_ad_tbl_3]            // Load h_3
      fmpy.s1    FR_G         = FR_G, FR_G2              // G = G_1 * G_2
      nop.i 0
}
{ .mfi
      nop.m 0
      fadd.s1    FR_H         = FR_H, FR_H2              // H = H_1 + H_2
      nop.i 0
};;

{ .mmf
      nop.m 0
      nop.m 0
      fadd.s1    FR_h         = FR_h, FR_h2              // h = h_1 + h_2
};;

{ .mfi
      nop.m 0
      fmpy.s1    FR_G         = FR_G, FR_G3              // G = (G_1 * G_2)*G_3
      nop.i 0
}
{ .mfi
      nop.m 0
      fadd.s1    FR_H         = FR_H, FR_H3              // H = (H_1 + H_2)+H_3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fadd.s1    FR_h         = FR_h, FR_h3            // h = (h_1 + h_2) + h_3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fms.s1     FR_r         = FR_G, FR_S_hi, f1           // r = G * S_hi - 1
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_Y_hi      = FR_float_N, FR_log2_hi, FR_H // Y_hi=N*log2_hi+H
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_h         = FR_float_N, FR_log2_lo, FR_h  // h = N*log2_lo+h
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_poly_lo   = FR_r, FR_Q4, FR_Q3      // poly_lo = r * Q4 + Q3
      nop.i 0
}
{ .mfi
      nop.m 0
      fmpy.s1    FR_rsq       = FR_r, FR_r              // rsq = r * r
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_poly_lo   = FR_poly_lo, FR_r, FR_Q2 // poly_lo=poly_lo*r+Q2
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_rcub      = FR_rsq, FR_r, f0        // rcub = r^3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_poly_hi   = FR_Q1, FR_rsq, FR_r     // poly_hi = Q1*rsq + r
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_poly_lo   = FR_poly_lo, FR_rcub, FR_h//poly_lo=poly_lo*r^3+h
      nop.i 0
};;
{ .mfi
      nop.m 0
      fadd.s0    FR_Y_lo      = FR_poly_hi, FR_poly_lo  // Y_lo=poly_hi+poly_lo
      nop.i 0
};;
{ .mfb
      nop.m 0
      fadd.s0    FR_Res       = FR_Y_lo,FR_Y_hi    // Result=Y_lo+Y_hi
      br.ret.sptk   b0                        // Common exit
};;


// NEAR ONE INTERVAL
near_1:
{ .mfi
      nop.m 0
      frsqrta.s1 FR_Rcp, p0   = FR_2XM1 // Rcp = 1/x reciprocal appr. &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_PV6       = FR_PP5, FR_XM1, FR_PP4 // pv6 = P5*xm1+P4 $POLY$
      nop.i 0
}
{ .mfi
      nop.m 0
	  fma.s1     FR_QV6       = FR_QQ5, FR_XM1, FR_QQ4 // qv6 = Q5*xm1+Q4 $POLY$
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fma.s1     FR_PV4       = FR_PP3, FR_XM1, FR_PP2 // pv4 = P3*xm1+P2 $POLY$
      nop.i 0
}
{ .mfi
      nop.m 0
	  fma.s1     FR_QV4       = FR_QQ3, FR_XM1, FR_QQ2 // qv4 = Q3*xm1+Q2 $POLY$
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fma.s1     FR_XM12      = FR_XM1, FR_XM1, f0 // xm1^2 = xm1 * xm1 $POLY$
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fma.s1     FR_PV2       = FR_PP1, FR_XM1, FR_PP0 // pv2 = P1*xm1+P0 $POLY$
      nop.i 0
}
{ .mfi
      nop.m 0
	  fma.s1     FR_QV2       = FR_QQ1, FR_XM1, FR_QQ0 // qv2 = Q1*xm1+Q0 $POLY$
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_GG        = FR_Rcp, FR_2XM1, f0 // g = Rcp * x &SQRT&
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HH        = FR_Half, FR_Rcp, f0 // h = 0.5 * Rcp &SQRT&
      nop.i 0
};;


{ .mfi
      nop.m 0
	  fma.s1    FR_PV3       = FR_XM12, FR_PV6, FR_PV4//pv3=pv6*xm1^2+pv4 $POLY$
      nop.i 0
}
{ .mfi
      nop.m 0
	  fma.s1    FR_QV3       = FR_XM12, FR_QV6, FR_QV4//qv3=qv6*xm1^2+qv4 $POLY$
      nop.i 0
};;


{ .mfi
      nop.m 0
      fnma.s1   FR_EE        = FR_GG, FR_HH, FR_Half   // e = 0.5 - g * h &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fma.s1    FR_PP        = FR_XM12, FR_PV3, FR_PV2 //pp=pv3*xm1^2+pv2 $POLY$
      nop.i 0
}
{ .mfi
      nop.m 0
	  fma.s1    FR_QQ        = FR_XM12, FR_QV3, FR_QV2 //qq=qv3*xm1^2+qv2 $POLY$
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_GG        = FR_GG, FR_EE, FR_GG  // g = g * e + g &SQRT&
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HH        = FR_HH, FR_EE, FR_HH  // h = h * e + h &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      frcpa.s1   FR_Y0,p0     = f1,FR_QQ // y = frcpa(b)  #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fnma.s1    FR_EE        = FR_GG, FR_HH, FR_Half // e = 0.5 - g*h &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_Q0        = FR_PP,FR_Y0,f0 // q = a*y  #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fnma.s1    FR_E0        = FR_Y0,FR_QQ,f1 // e = 1 - b*y  #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_GG        = FR_GG, FR_EE, FR_GG // g = g * e + g &SQRT&
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HH        = FR_HH, FR_EE, FR_HH // h = h * e + h &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_E2        = FR_E0,FR_E0,FR_E0 // e2 = e+e^2 #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_E1        = FR_E0,FR_E0,f0 // e1 = e^2 #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fnma.s1   FR_EE        = FR_GG, FR_HH, FR_Half   // e = 0.5 - g * h &SQRT&
      nop.i 0
}
{ .mfi
      nop.m 0
	  fnma.s1   FR_DD        = FR_GG, FR_GG, FR_2XM1   // d = x - g * g &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_Y1        = FR_Y0,FR_E2,FR_Y0 // y1 = y+y*e2 #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_E3        = FR_E1,FR_E1,FR_E0 // e3 = e+e1^2 #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_GG        = FR_DD, FR_HH, FR_GG // g = d * h + g &SQRT&
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HH        = FR_HH, FR_EE, FR_HH // h = h * e + h &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_Y2        = FR_Y1,FR_E3,FR_Y0 // y2 = y+y1*e3 #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fnma.s1    FR_R0        = FR_QQ,FR_Q0,FR_PP // r = a-b*q #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fnma.s1    FR_DD        = FR_GG, FR_GG, FR_2XM1 // d = x - g * g &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fnma.s1    FR_E4        = FR_QQ,FR_Y2,f1    // e4 = 1-b*y2 #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_X_Hi      = FR_R0,FR_Y2,FR_Q0 // x = q+r*y2 #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_GL        = FR_DD, FR_HH, f0   // gl = d * h &SQRT&
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_Y3        = FR_Y2,FR_E4,FR_Y2 // y3 = y2+y2*e4 #DIV#
      nop.i 0
}
{ .mfi
      nop.m 0
      fnma.s1    FR_R1        = FR_QQ,FR_X_Hi,FR_PP // r1 = a-b*x #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_HH        = FR_GG, FR_X_Hi, f0 // hh = gg * x_hi
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_LH        = FR_GL, FR_X_Hi, f0 // lh = gl * x_hi
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_X_lo      = FR_R1,FR_Y3,f0 // x_lo = r1*y3 #DIV#
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1     FR_LL        = FR_GL, FR_X_lo, f0 // ll = gl*x_lo
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1     FR_HL        = FR_GG, FR_X_lo, f0 // hl = gg * x_lo
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fms.s1     FR_Res       = FR_GL,  f1, FR_LL // res = gl + ll
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fms.s1     FR_Res       = FR_Res, f1, FR_LH // res = res + lh
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fms.s1     FR_Res       = FR_Res, f1, FR_HL // res = res + hl
      nop.i 0
};;

{ .mfi
      nop.m 0
	  fms.s1     FR_Res       = FR_Res, f1, FR_HH // res = res + hh
      nop.i 0
};;

{ .mfb
      nop.m 0
	  fma.s0     FR_Res       = FR_Res, f1, FR_GG  // result = res + gg
      br.ret.sptk   b0                     // Exit for near 1 path
};;
// NEAR ONE INTERVAL END




acoshl_lt_pone:
{ .mfi
      nop.m 0
      fmerge.s   FR_Arg_X            = FR_Arg, FR_Arg
      nop.i 0
};;
{ .mfb
      mov        GR_Parameter_TAG    = 135
      frcpa.s0   FR_Res,p0           = f0,f0 // get QNaN,and raise invalid
      br.cond.sptk  __libm_error_region      // exit if x < 1.0
};;

GLOBAL_LIBM_END(acoshl)
libm_alias_ldouble_other (acosh, acosh)



LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add      GR_Parameter_Y      = -32,sp        // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov      GR_SAVE_PFS         = ar.pfs        // Save ar.pfs
}
{ .mfi
.fframe 64
        add      sp                  = -64,sp        // Create new stack
        nop.f 0
        mov      GR_SAVE_GP          = gp            // Save gp
};;

{ .mmi
        stfe     [GR_Parameter_Y]    = FR_Arg_Y,16   // Parameter 2 to stack
        add      GR_Parameter_X      = 16,sp         // Parameter 1 address
.save   b0,GR_SAVE_B0
        mov      GR_SAVE_B0          = b0            // Save b0
};;

.body
{ .mib
        stfe     [GR_Parameter_X]    = FR_Arg_X         // Parameter 1 to stack
        add      GR_Parameter_RESULT = 0,GR_Parameter_Y // Parameter 3 address
        nop.b 0
}
{ .mib
        stfe     [GR_Parameter_Y]    = FR_Res        // Parameter 3 to stack
        add      GR_Parameter_Y      = -16,GR_Parameter_Y
        br.call.sptk b0 = __libm_error_support#      // Error handling function
};;

{ .mmi
        nop.m 0
        nop.m 0
        add      GR_Parameter_RESULT = 48,sp
};;

{ .mmi
        ldfe     f8                  = [GR_Parameter_RESULT]  // Get return res
.restore sp
        add      sp                  = 64,sp       // Restore stack pointer
        mov      b0                  = GR_SAVE_B0  // Restore return address
};;

{ .mib
        mov      gp                  = GR_SAVE_GP  // Restore gp
        mov      ar.pfs              = GR_SAVE_PFS // Restore ar.pfs
        br.ret.sptk b0                             // Return
};;

LOCAL_LIBM_END(__libm_error_region#)

.type   __libm_error_support#,@function
.global __libm_error_support#
