.file "asinhl.s"


// Copyright (c) 2000 - 2003, Intel Corporation
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
// 09/04/01 Initial version
// 09/13/01 Performance improved, symmetry problems fixed
// 10/10/01 Performance improved, split issues removed
// 12/11/01 Changed huges_logp to not be global
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
//
//*********************************************************************
//
// API
//==============================================================
// long double asinhl(long double);
//
// Overview of operation
//==============================================================
//
// There are 6 paths:
// 1. x = 0, [S,Q]Nan or +/-INF
//    Return asinhl(x) = x + x;
//
// 2. x = + denormal
//    Return asinhl(x) = x - x^2;
//
// 3. x = - denormal
//    Return asinhl(x) = x + x^2;
//
// 4. 'Near 0': max denormal < |x| < 1/128
//    Return asinhl(x) = sign(x)*(x+x^3*(c3+x^2*(c5+x^2*(c7+x^2*(c9)))));
//
// 5. 'Huges': |x| > 2^63
//    Return asinhl(x) = sign(x)*(logl(2*x));
//
// 6. 'Main path': 1/128 < |x| < 2^63
//    b_hi + b_lo = x + sqrt(x^2 + 1);
//    asinhl(x) = sign(x)*(log_special(b_hi, b_lo));
//
// Algorithm description
//==============================================================
//
// Main path algorithm
// ( thanks to Peter Markstein for the idea of sqrt(x^2+1) computation! )
// *************************************************************************
//
// There are 3 parts of x+sqrt(x^2+1) computation:
//
//  1) p2 = (p2_hi+p2_lo) = x^2+1 obtaining
//     ------------------------------------
//     p2_hi = x2_hi + 1, where x2_hi = x * x;
//     p2_lo = x2_lo + p1_lo, where
//                            x2_lo = FMS(x*x-x2_hi),
//                            p1_lo = (1 - p2_hi) + x2_hi;
//
//  2) g = (g_hi+g_lo) = sqrt(p2) = sqrt(p2_hi+p2_lo)
//     ----------------------------------------------
//     r = invsqrt(p2_hi) (8-bit reciprocal square root approximation);
//     g = p2_hi * r (first 8 bit-approximation of sqrt);
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
//     d = (p2_hi - g_hi * g_hi) + p2_lo;
//     g_lo = d * h;
//
//  3) b = (b_hi + b_lo) = x + g, where g = (g_hi + g_lo) = sqrt(x^2+1)
//     -------------------------------------------------------------------
//     b_hi = (g_hi + x) + gl;
//     b_lo = (g_hi - b_hi) + x + gl;
//
//  Now we pass b presented as sum b_hi + b_lo to special version
//  of logl function which accept a pair of arguments as
//  'mutiprecision' value.
//
//  Special log algorithm overview
//  ================================
//   Here we use a table lookup method. The basic idea is that in
//   order to compute logl(Arg) = logl (Arg-1) for an argument Arg in [1,2),
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
//   We need to calculate logl( X ). Obtain N, S_hi such that
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
//   Finally,
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
// f32 -> f101 (70 registers)

// General registers used:
// r32 -> r57 (26 registers)

// Predicate registers used:
// p6 -> p11
// p6  for '0, NaNs, Inf' path
// p7  for '+ denormals' path
// p8  for 'near 0' path
// p9  for 'huges' path
// p10 for '- denormals' path
// p11 for negative values
//
// Data tables
//==============================================================

RODATA
.align 64

// C7, C9 'near 0' polynomial coefficients
LOCAL_OBJECT_START(Poly_C_near_0_79)
data8 0xF8DC939BBEDD5A54, 0x00003FF9
data8 0xB6DB6DAB21565AC5, 0x0000BFFA
LOCAL_OBJECT_END(Poly_C_near_0_79)

// C3, C5 'near 0' polynomial coefficients
LOCAL_OBJECT_START(Poly_C_near_0_35)
data8 0x999999999991D582, 0x00003FFB
data8 0xAAAAAAAAAAAAAAA9, 0x0000BFFC
LOCAL_OBJECT_END(Poly_C_near_0_35)

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
FR_AX           = f32
FR_XLog_Hi      = f33
FR_XLog_Lo      = f34

    // Special logl registers
FR_Y_hi         = f35
FR_Y_lo         = f36

FR_Scale        = f37
FR_X_Prime      = f38
FR_S_hi         = f39
FR_W            = f40
FR_G            = f41

FR_H            = f42
FR_wsq          = f43
FR_w4           = f44
FR_h            = f45
FR_w6           = f46

FR_G2           = f47
FR_H2           = f48
FR_poly_lo      = f49
FR_P8           = f50
FR_poly_hi      = f51

FR_P7           = f52
FR_h2           = f53
FR_rsq          = f54
FR_P6           = f55
FR_r            = f56

FR_log2_hi      = f57
FR_log2_lo      = f58

FR_float_N      = f59
FR_Q4           = f60

FR_G3           = f61
FR_H3           = f62
FR_h3           = f63

FR_Q3           = f64
FR_Q2           = f65
FR_1LN10_hi     = f66

FR_Q1           = f67
FR_1LN10_lo     = f68
FR_P5           = f69
FR_rcub         = f70

FR_Neg_One      = f71
FR_Z            = f72
FR_AA           = f73
FR_BB           = f74
FR_S_lo         = f75
FR_2_to_minus_N = f76


    // Huge & Main path prolog registers
FR_Half         = f77
FR_Two          = f78
FR_X2           = f79
FR_P2           = f80
FR_P2L          = f81
FR_Rcp          = f82
FR_GG           = f83
FR_HH           = f84
FR_EE           = f85
FR_DD           = f86
FR_GL           = f87
FR_A            = f88
FR_AL           = f89
FR_B            = f90
FR_BL           = f91
FR_Tmp          = f92

    // Near 0 & Huges path prolog registers
FR_C3           = f93
FR_C5           = f94
FR_C7           = f95
FR_C9           = f96

FR_X3           = f97
FR_X4           = f98
FR_P9           = f99
FR_P5           = f100
FR_P3           = f101


// General Purpose Registers

    // General prolog registers
GR_PFS          = r32
GR_TwoN7        = r40
GR_TwoP63       = r41
GR_ExpMask      = r42
GR_ArgExp       = r43
GR_Half         = r44

    // Near 0 path prolog registers
GR_Poly_C_35    = r45
GR_Poly_C_79    = r46

    // Special logl registers
GR_Index1       = r34
GR_Index2       = r35
GR_signif       = r36
GR_X_0          = r37
GR_X_1          = r38
GR_X_2          = r39
GR_Z_1          = r40
GR_Z_2          = r41
GR_N            = r42
GR_Bias         = r43
GR_M            = r44
GR_Index3       = r45
GR_exp_2tom80   = r45
GR_exp_mask     = r47
GR_exp_2tom7    = r48
GR_ad_ln10      = r49
GR_ad_tbl_1     = r50
GR_ad_tbl_2     = r51
GR_ad_tbl_3     = r52
GR_ad_q         = r53
GR_ad_z_1       = r54
GR_ad_z_2       = r55
GR_ad_z_3       = r56
GR_minus_N      = r57



.section .text
GLOBAL_LIBM_ENTRY(asinhl)

{ .mfi
      alloc     GR_PFS        = ar.pfs,0,27,0,0
      fma.s1    FR_P2         = FR_Arg, FR_Arg, f1  // p2 = x^2 + 1
      mov   	GR_Half       = 0xfffe              // 0.5's exp
}
{ .mfi
      addl      GR_Poly_C_79  = @ltoff(Poly_C_near_0_79), gp // C7, C9 coeffs
      fma.s1    FR_X2         = FR_Arg, FR_Arg, f0           // Obtain x^2
      addl      GR_Poly_C_35  = @ltoff(Poly_C_near_0_35), gp // C3, C5 coeffs
};;

{ .mfi
      getf.exp  GR_ArgExp     = FR_Arg        // get arument's exponent
      fabs      FR_AX         = FR_Arg        // absolute value of argument
      mov       GR_TwoN7      = 0xfff8        // 2^-7 exp
}
{ .mfi
      ld8       GR_Poly_C_79  = [GR_Poly_C_79] // get actual coeff table address
      fma.s0       FR_Two        = f1, f1, f1        // construct 2.0
      mov       GR_ExpMask    = 0x1ffff        // mask for exp
};;

{ .mfi
      ld8       GR_Poly_C_35  = [GR_Poly_C_35] // get actual coeff table address
      fclass.m  p6,p0         = FR_Arg, 0xe7   // if arg NaN inf zero
      mov       GR_TwoP63     = 0x1003e        // 2^63 exp
}
{ .mfi
      addl      GR_ad_z_1     = @ltoff(Constants_Z_1#),gp
      nop.f 0
      nop.i 0
};;

{ .mfi
      setf.exp	FR_Half       = GR_Half              // construct 0.5
      fclass.m  p7,p0         = FR_Arg, 0x09  //  if arg + denorm
      and       GR_ArgExp     = GR_ExpMask, GR_ArgExp // select exp
}
{ .mfb
      ld8       GR_ad_z_1     = [GR_ad_z_1]   // Get pointer to Constants_Z_1
      nop.f 0
      nop.b 0
};;
{ .mfi
      ldfe      FR_C9         = [GR_Poly_C_79],16 // load C9
      fclass.m  p10,p0        = FR_Arg, 0x0a    //  if arg - denorm
      cmp.gt    p8, p0        = GR_TwoN7,  GR_ArgExp // if arg < 2^-7 ('near 0')
}
{ .mfb
      cmp.le    p9, p0        = GR_TwoP63, GR_ArgExp  // if arg > 2^63 ('huges')
(p6)  fma.s0    FR_Res        = FR_Arg,f1,FR_Arg     // r = a + a
(p6)  br.ret.spnt b0                            // return
};;
// (X^2 + 1) computation
{ .mfi
(p8)  ldfe      FR_C5         = [GR_Poly_C_35],16        // load C5
      fms.s1    FR_Tmp        = f1, f1, FR_P2           // Tmp = 1 - p2
      add       GR_ad_tbl_1   = 0x040, GR_ad_z_1    // Point to Constants_G_H_h1
}
{ .mfb
(p8)  ldfe      FR_C7         = [GR_Poly_C_79],16        // load C7
(p7)  fnma.s0   FR_Res        =  FR_Arg,FR_Arg,FR_Arg // r = a - a*a
(p7)  br.ret.spnt b0                              // return
};;

{ .mfi
(p8)  ldfe      FR_C3         = [GR_Poly_C_35],16     // load C3
      fcmp.lt.s1 p11, p12      = FR_Arg, f0     // if arg is negative
      add       GR_ad_q       = -0x60, GR_ad_z_1    // Point to Constants_P
}
{ .mfb
      add       GR_ad_z_2     = 0x140, GR_ad_z_1    // Point to Constants_Z_2
(p10) fma.s0    FR_Res        =  FR_Arg,FR_Arg,FR_Arg // r = a + a*a
(p10) br.ret.spnt b0                             // return
};;

{ .mfi
      add       GR_ad_tbl_2   = 0x180, GR_ad_z_1    // Point to Constants_G_H_h2
      frsqrta.s1 FR_Rcp, p0   = FR_P2           // Rcp = 1/p2 reciprocal appr.
      add       GR_ad_tbl_3   = 0x280, GR_ad_z_1    // Point to Constants_G_H_h3
}
{ .mfi
      nop.m 0
      fms.s1    FR_P2L        = FR_AX, FR_AX, FR_X2 //low part of p2=fma(X*X-p2)
      mov       GR_Bias       = 0x0FFFF            // Create exponent bias
};;

{ .mfb
      nop.m 0
(p9)  fms.s1    FR_XLog_Hi    = FR_Two, FR_AX, f0  // Hi  of log1p arg = 2*X - 1
(p9)  br.cond.spnt huges_logl                      // special version of log1p
};;

{ .mfb
      ldfe      FR_log2_hi    = [GR_ad_q],16      // Load log2_hi
(p8)  fma.s1    FR_X3         = FR_X2, FR_Arg, f0        // x^3 = x^2 * x
(p8)  br.cond.spnt near_0                                // Go to near 0 branch
};;

{ .mfi
      ldfe      FR_log2_lo    = [GR_ad_q],16      // Load log2_lo
      nop.f 0
      nop.i 0
};;

{ .mfi
      ldfe      FR_Q4         = [GR_ad_q],16          // Load Q4
      fma.s1    FR_Tmp        = FR_Tmp, f1, FR_X2       // Tmp = Tmp + x^2
      mov       GR_exp_mask   = 0x1FFFF        // Create exponent mask
};;

{ .mfi
      ldfe      FR_Q3         = [GR_ad_q],16   // Load Q3
      fma.s1    FR_GG         = FR_Rcp, FR_P2, f0        // g = Rcp * p2
                                               // 8 bit Newton Raphson iteration
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_HH         = FR_Half, FR_Rcp, f0      // h = 0.5 * Rcp
      nop.i 0
};;
{ .mfi
      ldfe      FR_Q2         = [GR_ad_q],16      // Load Q2
      fnma.s1   FR_EE         = FR_GG, FR_HH, FR_Half   // e = 0.5 - g * h
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_P2L        = FR_Tmp, f1, FR_P2L // low part of p2 = Tmp + p2l
      nop.i 0
};;

{ .mfi
      ldfe      FR_Q1         = [GR_ad_q]                // Load Q1
      fma.s1    FR_GG         = FR_GG, FR_EE, FR_GG     // g = g * e + g
                                              // 16 bit Newton Raphson iteration
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
      fnma.s1   FR_DD         = FR_GG, FR_GG, FR_P2  // Remainder d = g * g - p2
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_XLog_Hi     = FR_AX, f1, FR_GG // bh = z + gh
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_DD         = FR_DD, f1, FR_P2L       // add p2l: d = d + p2l
      nop.i 0
};;


{ .mfi
      getf.sig  GR_signif     = FR_XLog_Hi     // Get significand of x+1
      fmerge.ns FR_Neg_One    = f1, f1         // Form -1.0
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
      fms.s1    FR_XLog_Lo     = FR_GG,  f1,   FR_XLog_Hi // bl = gh - bh
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
      fma.s1    FR_XLog_Lo    = FR_XLog_Lo, f1, FR_AX // bl = bl + x
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
// So we can negate Q coefficients there for negative values

{ .mfi
      nop.m 0
(p11) fma.s1    FR_Q1         = FR_Q1, FR_Neg_One, f0 // Negate Q1
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_XLog_Lo     = FR_XLog_Lo, f1, FR_GL // bl = bl + gl
      nop.i 0
};;

{ .mfi
      nop.m 0
(p11) fma.s1    FR_Q2         = FR_Q2, FR_Neg_One, f0 // Negate Q2
      nop.i 0
};;

{ .mfi
      nop.m 0
(p11) fma.s1    FR_Q3         = FR_Q3, FR_Neg_One, f0 // Negate Q3
      nop.i 0
};;

{ .mfi
      nop.m 0
(p11) fma.s1    FR_Q4         = FR_Q4, FR_Neg_One, f0 // Negate Q4
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
      fma.s1    FR_S_lo       = FR_XLog_Lo, FR_2_to_minus_N, f0 //S_lo=S_lo*2^-N
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

.pred.rel "mutex",p12,p11
{ .mfi
      nop.m 0
(p12) fma.s1    FR_poly_hi    = FR_Q1, FR_rsq, FR_r // poly_hi = Q1*rsq + r
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fms.s1    FR_poly_hi    = FR_Q1, FR_rsq, FR_r     // poly_hi = Q1*rsq + r
      nop.i 0
};;


.pred.rel "mutex",p12,p11
{ .mfi
      nop.m 0
(p12) fma.s1    FR_poly_lo    = FR_poly_lo, FR_rcub, FR_h//poly_lo=poly_lo*r^3+h
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fms.s1    FR_poly_lo    = FR_poly_lo, FR_rcub, FR_h//poly_lo=poly_lo*r^3+h
      nop.i 0
}
;;

{ .mfi
      nop.m 0
      fadd.s0   FR_Y_lo       = FR_poly_hi, FR_poly_lo
	                                                     // Y_lo=poly_hi+poly_lo
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fma.s0    FR_Y_hi       = FR_Y_hi, FR_Neg_One, f0 // FR_Y_hi sign for neg
      nop.i 0
};;

{ .mfb
      nop.m 0
      fadd.s0   FR_Res        = FR_Y_lo,FR_Y_hi    // Result=Y_lo+Y_hi
      br.ret.sptk   b0                         // Common exit for 2^-7 < x < inf
};;

// * SPECIAL VERSION OF LOGL FOR HUGE ARGUMENTS *

huges_logl:
{ .mfi
      getf.sig  GR_signif     = FR_XLog_Hi     // Get significand of x+1
      fmerge.ns FR_Neg_One    = f1, f1         // Form -1.0
      mov       GR_exp_2tom7  = 0x0fff8        // Exponent of 2^-7
};;

{ .mfi
      add       GR_ad_tbl_1   = 0x040, GR_ad_z_1    // Point to Constants_G_H_h1
      nop.f 0
      add       GR_ad_q       = -0x60, GR_ad_z_1    // Point to Constants_P
}
{ .mfi
      add       GR_ad_z_2     = 0x140, GR_ad_z_1    // Point to Constants_Z_2
      nop.f 0
      add       GR_ad_tbl_2   = 0x180, GR_ad_z_1    // Point to Constants_G_H_h2
};;

{ .mfi
      nop.m 0
      nop.f 0
      extr.u    GR_Index1     = GR_signif, 59, 4    // Get high 4 bits of signif
}
{ .mfi
      add       GR_ad_tbl_3   = 0x280, GR_ad_z_1    // Point to Constants_G_H_h3
      nop.f 0
      nop.i 0
};;

{ .mfi
      shladd    GR_ad_z_1     = GR_Index1, 2, GR_ad_z_1  // Point to Z_1
      nop.f 0
      extr.u    GR_X_0        = GR_signif, 49, 15 // Get high 15 bits of signif.
};;

{ .mfi
      ld4       GR_Z_1        = [GR_ad_z_1]    // Load Z_1
      nop.f 0
      mov       GR_exp_mask   = 0x1FFFF        // Create exponent mask
}
{ .mfi
      shladd    GR_ad_tbl_1   = GR_Index1, 4, GR_ad_tbl_1  // Point to G_1
      nop.f 0
      mov       GR_Bias       = 0x0FFFF            // Create exponent bias
};;

{ .mfi
      ldfps     FR_G, FR_H    = [GR_ad_tbl_1],8     // Load G_1, H_1
      fmerge.se FR_S_hi       =  f1,FR_XLog_Hi            // Form |x+1|
      nop.i 0
};;

{ .mmi
      getf.exp  GR_N          =  FR_XLog_Hi          // Get N = exponent of x+1
      ldfd      FR_h          = [GR_ad_tbl_1]        // Load h_1
      nop.i 0
};;

{ .mfi
      ldfe      FR_log2_hi    = [GR_ad_q],16      // Load log2_hi
      nop.f 0
      pmpyshr2.u GR_X_1       = GR_X_0,GR_Z_1,15  // Get bits 30-15 of X_0 * Z_1
};;

// WE CANNOT USE GR_X_1 IN NEXT 3 CYCLES BECAUSE OF POSSIBLE 10 CLOCKS STALL!
// "DEAD" ZONE!

{ .mmi
      ldfe      FR_log2_lo    = [GR_ad_q],16      // Load log2_lo
      sub       GR_N          = GR_N, GR_Bias
      mov       GR_exp_2tom80 = 0x0ffaf           // Exponent of 2^-80
};;

{ .mfi
      ldfe      FR_Q4         = [GR_ad_q],16          // Load Q4
      nop.f 0
      sub       GR_minus_N    = GR_Bias, GR_N         // Form exponent of 2^(-N)
};;

{ .mmf
      ldfe      FR_Q3         = [GR_ad_q],16   // Load Q3
      setf.sig  FR_float_N    = GR_N        // Put integer N into rightmost sign
      nop.f 0
};;

{ .mmi
      nop.m 0
      ldfe      FR_Q2         = [GR_ad_q],16      // Load Q2
      extr.u    GR_Index2     = GR_X_1, 6, 4      // Extract bits 6-9 of X_1
};;

{ .mmi
      ldfe      FR_Q1         = [GR_ad_q]                // Load Q1
      shladd    GR_ad_z_2     = GR_Index2, 2, GR_ad_z_2  // Point to Z_2
      nop.i 0
};;

{ .mmi
      ld4       GR_Z_2        = [GR_ad_z_2]                // Load Z_2
      shladd    GR_ad_tbl_2   = GR_Index2, 4, GR_ad_tbl_2  // Point to G_2
      nop.i 0
};;

{ .mmi
      ldfps     FR_G2, FR_H2  = [GR_ad_tbl_2],8       // Load G_2, H_2
      nop.m 0
      nop.i 0
};;

{ .mfi
      ldfd      FR_h2         = [GR_ad_tbl_2]             // Load h_2
      nop.f 0
      nop.i 0
}
{ .mfi
      setf.exp  FR_2_to_minus_N = GR_minus_N   // Form 2^(-N)
      nop.f 0
      nop.i 0
};;

{ .mfi
      nop.m 0
      nop.f 0
      pmpyshr2.u GR_X_2       = GR_X_1,GR_Z_2,15 // Get bits 30-15 of X_1 * Z_2
};;

// WE CANNOT USE GR_X_2 IN NEXT 3 CYCLES BECAUSE OF POSSIBLE 10 CLOCKS STALL!
// "DEAD" ZONE!
// JUST HAVE TO INSERT 3 NOP CYCLES (nothing to do here)

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
(p11) fma.s1    FR_Q4         = FR_Q4, FR_Neg_One, f0 // Negate Q4
      extr.u    GR_Index3     = GR_X_2, 1, 5          // Extract bits 1-5 of X_2
 };;

{ .mfi
      shladd    GR_ad_tbl_3   = GR_Index3, 4, GR_ad_tbl_3  // Point to G_3
	  fcvt.xf   FR_float_N    = FR_float_N
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fma.s1    FR_Q3         = FR_Q3, FR_Neg_One, f0 // Negate Q3
      nop.i 0
};;

{ .mfi
      ldfps     FR_G3, FR_H3  = [GR_ad_tbl_3],8   // Load G_3, H_3
(p11) fma.s1    FR_Q2         = FR_Q2, FR_Neg_One, f0 // Negate Q2
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fma.s1    FR_Q1         = FR_Q1, FR_Neg_One, f0 // Negate Q1
      nop.i 0
};;

{ .mfi
      ldfd      FR_h3         = [GR_ad_tbl_3]            // Load h_3
      fmpy.s1   FR_G          = FR_G, FR_G2              // G = G_1 * G_2
      nop.i 0
}
{ .mfi
      nop.m 0
      fadd.s1   FR_H          = FR_H, FR_H2              // H = H_1 + H_2
      nop.i 0
};;

{ .mmf
      nop.m 0
      nop.m 0
      fadd.s1   FR_h          = FR_h, FR_h2              // h = h_1 + h_2
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

.pred.rel "mutex",p12,p11
{ .mfi
      nop.m 0
(p12) fma.s1    FR_poly_hi    = FR_Q1, FR_rsq, FR_r // poly_hi = Q1*rsq + r
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fms.s1    FR_poly_hi    = FR_Q1, FR_rsq, FR_r // poly_hi = Q1*rsq + r
      nop.i 0
};;


.pred.rel "mutex",p12,p11
{ .mfi
      nop.m 0
(p12) fma.s1    FR_poly_lo    = FR_poly_lo, FR_rcub, FR_h//poly_lo=poly_lo*r^3+h
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fms.s1    FR_poly_lo    = FR_poly_lo, FR_rcub, FR_h//poly_lo=poly_lo*r^3+h
      nop.i 0
};;

{ .mfi
      nop.m 0
      fadd.s0   FR_Y_lo       = FR_poly_hi, FR_poly_lo  // Y_lo=poly_hi+poly_lo
      nop.i 0
}
{ .mfi
      nop.m 0
(p11) fma.s0    FR_Y_hi       = FR_Y_hi, FR_Neg_One, f0 // FR_Y_hi sign for neg
      nop.i 0
};;

{ .mfb
      nop.m 0
      fadd.s0   FR_Res        = FR_Y_lo,FR_Y_hi    // Result=Y_lo+Y_hi
      br.ret.sptk   b0                         // Common exit for 2^-7 < x < inf
};;

// NEAR ZERO POLYNOMIAL INTERVAL
near_0:
{ .mfi
      nop.m 0
      fma.s1    FR_X4         = FR_X2, FR_X2, f0 // x^4 = x^2 * x^2
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_P9         = FR_C9,FR_X2,FR_C7  // p9 = C9*x^2 + C7
      nop.i 0
}
{ .mfi
      nop.m 0
      fma.s1    FR_P5         = FR_C5,FR_X2,FR_C3  // p5 = C5*x^2 + C3
      nop.i 0
};;

{ .mfi
      nop.m 0
      fma.s1    FR_P3         = FR_P9,FR_X4,FR_P5  // p3 = p9*x^4 + p5
      nop.i 0
};;

{ .mfb
      nop.m 0
      fma.s0    FR_Res        = FR_P3,FR_X3,FR_Arg // res = p3*C3 + x
      br.ret.sptk   b0                          // Near 0 path return
};;

GLOBAL_LIBM_END(asinhl)
libm_alias_ldouble_other (asinh, asinh)
