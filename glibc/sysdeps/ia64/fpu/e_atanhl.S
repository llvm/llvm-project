.file "atanhl.s"


// Copyright (c) 2001 - 2003, Intel Corporation
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
// 09/10/01  Initial version
// 12/11/01  Corrected .restore syntax
// 05/20/02  Cleaned up namespace and sf0 syntax
// 02/10/03  Reordered header: .section, .global, .proc, .align;
//           used data8 for long double table values
//
//*********************************************************************
//
//*********************************************************************
//
// Function: atanhl(x) computes the principle value of the inverse
// hyperbolic tangent of x.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 (Input and Return Value)
//                              f33-f73
//
//    General Purpose Registers:
//      r32-r52
//      r49-r52 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    atanhl(inf) = QNaN
//    atanhl(-inf) = QNaN
//    atanhl(+/-0) = +/-0
//    atanhl(1) =  +inf
//    atanhl(-1) =  -inf
//    atanhl(|x|>1) = QNaN
//    atanhl(SNaN) = QNaN
//    atanhl(QNaN) = QNaN
//
//*********************************************************************
//
// Overview
//
// The method consists of two cases.
//
// If      |x| < 1/32  use case atanhl_near_zero;
// else                 use case atanhl_regular;
//
// Case atanhl_near_zero:
//
//   atanhl(x) can be approximated by the Taylor series expansion
//   up to order 17.
//
// Case atanhl_regular:
//
//   Here we use formula atanhl(x) = sign(x)*log1pl(2*|x|/(1-|x|))/2 and
//   calculation is subdivided into two stages. The first stage is
//   calculating of X = 2*|x|/(1-|x|). The second one is calculating of
//   sign(x)*log1pl(X)/2. To obtain required accuracy we use precise division
//   algorithm output of which is a pair of two extended precision values those
//   approximate result of division with accuracy higher than working
//   precision. This pair is passed to modified log1pl function.
//
//
//   1. calculating of X = 2*|x|/(1-|x|)
//   ( based on Peter Markstein's "IA-64 and Elementary Functions" book )
//   ********************************************************************
//
//     a = 2*|x|
//     b = 1 - |x|
//     b_lo = |x| - (1 - b)
//
//     y = frcpa(b)         initial approximation of 1/b
//     q = a*y              initial approximation of a/b
//
//     e = 1 - b*y
//     e2 = e + e^2
//     e1 = e^2
//     y1 = y + y*e2 = y + y*(e+e^2)
//
//     e3 = e + e1^2
//     y2 = y + y1*e3 = y + y*(e+e^2+..+e^6)
//
//     r = a - b*q
//     e = 1 - b*y2
//     X = q + r*y2         high part of a/b
//
//     y3 = y2 + y2*e4
//     r1 = a - b*X
//     r1 = r1 - b_lo*X
//     X_lo = r1*y3         low part of a/b
//
//   2. special log1p algorithm overview
//   ***********************************
//
//    Here we use a table lookup method. The basic idea is that in
//    order to compute logl(Arg) = log1pl (Arg-1) for an argument Arg in [1,2),
//    we construct a value G such that G*Arg is close to 1 and that
//    logl(1/G) is obtainable easily from a table of values calculated
//    beforehand. Thus
//
//      logl(Arg) = logl(1/G) + logl(G*Arg)
//           = logl(1/G) + logl(1 + (G*Arg - 1))
//
//    Because |G*Arg - 1| is small, the second term on the right hand
//    side can be approximated by a short polynomial. We elaborate
//    this method in several steps.
//
//    Step 0: Initialization
//    ------
//    We need to calculate logl(X + X_lo + 1). Obtain N, S_hi such that
//
//      X + X_lo + 1 = 2^N * ( S_hi + S_lo )   exactly
//
//    where S_hi in [1,2) and S_lo is a correction to S_hi in the sense
//    that |S_lo| <= ulp(S_hi).
//
//    For the special version of log1p we add X_lo to S_lo (S_lo = S_lo + X_lo)
//    !-----------------------------------------------------------------------!
//
//    Step 1: Argument Reduction
//    ------
//    Based on S_hi, obtain G_1, G_2, G_3 from a table and calculate
//
//      G := G_1 * G_2 * G_3
//      r := (G * S_hi - 1) + G * S_lo
//
//    These G_j's have the property that the product is exactly
//    representable and that |r| < 2^(-12) as a result.
//
//    Step 2: Approximation
//    ------
//    logl(1 + r) is approximated by a short polynomial poly(r).
//
//    Step 3: Reconstruction
//    ------
//    Finally, log1pl(X + X_lo) = logl(X + X_lo + 1) is given by
//
//    logl(X + X_lo + 1) =  logl(2^N * (S_hi + S_lo))
//                      ~=~ N*logl(2) + logl(1/G) + logl(1 + r)
//                      ~=~ N*logl(2) + logl(1/G) + poly(r).
//
//    For detailed description see log1p1 function, regular path.
//
//*********************************************************************

RODATA
.align 64

// ************* DO NOT CHANGE THE ORDER OF THESE TABLES *************

LOCAL_OBJECT_START(Constants_TaylorSeries)
data8  0xF0F0F0F0F0F0F0F1,0x00003FFA // C17
data8  0x8888888888888889,0x00003FFB // C15
data8  0x9D89D89D89D89D8A,0x00003FFB // C13
data8  0xBA2E8BA2E8BA2E8C,0x00003FFB // C11
data8  0xE38E38E38E38E38E,0x00003FFB // C9
data8  0x9249249249249249,0x00003FFC // C7
data8  0xCCCCCCCCCCCCCCCD,0x00003FFC // C5
data8  0xAAAAAAAAAAAAAAAA,0x00003FFD // C3
data4  0x3f000000                    // 1/2
data4  0x00000000                    // pad
data4  0x00000000
data4  0x00000000
LOCAL_OBJECT_END(Constants_TaylorSeries)

LOCAL_OBJECT_START(Constants_Q)
data4  0x00000000,0xB1721800,0x00003FFE,0x00000000 // log2_hi
data4  0x4361C4C6,0x82E30865,0x0000BFE2,0x00000000 // log2_lo
data4  0x328833CB,0xCCCCCAF2,0x00003FFC,0x00000000 // Q4
data4  0xA9D4BAFB,0x80000077,0x0000BFFD,0x00000000 // Q3
data4  0xAAABE3D2,0xAAAAAAAA,0x00003FFD,0x00000000 // Q2
data4  0xFFFFDAB7,0xFFFFFFFF,0x0000BFFD,0x00000000 // Q1
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



// Floating Point Registers

FR_C17              = f50
FR_C15              = f51
FR_C13              = f52
FR_C11              = f53
FR_C9               = f54
FR_C7               = f55
FR_C5               = f56
FR_C3               = f57
FR_x2               = f58
FR_x3               = f59
FR_x4               = f60
FR_x8               = f61

FR_Rcp              = f61

FR_A                = f33
FR_R1               = f33

FR_E1               = f34
FR_E3               = f34
FR_Y2               = f34
FR_Y3               = f34

FR_E2               = f35
FR_Y1               = f35

FR_B                = f36
FR_Y0               = f37
FR_E0               = f38
FR_E4               = f39
FR_Q0               = f40
FR_R0               = f41
FR_B_lo             = f42

FR_abs_x            = f43
FR_Bp               = f44
FR_Bn               = f45
FR_Yp               = f46
FR_Yn               = f47

FR_X                = f48
FR_BB               = f48
FR_X_lo             = f49

FR_G                = f50
FR_Y_hi             = f51
FR_H                = f51
FR_h                = f52
FR_G2               = f53
FR_H2               = f54
FR_h2               = f55
FR_G3               = f56
FR_H3               = f57
FR_h3               = f58

FR_Q4               = f59
FR_poly_lo          = f59
FR_Y_lo             = f59

FR_Q3               = f60
FR_Q2               = f61

FR_Q1               = f62
FR_poly_hi          = f62

FR_float_N          = f63

FR_AA               = f64
FR_S_lo             = f64

FR_S_hi             = f65
FR_r                = f65

FR_log2_hi          = f66
FR_log2_lo          = f67
FR_Z                = f68
FR_2_to_minus_N     = f69
FR_rcub             = f70
FR_rsq              = f71
FR_05r              = f72
FR_Half             = f73

FR_Arg_X            = f50
FR_Arg_Y            = f0
FR_RESULT           = f8



// General Purpose Registers

GR_ad_05            = r33
GR_Index1           = r34
GR_ArgExp           = r34
GR_Index2           = r35
GR_ExpMask          = r35
GR_NearZeroBound    = r36
GR_signif           = r36
GR_X_0              = r37
GR_X_1              = r37
GR_X_2              = r38
GR_Index3           = r38
GR_minus_N          = r39
GR_Z_1              = r40
GR_Z_2              = r40
GR_N                = r41
GR_Bias             = r42
GR_M                = r43
GR_ad_taylor        = r44
GR_ad_taylor_2      = r45
GR_ad2_tbl_3        = r45
GR_ad_tbl_1         = r46
GR_ad_tbl_2         = r47
GR_ad_tbl_3         = r48
GR_ad_q             = r49
GR_ad_z_1           = r50
GR_ad_z_2           = r51
GR_ad_z_3           = r52

//
// Added for unwind support
//
GR_SAVE_PFS         = r46
GR_SAVE_B0          = r47
GR_SAVE_GP          = r48
GR_Parameter_X      = r49
GR_Parameter_Y      = r50
GR_Parameter_RESULT = r51
GR_Parameter_TAG    = r52



.section .text
GLOBAL_LIBM_ENTRY(atanhl)

{ .mfi
      alloc         r32 = ar.pfs,0,17,4,0
      fnma.s1       FR_Bp = f8,f1,f1 // b = 1 - |arg| (for x>0)
      mov           GR_ExpMask = 0x1ffff
}
{ .mfi
      addl          GR_ad_taylor = @ltoff(Constants_TaylorSeries),gp
      fma.s1        FR_Bn = f8,f1,f1 // b = 1 - |arg| (for x<0)
      mov           GR_NearZeroBound = 0xfffa  // biased exp of 1/32
};;
{ .mfi
      getf.exp      GR_ArgExp = f8
      fcmp.lt.s1    p6,p7 = f8,f0 // is negative?
      nop.i         0
}
{ .mfi
      ld8           GR_ad_taylor = [GR_ad_taylor]
      fmerge.s      FR_abs_x =  f1,f8
      nop.i         0
};;
{ .mfi
      nop.m         0
      fclass.m      p8,p0 = f8,0x1C7 // is arg NaT,Q/SNaN or +/-0 ?
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_x2 = f8,f8,f0
      nop.i         0
};;
{ .mfi
      add           GR_ad_z_1 = 0x0F0,GR_ad_taylor
      fclass.m      p9,p0 = f8,0x0a // is arg -denormal ?
      add           GR_ad_taylor_2 = 0x010,GR_ad_taylor
}
{ .mfi
      add           GR_ad_05 = 0x080,GR_ad_taylor
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfe          FR_C17 = [GR_ad_taylor],32
      fclass.m      p10,p0 = f8,0x09 // is arg +denormal ?
      add           GR_ad_tbl_1 = 0x040,GR_ad_z_1 // point to Constants_G_H_h1
}
{ .mfb
      add           GR_ad_z_2 = 0x140,GR_ad_z_1 // point to Constants_Z_2
 (p8) fma.s0        f8 =  f8,f1,f0 // NaN or +/-0
 (p8) br.ret.spnt   b0             // exit for Nan or +/-0
};;
{ .mfi
      ldfe          FR_C15 = [GR_ad_taylor_2],32
      fclass.m      p15,p0 = f8,0x23 // is +/-INF ?
      add           GR_ad_tbl_2 = 0x180,GR_ad_z_1 // point to Constants_G_H_h2
}
{ .mfb
      ldfe          FR_C13 = [GR_ad_taylor],32
 (p9) fnma.s0       f8 =  f8,f8,f8 // -denormal
 (p9) br.ret.spnt   b0             // exit for -denormal
};;
{ .mfi
      ldfe          FR_C11 = [GR_ad_taylor_2],32
      fcmp.eq.s0       p13,p0 = FR_abs_x,f1 // is |arg| = 1?
      nop.i         0
}
{ .mfb
      ldfe          FR_C9 = [GR_ad_taylor],32
(p10) fma.s0        f8 =  f8,f8,f8 // +denormal
(p10) br.ret.spnt   b0             // exit for +denormal
};;
{ .mfi
      ldfe          FR_C7 = [GR_ad_taylor_2],32
 (p6) frcpa.s1      FR_Yn,p11 = f1,FR_Bn // y = frcpa(b)
      and           GR_ArgExp = GR_ArgExp,GR_ExpMask // biased exponent
}
{ .mfb
      ldfe          FR_C5 = [GR_ad_taylor],32
      fnma.s1       FR_B = FR_abs_x,f1,f1 // b = 1 - |arg|
(p15) br.cond.spnt  atanhl_gt_one // |arg| > 1
};;
{ .mfb
      cmp.gt        p14,p0 = GR_NearZeroBound,GR_ArgExp
 (p7) frcpa.s1      FR_Yp,p12 = f1,FR_Bp // y = frcpa(b)
(p13) br.cond.spnt  atanhl_eq_one // |arg| = 1/32
}
{ .mfb
      ldfe          FR_C3 = [GR_ad_taylor_2],32
      fma.s1        FR_A = FR_abs_x,f1,FR_abs_x // a = 2 * |arg|
(p14) br.cond.spnt  atanhl_near_zero // |arg| < 1/32
};;
{ .mfi
      nop.m         0
      fcmp.gt.s0       p8,p0 = FR_abs_x,f1 // is |arg| > 1 ?
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
 (p6) fnma.s1       FR_B_lo = FR_Bn,f1,f1 // argt = 1 - (1 - |arg|)
      nop.i         0
}
{ .mfi
      ldfs          FR_Half = [GR_ad_05]
 (p7) fnma.s1       FR_B_lo = FR_Bp,f1,f1
      nop.i         0
};;
{ .mfi
      nop.m         0
 (p6) fnma.s1       FR_E0 = FR_Yn,FR_Bn,f1 // e = 1-b*y
      nop.i         0
}
{ .mfb
      nop.m         0
 (p6) fma.s1        FR_Y0 = FR_Yn,f1,f0
 (p8) br.cond.spnt  atanhl_gt_one // |arg| > 1
};;
{ .mfi
      nop.m         0
 (p7) fnma.s1       FR_E0 = FR_Yp,FR_Bp,f1
      nop.i         0
}
{ .mfi
      nop.m         0
 (p6) fma.s1        FR_Q0 = FR_A,FR_Yn,f0 // q = a*y
      nop.i         0
};;
{ .mfi
      nop.m         0
 (p7) fma.s1        FR_Q0 = FR_A,FR_Yp,f0
      nop.i         0
}
{ .mfi
      nop.m         0
 (p7) fma.s1        FR_Y0 = FR_Yp,f1,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fclass.nm     p10,p0 = f8,0x1FF  // test for unsupported
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_E2 = FR_E0,FR_E0,FR_E0 // e2 = e+e^2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_E1 = FR_E0,FR_E0,f0 // e1 = e^2
      nop.i         0
};;
{ .mfb
      nop.m         0
//    Return generated NaN or other value for unsupported values.
(p10) fma.s0        f8 = f8, f0, f0
(p10) br.ret.spnt   b0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Y1 = FR_Y0,FR_E2,FR_Y0 // y1 = y+y*e2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_E3 = FR_E1,FR_E1,FR_E0 // e3 = e+e1^2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_B_lo = FR_abs_x,f1,FR_B_lo // b_lo = argt-|arg|
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Y2 = FR_Y1,FR_E3,FR_Y0 // y2 = y+y1*e3
      nop.i         0
}
{ .mfi
      nop.m         0
      fnma.s1       FR_R0 = FR_B,FR_Q0,FR_A // r = a-b*q
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_E4 = FR_B,FR_Y2,f1 // e4 = 1-b*y2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_X = FR_R0,FR_Y2,FR_Q0 // x = q+r*y2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Z = FR_X,f1,f1 // x+1
      nop.i         0
};;
{ .mfi
      nop.m         0
 (p6) fnma.s1       FR_Half = FR_Half,f1,f0 // sign(arg)/2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Y3 = FR_Y2,FR_E4,FR_Y2 // y3 = y2+y2*e4
      nop.i         0
}
{ .mfi
      nop.m         0
      fnma.s1       FR_R1 = FR_B,FR_X,FR_A // r1 = a-b*x
      nop.i         0
};;
{ .mfi
      getf.sig      GR_signif = FR_Z // get significand of x+1
      nop.f         0
      nop.i         0
};;


{ .mfi
      add           GR_ad_q = -0x060,GR_ad_z_1
      nop.f         0
      extr.u        GR_Index1 = GR_signif,59,4 // get high 4 bits of signif
}
{ .mfi
      add           GR_ad_tbl_3 = 0x280,GR_ad_z_1 // point to Constants_G_H_h3
      nop.f         0
      nop.i         0
};;
{ .mfi
      shladd        GR_ad_z_1 = GR_Index1,2,GR_ad_z_1 // point to Z_1
      nop.f         0
      extr.u        GR_X_0 = GR_signif,49,15 // get high 15 bits of significand
};;
{ .mfi
      ld4           GR_Z_1 = [GR_ad_z_1] // load Z_1
      fmax.s1       FR_AA = FR_X,f1 // for S_lo,form AA = max(X,1.0)
      nop.i         0
}
{ .mfi
      shladd        GR_ad_tbl_1 = GR_Index1,4,GR_ad_tbl_1 // point to G_1
      nop.f         0
      mov           GR_Bias = 0x0FFFF // exponent bias
};;
{ .mfi
      ldfps         FR_G,FR_H = [GR_ad_tbl_1],8  // load G_1,H_1
      fmerge.se     FR_S_hi =  f1,FR_Z // form |x+1|
      nop.i         0
};;
{ .mfi
      getf.exp      GR_N =  FR_Z // get N = exponent of x+1
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfd          FR_h = [GR_ad_tbl_1] // load h_1
      fnma.s1       FR_R1 = FR_B_lo,FR_X,FR_R1 // r1 = r1-b_lo*x
      nop.i         0
};;
{ .mfi
      ldfe          FR_log2_hi = [GR_ad_q],16 // load log2_hi
      nop.f         0
      pmpyshr2.u    GR_X_1 = GR_X_0,GR_Z_1,15 // get bits 30-15 of X_0 * Z_1
};;
//
//    For performance,don't use result of pmpyshr2.u for 4 cycles.
//
{ .mfi
      ldfe          FR_log2_lo = [GR_ad_q],16 // load log2_lo
      nop.f         0
      sub           GR_N = GR_N,GR_Bias
};;
{ .mfi
      ldfe          FR_Q4 = [GR_ad_q],16  // load Q4
      fms.s1        FR_S_lo = FR_AA,f1,FR_Z // form S_lo = AA - Z
      sub           GR_minus_N = GR_Bias,GR_N // form exponent of 2^(-N)
};;
{ .mmf
      ldfe          FR_Q3 = [GR_ad_q],16 // load Q3
      // put integer N into rightmost significand
      setf.sig      FR_float_N = GR_N
      fmin.s1       FR_BB = FR_X,f1 // for S_lo,form BB = min(X,1.0)
};;
{ .mfi
      ldfe          FR_Q2 = [GR_ad_q],16 // load Q2
      nop.f         0
      extr.u        GR_Index2 = GR_X_1,6,4 // extract bits 6-9 of X_1
};;
{ .mmi
      ldfe          FR_Q1 = [GR_ad_q] // load Q1
      shladd        GR_ad_z_2 = GR_Index2,2,GR_ad_z_2 // point to Z_2
      nop.i         0
};;
{ .mmi
      ld4           GR_Z_2 = [GR_ad_z_2] // load Z_2
      shladd        GR_ad_tbl_2 = GR_Index2,4,GR_ad_tbl_2 // point to G_2
      nop.i         0
};;
{ .mfi
      ldfps         FR_G2,FR_H2 = [GR_ad_tbl_2],8 // load G_2,H_2
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfd          FR_h2 = [GR_ad_tbl_2] // load h_2
      fma.s1        FR_S_lo = FR_S_lo,f1,FR_BB // S_lo = S_lo + BB
      nop.i         0
}
{ .mfi
      setf.exp      FR_2_to_minus_N = GR_minus_N // form 2^(-N)
      fma.s1        FR_X_lo = FR_R1,FR_Y3,f0 // x_lo = r1*y3
      nop.i         0
};;
{ .mfi
      nop.m         0
      nop.f         0
      pmpyshr2.u    GR_X_2 = GR_X_1,GR_Z_2,15 // get bits 30-15 of X_1 * Z_2
};;
//
//    For performance,don't use result of pmpyshr2.u for 4 cycles
//
{ .mfi
      add           GR_ad2_tbl_3 = 8,GR_ad_tbl_3
      nop.f         0
      nop.i         0
}
{ .mfi
      nop.m         0
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
      nop.f         0
      nop.i         0
};;

//
//    Now GR_X_2 can be used
//
{ .mfi
      nop.m         0
      nop.f         0
      extr.u        GR_Index3 = GR_X_2,1,5 // extract bits 1-5 of X_2
}
{ .mfi
      nop.m         0
      fma.s1        FR_S_lo = FR_S_lo,f1,FR_X_lo // S_lo = S_lo + Arg_lo
      nop.i         0
};;

{ .mfi
      shladd        GR_ad_tbl_3 = GR_Index3,4,GR_ad_tbl_3 // point to G_3
      fcvt.xf       FR_float_N = FR_float_N
      nop.i         0
}
{ .mfi
      shladd        GR_ad2_tbl_3 = GR_Index3,4,GR_ad2_tbl_3 // point to h_3
      fma.s1        FR_Q1 = FR_Q1,FR_Half,f0 // sign(arg)*Q1/2
      nop.i         0
};;
{ .mmi
      ldfps         FR_G3,FR_H3 = [GR_ad_tbl_3],8 // load G_3,H_3
      ldfd          FR_h3 = [GR_ad2_tbl_3] // load h_3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fmpy.s1       FR_G = FR_G,FR_G2 // G = G_1 * G_2
      nop.i         0
}
{ .mfi
      nop.m         0
      fadd.s1       FR_H = FR_H,FR_H2 // H = H_1 + H_2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fadd.s1       FR_h = FR_h,FR_h2 // h = h_1 + h_2
      nop.i         0
};;
{ .mfi
      nop.m         0
      // S_lo = S_lo * 2^(-N)
      fma.s1        FR_S_lo = FR_S_lo,FR_2_to_minus_N,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fmpy.s1       FR_G = FR_G,FR_G3 // G = (G_1 * G_2) * G_3
      nop.i         0
}
{ .mfi
      nop.m         0
      fadd.s1       FR_H = FR_H,FR_H3 // H = (H_1 + H_2) + H_3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fadd.s1       FR_h = FR_h,FR_h3 // h = (h_1 + h_2) + h_3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_r = FR_G,FR_S_hi,f1 // r = G * S_hi - 1
      nop.i         0
}
{ .mfi
      nop.m         0
      // Y_hi = N * log2_hi + H
      fma.s1        FR_Y_hi = FR_float_N,FR_log2_hi,FR_H
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_h = FR_float_N,FR_log2_lo,FR_h // h = N * log2_lo + h
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_r = FR_G,FR_S_lo,FR_r // r = G * S_lo + (G * S_hi - 1)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_poly_lo = FR_r,FR_Q4,FR_Q3 // poly_lo = r * Q4 + Q3
      nop.i         0
}
{ .mfi
      nop.m         0
      fmpy.s1       FR_rsq = FR_r,FR_r // rsq = r * r
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_05r = FR_r,FR_Half,f0 // sign(arg)*r/2
      nop.i         0
};;
{ .mfi
      nop.m         0
      // poly_lo = poly_lo * r + Q2
      fma.s1        FR_poly_lo = FR_poly_lo,FR_r,FR_Q2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_rcub = FR_rsq,FR_r,f0 // rcub = r^3
      nop.i         0
};;
{ .mfi
      nop.m         0
      // poly_hi = sing(arg)*(Q1*r^2 + r)/2
      fma.s1        FR_poly_hi = FR_Q1,FR_rsq,FR_05r
      nop.i         0
};;
{ .mfi
      nop.m         0
      // poly_lo = poly_lo*r^3 + h
      fma.s1        FR_poly_lo = FR_poly_lo,FR_rcub,FR_h
      nop.i         0
};;
{ .mfi
      nop.m         0
      // Y_lo = poly_hi + poly_lo/2
      fma.s0        FR_Y_lo = FR_poly_lo,FR_Half,FR_poly_hi
      nop.i         0
};;
{ .mfb
      nop.m         0
     // Result = arctanh(x) = Y_hi/2 + Y_lo
      fma.s0        f8 = FR_Y_hi,FR_Half,FR_Y_lo
      br.ret.sptk   b0
};;

// Taylor's series
atanhl_near_zero:
{ .mfi
      nop.m         0
      fma.s1        FR_x3 = FR_x2,f8,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C17 = FR_C17,FR_x2,FR_C15
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C13 = FR_C13,FR_x2,FR_C11
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C9 = FR_C9,FR_x2,FR_C7
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C5 = FR_C5,FR_x2,FR_C3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_x8 = FR_x4,FR_x4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C17 = FR_C17,FR_x4,FR_C13
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C9 = FR_C9,FR_x4,FR_C5
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C17 = FR_C17,FR_x8,FR_C9
      nop.i         0
};;
{ .mfb
      nop.m         0
      fma.s0        f8 = FR_C17,FR_x3,f8
      br.ret.sptk   b0
};;

atanhl_eq_one:
{ .mfi
      nop.m         0
      frcpa.s0      FR_Rcp,p0 = f1,f0 // get inf,and raise Z flag
      nop.i         0
}
{ .mfi
      nop.m         0
      fmerge.s      FR_Arg_X = f8, f8
      nop.i         0
};;
{ .mfb
      mov           GR_Parameter_TAG = 130
      fmerge.s      FR_RESULT = f8,FR_Rcp // result is +-inf
      br.cond.sptk  __libm_error_region // exit if |x| = 1.0
};;

atanhl_gt_one:
{ .mfi
      nop.m         0
      fmerge.s      FR_Arg_X = f8, f8
      nop.i         0
};;
{ .mfb
      mov           GR_Parameter_TAG = 129
      frcpa.s0      FR_RESULT,p0 = f0,f0 // get QNaN,and raise invalid
      br.cond.sptk  __libm_error_region // exit if |x| > 1.0
};;

GLOBAL_LIBM_END(atanhl)
libm_alias_ldouble_other (atanh, atanh)

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
        stfe [GR_Parameter_Y] = FR_Arg_Y,16     // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0,GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfe [GR_Parameter_X] = FR_Arg_X        // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfe [GR_Parameter_Y] = FR_RESULT       // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
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
