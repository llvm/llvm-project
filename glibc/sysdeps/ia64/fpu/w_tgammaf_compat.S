.file "tgammaf.s"


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
// 11/30/01  Initial version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 02/10/03  Reordered header: .section, .global, .proc, .align
// 04/04/03  Changed error codes for overflow and negative integers
// 04/10/03  Changed code for overflow near zero handling
// 12/16/03  Fixed parameter passing to/from error handling routine
// 03/31/05  Reformatted delimiters between data tables
//
//*********************************************************************
//
//*********************************************************************
//
// Function: tgammaf(x) computes the principle value of the GAMMA
// function of x.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8-f15
//                              f33-f75
//
//    General Purpose Registers:
//      r8-r11
//      r14-r29
//      r32-r36
//      r37-r40 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    tgammaf(+inf) = +inf
//    tgammaf(-inf) = QNaN
//    tgammaf(+/-0) = +/-inf
//    tgammaf(x<0, x - integer) = QNaN
//    tgammaf(SNaN) = QNaN
//    tgammaf(QNaN) = QNaN
//
//*********************************************************************
//
// Overview
//
// The method consists of three cases.
//
// If       2 <= x < OVERFLOW_BOUNDARY   use case tgamma_regular;
// else if  0 < x < 2                    use case tgamma_from_0_to_2;
// else if  -(i+1) <  x < -i, i = 0...43 use case tgamma_negatives;
//
// Case 2 <= x < OVERFLOW_BOUNDARY
// -------------------------------
//   Here we use algorithm based on the recursive formula
//   GAMMA(x+1) = x*GAMMA(x). For that we subdivide interval
//   [2; OVERFLOW_BOUNDARY] into intervals [8*n; 8*(n+1)] and
//   approximate GAMMA(x) by polynomial of 22th degree on each
//   [8*n; 8*n+1], recursive formula is used to expand GAMMA(x)
//   to [8*n; 8*n+1]. In other words we need to find n, i and r
//   such that x = 8 * n + i + r where n and i are integer numbers
//   and r is fractional part of x. So GAMMA(x) = GAMMA(8*n+i+r) =
//   = (x-1)*(x-2)*...*(x-i)*GAMMA(x-i) =
//   = (x-1)*(x-2)*...*(x-i)*GAMMA(8*n+r) ~
//   ~ (x-1)*(x-2)*...*(x-i)*P12n(r).
//
//   Step 1: Reduction
//   -----------------
//    N = [x] with truncate
//    r = x - N, note 0 <= r < 1
//
//    n = N & ~0xF - index of table that contains coefficient of
//                   polynomial approximation
//    i = N & 0xF  - is used in recursive formula
//
//
//   Step 2: Approximation
//   ---------------------
//    We use factorized minimax approximation polynomials
//    P12n(r) = A12*(r^2+C01(n)*r+C00(n))*
//              *(r^2+C11(n)*r+C10(n))*...*(r^2+C51(n)*r+C50(n))
//
//   Step 3: Recursion
//   -----------------
//    In case when i > 0 we need to multiply P12n(r) by product
//    R(i,x)=(x-1)*(x-2)*...*(x-i). To reduce number of fp-instructions
//    we can calculate R as follow:
//    R(i,x) = ((x-1)*(x-2))*((x-3)*(x-4))*...*((x-(i-1))*(x-i)) if i is
//    even or R = ((x-1)*(x-2))*((x-3)*(x-4))*...*((x-(i-2))*(x-(i-1)))*
//    *(i-1) if i is odd. In both cases we need to calculate
//    R2(i,x) = (x^2-3*x+2)*(x^2-7*x+12)*...*(x^2+x+2*j*(2*j-1)) =
//    = ((x^2-x)+2*(1-x))*((x^2-x)+6*(2-x))*...*((x^2-x)+2*(2*j-1)*(j-x)) =
//    = (RA+2*RB)*(RA+6*(1-RB))*...*(RA+2*(2*j-1)*(j-1+RB))
//    where j = 1..[i/2], RA = x^2-x, RB = 1-x.
//
//   Step 4: Reconstruction
//   ----------------------
//    Reconstruction is just simple multiplication i.e.
//    GAMMA(x) = P12n(r)*R(i,x)
//
// Case 0 < x < 2
// --------------
//    To calculate GAMMA(x) on this interval we do following
//        if 1.0  <= x < 1.25  than  GAMMA(x) = P7(x-1)
//        if 1.25 <= x < 1.5   than  GAMMA(x) = P7(x-x_min) where
//              x_min is point of local minimum on [1; 2] interval.
//        if 1.5  <= x < 1.75  than  GAMMA(x) = P7(x-1.5)
//        if 1.75 <= x < 2.0   than  GAMMA(x) = P7(x-1.5)
//    and
//        if 0 < x < 1 than GAMMA(x) = GAMMA(x+1)/x
//
// Case -(i+1) <  x < -i, i = 0...43
// ----------------------------------
//    Here we use the fact that GAMMA(-x) = PI/(x*GAMMA(x)*sin(PI*x)) and
//    so we need to calculate GAMMA(x), sin(PI*x)/PI. Calculation of
//    GAMMA(x) is described above.
//
//   Step 1: Reduction
//   -----------------
//    Note that period of sin(PI*x) is 2 and range reduction for
//    sin(PI*x) is like to range reduction for GAMMA(x)
//    i.e rs = x - round(x) and |rs| <= 0.5.
//
//   Step 2: Approximation
//   ---------------------
//    To approximate sin(PI*x)/PI = sin(PI*(2*n+rs))/PI =
//    = (-1)^n*sin(PI*rs)/PI Taylor series is used.
//    sin(PI*rs)/PI ~ S17(rs).
//
//   Step 3: Division
//   ----------------
//    To calculate 1/x and 1/(GAMMA(x)*S12(rs)) we use frcpa
//    instruction with following Newton-Raphson interations.
//
//
//*********************************************************************

GR_ad_Data              = r8
GR_TAG                  = r8
GR_SignExp              = r9
GR_Sig                  = r10
GR_ArgNz                = r10
GR_RqDeg                = r11

GR_NanBound             = r14
GR_ExpOf025             = r15
GR_ExpOf05              = r16
GR_ad_Co                = r17
GR_ad_Ce                = r18
GR_TblOffs              = r19
GR_Arg                  = r20
GR_Exp2Ind              = r21
GR_TblOffsMask          = r21
GR_Offs                 = r22
GR_OvfNzBound           = r23
GR_ZeroResBound         = r24
GR_ad_SinO              = r25
GR_ad_SinE              = r26
GR_Correction           = r27
GR_Tbl12Offs            = r28
GR_NzBound              = r28
GR_ExpOf1               = r29
GR_fpsr                 = r29

GR_SAVE_B0              = r33
GR_SAVE_PFS             = r34
GR_SAVE_GP              = r35
GR_SAVE_SP              = r36

GR_Parameter_X          = r37
GR_Parameter_Y          = r38
GR_Parameter_RESULT     = r39
GR_Parameter_TAG        = r40


FR_X                    = f10
FR_Y                    = f1
FR_RESULT               = f8

FR_iXt                  = f11
FR_Xt                   = f12
FR_r                    = f13
FR_r2                   = f14
FR_r4                   = f15

FR_C01                  = f33
FR_A7                   = f33
FR_C11                  = f34
FR_A6                   = f34
FR_C21                  = f35
FR_A5                   = f35
FR_C31                  = f36
FR_A4                   = f36
FR_C41                  = f37
FR_A3                   = f37
FR_C51                  = f38
FR_A2                   = f38

FR_C00                  = f39
FR_A1                   = f39
FR_C10                  = f40
FR_A0                   = f40
FR_C20                  = f41
FR_C30                  = f42
FR_C40                  = f43
FR_C50                  = f44
FR_An                   = f45
FR_OvfBound             = f46
FR_InvAn                = f47

FR_Multplr              = f48
FR_NormX                = f49
FR_X2mX                 = f50
FR_1mX                  = f51
FR_Rq0                  = f51
FR_Rq1                  = f52
FR_Rq2                  = f53
FR_Rq3                  = f54

FR_Rcp0                 = f55
FR_Rcp1                 = f56
FR_Rcp2                 = f57

FR_InvNormX1            = f58
FR_InvNormX2            = f59

FR_rs                   = f60
FR_rs2                  = f61

FR_LocalMin             = f62
FR_10                   = f63

FR_05                   = f64

FR_S32                  = f65
FR_S31                  = f66
FR_S01                  = f67
FR_S11                  = f68
FR_S21                  = f69
FR_S00                  = f70
FR_S10                  = f71
FR_S20                  = f72

FR_GAMMA                = f73
FR_2                    = f74
FR_6                    = f75




// Data tables
//==============================================================
RODATA
.align 16
LOCAL_OBJECT_START(tgammaf_data)
data8 0x3FDD8B618D5AF8FE // local minimum (0.461632144968362356785)
data8 0x4024000000000000 // 10.0
data8 0x3E90FC992FF39E13 // S32
data8 0xBEC144B2760626E2 // S31
//
//[2; 8)
data8 0x4009EFD1BA0CB3B4 // C01
data8 0x3FFFB35378FF4822 // C11
data8 0xC01032270413B896 // C41
data8 0xC01F171A4C0D6827 // C51
data8 0x40148F8E197396AC // C20
data8 0x401C601959F1249C // C30
data8 0x3EE21AD881741977 // An
data8 0x4041852200000000 // overflow boundary (35.04010009765625)
data8 0x3FD9CE68F695B198 // C21
data8 0xBFF8C30AC900DA03 // C31
data8 0x400E17D2F0535C02 // C00
data8 0x4010689240F7FAC8 // C10
data8 0x402563147DDCCF8D // C40
data8 0x4033406D0480A21C // C50
//
//[8; 16)
data8 0x4006222BAE0B793B // C01
data8 0x4002452733473EDA // C11
data8 0xC0010EF3326FDDB3 // C41
data8 0xC01492B817F99C0F // C51
data8 0x40099C905A249B75 // C20
data8 0x4012B972AE0E533D // C30
data8 0x3FE6F6DB91D0D4CC // An
data8 0x4041852200000000 // overflow boundary
data8 0x3FF545828F7B73C5 // C21
data8 0xBFBBD210578764DF // C31
data8 0x4000542098F53CFC // C00
data8 0x40032C1309AD6C81 // C10
data8 0x401D7331E19BD2E1 // C40
data8 0x402A06807295EF57 // C50
//
//[16; 24)
data8 0x4000131002867596 // C01
data8 0x3FFAA362D5D1B6F2 // C11
data8 0xBFFCB6985697DB6D // C41
data8 0xC0115BEE3BFC3B3B // C51
data8 0x3FFE62FF83456F73 // C20
data8 0x4007E33478A114C4 // C30
data8 0x41E9B2B73795ED57 // An
data8 0x4041852200000000 // overflow boundary
data8 0x3FEEB1F345BC2769 // C21
data8 0xBFC3BBE6E7F3316F // C31
data8 0x3FF14E07DA5E9983 // C00
data8 0x3FF53B76BF81E2C0 // C10
data8 0x4014051E0269A3DC // C40
data8 0x40229D4227468EDB // C50
//
//[24; 32)
data8 0x3FFAF7BD498384DE // C01
data8 0x3FF62AD8B4D1C3D2 // C11
data8 0xBFFABCADCD004C32 // C41
data8 0xC00FADE97C097EC9 // C51
data8 0x3FF6DA9ED737707E // C20
data8 0x4002A29E9E0C782C // C30
data8 0x44329D5B5167C6C3 // An
data8 0x4041852200000000 // overflow boundary
data8 0x3FE8943CBBB4B727 // C21
data8 0xBFCB39D466E11756 // C31
data8 0x3FE879AF3243D8C1 // C00
data8 0x3FEEC7DEBB14CE1E // C10
data8 0x401017B79BA80BCB // C40
data8 0x401E941DC3C4DE80 // C50
//
//[32; 40)
data8 0x3FF7ECB3A0E8FE5C // C01
data8 0x3FF3815A8516316B // C11
data8 0xBFF9ABD8FCC000C3 // C41
data8 0xC00DD89969A4195B // C51
data8 0x3FF2E43139CBF563 // C20
data8 0x3FFF96DC3474A606 // C30
data8 0x46AFF4CA9B0DDDF0 // An
data8 0x4041852200000000 // overflow boundary
data8 0x3FE4CE76DA1B5783 // C21
data8 0xBFD0524DB460BC4E // C31
data8 0x3FE35852DF14E200 // C00
data8 0x3FE8C7610359F642 // C10
data8 0x400BCF750EC16173 // C40
data8 0x401AC14E02EA701C // C50
//
//[40; 48)
data8 0x3FF5DCE4D8193097 // C01
data8 0x3FF1B0D8C4974FFA // C11
data8 0xBFF8FB450194CAEA // C41
data8 0xC00C9658E030A6C4 // C51
data8 0x3FF068851118AB46 // C20
data8 0x3FFBF7C7BB46BF7D // C30
data8 0x3FF0000000000000 // An
data8 0x4041852200000000 // overflow boundary
data8 0x3FE231DEB11D847A // C21
data8 0xBFD251ECAFD7E935 // C31
data8 0x3FE0368AE288F6BF // C00
data8 0x3FE513AE4215A70C // C10
data8 0x4008F960F7141B8B // C40
data8 0x40183BA08134397B // C50
//
//[1.0; 1.25)
data8 0xBFD9909648921868 // A7
data8 0x3FE96FFEEEA8520F // A6
data8 0xBFED0800D93449B8 // A3
data8 0x3FEFA648D144911C // A2
data8 0xBFEE3720F7720B4D // A5
data8 0x3FEF4857A010CA3B // A4
data8 0xBFE2788CCD545AA4 // A1
data8 0x3FEFFFFFFFE9209E // A0
//
//[1.25; 1.5)
data8 0xBFB421236426936C // A7
data8 0x3FAF237514F36691 // A6
data8 0xBFC0BADE710A10B9 // A3
data8 0x3FDB6C5465BBEF1F // A2
data8 0xBFB7E7F83A546EBE // A5
data8 0x3FC496A01A545163 // A4
data8 0xBDEE86A39D8452EB // A1
data8 0x3FEC56DC82A39AA2 // A0
//
//[1.5; 1.75)
data8 0xBF94730B51795867 // A7
data8 0x3FBF4203E3816C7B // A6
data8 0xBFE85B427DBD23E4 // A3
data8 0x3FEE65557AB26771 // A2
data8 0xBFD59D31BE3AB42A // A5
data8 0x3FE3C90CC8F09147 // A4
data8 0xBFE245971DF735B8 // A1
data8 0x3FEFFC613AE7FBC8 // A0
//
//[1.75; 2.0)
data8 0xBF7746A85137617E // A7
data8 0x3FA96E37D09735F3 // A6
data8 0xBFE3C24AC40AC0BB // A3
data8 0x3FEC56A80A977CA5 // A2
data8 0xBFC6F0E707560916 // A5
data8 0x3FDB262D949175BE // A4
data8 0xBFE1C1AEDFB25495 // A1
data8 0x3FEFEE1E644B2022 // A0
//
// sin(pi*x)/pi
data8 0xC026FB0D377656CC // S01
data8 0x3FFFB15F95A22324 // S11
data8 0x406CE58F4A41C6E7 // S10
data8 0x404453786302C61E // S20
data8 0xC023D59A47DBFCD3 // S21
data8 0x405541D7ABECEFCA // S00
//
// 1/An for [40; 48)
data8 0xCAA7576DE621FCD5, 0x3F68
LOCAL_OBJECT_END(tgammaf_data)

//==============================================================
// Code
//==============================================================

.section .text
GLOBAL_LIBM_ENTRY(tgammaf)
{ .mfi
      getf.exp      GR_SignExp = f8
      fma.s1        FR_NormX = f8,f1,f0
      addl          GR_ad_Data = @ltoff(tgammaf_data), gp
}
{ .mfi
      mov           GR_ExpOf05 = 0xFFFE
      fcvt.fx.trunc.s1 FR_iXt = f8 // [x]
      mov           GR_Offs = 0 // 2 <= x < 8
};;
{ .mfi
      getf.d        GR_Arg = f8
      fcmp.lt.s1    p14,p15 = f8,f0
      mov           GR_Tbl12Offs = 0
}
{ .mfi
      setf.exp      FR_05 = GR_ExpOf05
      fma.s1        FR_2 = f1,f1,f1 // 2
      mov           GR_Correction = 0
};;
{ .mfi
      ld8           GR_ad_Data = [GR_ad_Data]
      fclass.m      p10,p0 = f8,0x1E7 // is x  NaTVal, NaN, +/-0 or +/-INF?
      tbit.z        p12,p13 = GR_SignExp,16 // p13 if |x| >= 2
}
{ .mfi
      mov           GR_ExpOf1 = 0xFFFF
      fcvt.fx.s1    FR_rs = f8 // round(x)
      and           GR_Exp2Ind = 7,GR_SignExp
};;
.pred.rel "mutex",p14,p15
{ .mfi
(p15) cmp.eq.unc    p11,p0 = GR_ExpOf1,GR_SignExp // p11 if 1 <= x < 2
(p14) fma.s1        FR_1mX = f1,f1,f8 // 1 - |x|
      mov           GR_Sig = 0 // if |x| < 2
}
{ .mfi
(p13) cmp.eq.unc    p7,p0 = 2,GR_Exp2Ind
(p15) fms.s1        FR_1mX = f1,f1,f8 // 1 - |x|
(p13) cmp.eq.unc    p8,p0 = 3,GR_Exp2Ind
};;
.pred.rel "mutex",p7,p8
{ .mfi
(p7)  mov           GR_Offs = 0x7    // 8 <= |x| < 16
      nop.f         0
(p8)  tbit.z.unc    p0,p6 = GR_Arg,51
}
{ .mib
(p13) cmp.lt.unc    p9,p0 = 3,GR_Exp2Ind
(p8)  mov           GR_Offs = 0xE // 16 <= |x| < 32
      // jump if x is NaTVal, NaN, +/-0 or +/-INF?
(p10) br.cond.spnt  tgammaf_spec_args
};;
.pred.rel "mutex",p14,p15
.pred.rel "mutex",p6,p9
{ .mfi
(p9)  mov           GR_Offs = 0x1C // 32 <= |x|
(p14) fma.s1        FR_X2mX = FR_NormX,FR_NormX,FR_NormX // x^2-|x|
(p9)  tbit.z.unc    p0,p8 = GR_Arg,50
}
{ .mfi
      ldfpd         FR_LocalMin,FR_10 = [GR_ad_Data],16
(p15) fms.s1        FR_X2mX = FR_NormX,FR_NormX,FR_NormX // x^2-|x|
(p6)  add           GR_Offs = 0x7,GR_Offs // 24 <= x < 32
};;
.pred.rel "mutex",p8,p12
{ .mfi
      add           GR_ad_Ce = 0x50,GR_ad_Data
(p15) fcmp.lt.unc.s1 p10,p0 = f8,f1 // p10 if 0 <= x < 1
      mov           GR_OvfNzBound = 2
}
{ .mib
      ldfpd         FR_S32,FR_S31 = [GR_ad_Data],16
(p8)  add           GR_Offs = 0x7,GR_Offs // 40 <= |x|
      // jump if 1 <= x < 2
(p11) br.cond.spnt  tgammaf_from_1_to_2
};;
{ .mfi
      shladd        GR_ad_Ce = GR_Offs,4,GR_ad_Ce
      fcvt.xf       FR_Xt = FR_iXt // [x]
(p13) cmp.eq.unc    p7,p0 = r0,GR_Offs // p7 if 2 <= |x| < 8
}
{ .mfi
      shladd        GR_ad_Co = GR_Offs,4,GR_ad_Data
      fma.s1        FR_6 = FR_2,FR_2,FR_2
      mov           GR_ExpOf05 = 0x7FC
};;
{ .mfi
(p13) getf.sig      GR_Sig = FR_iXt // if |x| >= 2
      frcpa.s1      FR_Rcp0,p0 = f1,FR_NormX
(p10) shr           GR_Arg = GR_Arg,51
}
{ .mib
      ldfpd         FR_C01,FR_C11 = [GR_ad_Co],16
(p7)  mov           GR_Correction = 2
      // jump if 0 < x < 1
(p10) br.cond.spnt  tgammaf_from_0_to_1
};;
{ .mfi
      ldfpd         FR_C21,FR_C31 = [GR_ad_Ce],16
      fma.s1        FR_Rq2 = f1,f1,FR_1mX // 2 - |x|
(p14) sub           GR_Correction = r0,GR_Correction
}
{ .mfi
      ldfpd         FR_C41,FR_C51 = [GR_ad_Co],16
(p14) fcvt.xf       FR_rs = FR_rs
(p14) add           GR_ad_SinO = 0x3A0,GR_ad_Data
};;
.pred.rel "mutex",p14,p15
{ .mfi
      ldfpd         FR_C00,FR_C10 = [GR_ad_Ce],16
      nop.f         0
(p14) sub           GR_Sig = GR_Correction,GR_Sig
}
{ .mfi
      ldfpd         FR_C20,FR_C30 = [GR_ad_Co],16
      fma.s1        FR_Rq1 = FR_1mX,FR_2,FR_X2mX // (x-1)*(x-2)
(p15) sub           GR_Sig = GR_Sig,GR_Correction
};;
{ .mfi
(p14) ldfpd         FR_S01,FR_S11 = [GR_ad_SinO],16
      fma.s1        FR_Rq3 = FR_2,f1,FR_1mX // 3 - |x|
      and           GR_RqDeg = 0x6,GR_Sig
}
{ .mfi
      ldfpd         FR_C40,FR_C50 = [GR_ad_Ce],16
(p14) fma.d.s0      FR_X = f0,f0,f8 // set deno flag
      mov           GR_NanBound = 0x30016 // -2^23
};;
.pred.rel "mutex",p14,p15
{ .mfi
(p14) add           GR_ad_SinE = 0x3C0,GR_ad_Data
(p15) fms.s1        FR_r = FR_NormX,f1,FR_Xt // r = x - [x]
      cmp.eq        p8,p0 = 2,GR_RqDeg
}
{ .mfi
      ldfpd         FR_An,FR_OvfBound = [GR_ad_Co]
(p14) fms.s1        FR_r = FR_Xt,f1,FR_NormX // r = |x - [x]|
      cmp.eq        p9,p0 = 4,GR_RqDeg
};;
.pred.rel "mutex",p8,p9
{ .mfi
(p14) ldfpd         FR_S21,FR_S00 = [GR_ad_SinE],16
(p8)  fma.s1        FR_Rq0 = FR_2,f1,FR_1mX // (3-x)
      tbit.z        p0,p6 = GR_Sig,0
}
{ .mfi
(p14) ldfpd         FR_S10,FR_S20 = [GR_ad_SinO],16
(p9)  fma.s1        FR_Rq0 = FR_2,FR_2,FR_1mX // (5-x)
      cmp.eq        p10,p0 = 6,GR_RqDeg
};;
{ .mfi
(p14) getf.s        GR_Arg = f8
(p14) fcmp.eq.unc.s1 p13,p0 = FR_NormX,FR_Xt
(p14) mov           GR_ZeroResBound = 0xC22C // -43
}
{ .mfi
(p14) ldfe          FR_InvAn = [GR_ad_SinE]
(p10) fma.s1        FR_Rq0 = FR_6,f1,FR_1mX // (7-x)
      cmp.eq        p7,p0 = r0,GR_RqDeg
};;
{ .mfi
(p14) cmp.ge.unc    p11,p0 = GR_SignExp,GR_NanBound
      fma.s1        FR_Rq2 = FR_Rq2,FR_6,FR_X2mX // (x-3)*(x-4)
(p14) shl           GR_ZeroResBound = GR_ZeroResBound,16
}
{ .mfb
(p14) mov           GR_OvfNzBound = 0x802
(p14) fms.s1        FR_rs = FR_rs,f1,FR_NormX // rs = round(x) - x
      // jump if  x < -2^23 i.e. x is negative integer
(p11) br.cond.spnt  tgammaf_singularity
};;
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_Rq1 = f0,f0,f1
(p14) shl           GR_OvfNzBound = GR_OvfNzBound,20
}
{ .mfb
      nop.m         0
      fma.s1        FR_Rq3 = FR_Rq3,FR_10,FR_X2mX // (x-5)*(x-6)
      // jump if x is negative integer such that -2^23 < x < 0
(p13) br.cond.spnt  tgammaf_singularity
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C01 = FR_C01,f1,FR_r
(p14) mov           GR_ExpOf05 = 0xFFFE
}
{ .mfi
(p14) cmp.eq.unc    p7,p0 = GR_Arg,GR_OvfNzBound
      fma.s1        FR_C11 = FR_C11,f1,FR_r
(p14) cmp.ltu.unc   p11,p0 = GR_Arg,GR_OvfNzBound
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,f1,FR_r
(p14) cmp.ltu.unc   p9,p0 = GR_ZeroResBound,GR_Arg
}
{ .mfb
      nop.m         0
      fma.s1        FR_C31 = FR_C31,f1,FR_r
      // jump if argument is close to 0 negative
(p11) br.cond.spnt  tgammaf_overflow
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C41 = FR_C41,f1,FR_r
      nop.i         0
}
{ .mfb
      nop.m         0
      fma.s1        FR_C51 = FR_C51,f1,FR_r
      // jump if x is negative noninteger such that -2^23 < x < -43
(p9)  br.cond.spnt  tgammaf_underflow
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_rs2 = FR_rs,FR_rs,f0
      nop.i         0
}
{ .mfb
      nop.m         0
(p14) fma.s1        FR_S01 = FR_rs,FR_rs,FR_S01
      // jump if argument is 0x80200000
(p7)  br.cond.spnt  tgammaf_overflow_near0_bound
};;
{ .mfi
      nop.m         0
(p6)  fnma.s1       FR_Rq1 = FR_Rq1,FR_Rq0,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_Rq2 = FR_Rq2,FR_Rq3,f0
      and           GR_Sig = 0x7,GR_Sig
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C01 = FR_C01,FR_r,FR_C00
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C11 = FR_C11,FR_r,FR_C10
      cmp.eq        p6,p7 = r0,GR_Sig // p6 if |x| from one of base intervals
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_r,FR_C20
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C31 = FR_C31,FR_r,FR_C30
(p7)  cmp.lt.unc    p9,p0 = 2,GR_RqDeg
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S11 = FR_rs,FR_rs,FR_S11
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S21 = FR_rs,FR_rs,FR_S21
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C41 = FR_C41,FR_r,FR_C40
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S32 = FR_rs2,FR_S32,FR_S31
      nop.i         0
};;
{ .mfi
      nop.m         0
(p9)  fma.s1        FR_Rq1 = FR_Rq1,FR_Rq2,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C51 = FR_C51,FR_r,FR_C50
      nop.i         0
};;
{ .mfi
(p14) getf.exp      GR_SignExp = FR_rs
      fma.s1        FR_C01 = FR_C01,FR_C11,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S01 = FR_S01,FR_rs2,FR_S00
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_C31,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      // NR-iteration
(p14) fnma.s1       FR_InvNormX1 = FR_Rcp0,FR_NormX,f1
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S11 = FR_S11,FR_rs2,FR_S10
(p14) tbit.z.unc    p11,p12 = GR_SignExp,17
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S21 = FR_S21,FR_rs2,FR_S20
      nop.i         0
};;
{ .mfi
      nop.m         0
(p15) fcmp.lt.unc.s1 p0,p13 = FR_NormX,FR_OvfBound
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S32 = FR_rs2,FR_S32,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C41 = FR_C41,FR_C51,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_An = FR_Rq1,FR_An,f0
      nop.i         0
};;
{ .mfb
      nop.m         0
      nop.f         0
      // jump if x > 35.04010009765625
(p13) br.cond.spnt  tgammaf_overflow
};;
{ .mfi
      nop.m         0
      // NR-iteration
(p14) fma.s1        FR_InvNormX1 = FR_Rcp0,FR_InvNormX1,FR_Rcp0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S01 = FR_S01,FR_S11,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_S21 = FR_S21,FR_S32,f0
      nop.i         0
};;
{ .mfi
(p14) getf.exp      GR_SignExp = FR_NormX
      fma.s1        FR_C01 = FR_C01,FR_C21,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C41 = FR_C41,FR_An,f0
(p14) mov           GR_ExpOf1 = 0x2FFFF
};;
{ .mfi
      nop.m         0
      // NR-iteration
(p14) fnma.s1       FR_InvNormX2 = FR_InvNormX1,FR_NormX,f1
      nop.i         0
};;
.pred.rel "mutex",p11,p12
{ .mfi
      nop.m         0
(p12) fnma.s1       FR_S01 = FR_S01,FR_S21,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p11) fma.s1        FR_S01 = FR_S01,FR_S21,f0
      nop.i         0
};;

{ .mfi
      nop.m         0
(p14) fma.s1        FR_GAMMA = FR_C01,FR_C41,f0
(p14) tbit.z.unc    p6,p7 = GR_Sig,0
}
{ .mfb
      nop.m         0
(p15) fma.s.s0      f8 = FR_C01,FR_C41,f0
(p15) br.ret.spnt   b0 // exit for positives
};;
.pred.rel "mutex",p11,p12
{ .mfi
      nop.m         0
(p12) fms.s1        FR_S01 = FR_rs,FR_S01,FR_rs
      nop.i         0
}
{ .mfi
      nop.m         0
(p11) fma.s1        FR_S01 = FR_rs,FR_S01,FR_rs
      nop.i         0
};;
{ .mfi
      nop.m         0
      // NR-iteration
      fma.s1        FR_InvNormX2 = FR_InvNormX1,FR_InvNormX2,FR_InvNormX1
      cmp.eq        p10,p0 = 0x23,GR_Offs
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_GAMMA = FR_S01,FR_GAMMA,f0
      cmp.gtu       p8,p0 = GR_SignExp,GR_ExpOf1
}
{ .mfi
      nop.m         0
(p7)  fnma.s1       FR_GAMMA = FR_S01,FR_GAMMA,f0
      cmp.eq        p9,p0 = GR_SignExp,GR_ExpOf1
};;
{ .mfi
      nop.m         0
      // NR-iteration
      fnma.s1       FR_InvNormX1 = FR_InvNormX2,FR_NormX,f1
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_InvNormX2 = FR_InvNormX2,FR_InvAn,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      frcpa.s1      FR_Rcp0,p0 = f1,FR_GAMMA
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_Multplr = FR_NormX,f1,f1 // x - 1
      nop.i         0
};;
{ .mfi
      nop.m         0
      // NR-iteration
      fnma.s1       FR_Rcp1 = FR_Rcp0,FR_GAMMA,f1
      nop.i         0
};;
.pred.rel "mutex",p8,p9
{ .mfi
      nop.m         0
      // 1/x or 1/(An*x)
(p8)  fma.s1        FR_Multplr = FR_InvNormX2,FR_InvNormX1,FR_InvNormX2
      nop.i         0
}
{ .mfi
      nop.m         0
(p9)  fma.s1        FR_Multplr = f1,f1,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      // NR-iteration
      fma.s1        FR_Rcp1 = FR_Rcp0,FR_Rcp1,FR_Rcp0
      nop.i         0
};;
{ .mfi
      nop.m         0
      // NR-iteration
      fnma.s1       FR_Rcp2 = FR_Rcp1,FR_GAMMA,f1
      nop.i         0
}
{ .mfi
      nop.m         0
      // NR-iteration
      fma.s1        FR_Rcp1 = FR_Rcp1,FR_Multplr,f0
      nop.i         0
};;
{ .mfb
      nop.m         0
      fma.s.s0      f8 = FR_Rcp1,FR_Rcp2,FR_Rcp1
      br.ret.sptk   b0
};;

// here if 0 < x < 1
//--------------------------------------------------------------------
.align 32
tgammaf_from_0_to_1:
{ .mfi
      cmp.lt        p7,p0 = GR_Arg,GR_ExpOf05
      // NR-iteration
      fnma.s1       FR_Rcp1 = FR_Rcp0,FR_NormX,f1
      cmp.eq        p8,p0 = GR_Arg,GR_ExpOf05
}
{ .mfi
      cmp.gt        p9,p0 = GR_Arg,GR_ExpOf05
      fma.s1        FR_r = f0,f0,FR_NormX // reduced arg for (0;1)
      mov           GR_ExpOf025 = 0x7FA
};;
{ .mfi
      getf.s        GR_ArgNz = f8
      fma.d.s0      FR_X = f0,f0,f8 // set deno flag
      shl           GR_OvfNzBound = GR_OvfNzBound,20
}
{ .mfi
(p8)  mov           GR_Tbl12Offs = 0x80 // 0.5 <= x < 0.75
      nop.f         0
(p7)  cmp.ge.unc    p6,p0 = GR_Arg,GR_ExpOf025
};;
.pred.rel "mutex",p6,p9
{ .mfi
(p9)  mov           GR_Tbl12Offs = 0xC0 // 0.75 <= x < 1
      nop.f         0
(p6)  mov           GR_Tbl12Offs = 0x40 // 0.25 <= x < 0.5
}
{ .mfi
      add           GR_ad_Ce = 0x2C0,GR_ad_Data
      nop.f         0
      add           GR_ad_Co = 0x2A0,GR_ad_Data
};;
{ .mfi
      add           GR_ad_Co = GR_ad_Co,GR_Tbl12Offs
      nop.f         0
      cmp.lt        p12,p0 = GR_ArgNz,GR_OvfNzBound
}
{ .mib
      add           GR_ad_Ce = GR_ad_Ce,GR_Tbl12Offs
      cmp.eq        p7,p0 = GR_ArgNz,GR_OvfNzBound
      // jump if argument is 0x00200000
(p7)  br.cond.spnt  tgammaf_overflow_near0_bound
};;
{ .mmb
      ldfpd         FR_A7,FR_A6 = [GR_ad_Co],16
      ldfpd         FR_A5,FR_A4 = [GR_ad_Ce],16
      // jump if argument is close to 0 positive
(p12) br.cond.spnt  tgammaf_overflow
};;
{ .mfi
      ldfpd         FR_A3,FR_A2 = [GR_ad_Co],16
      // NR-iteration
      fma.s1        FR_Rcp1 = FR_Rcp0,FR_Rcp1,FR_Rcp0
      nop.i         0
}
{ .mfb
      ldfpd         FR_A1,FR_A0 = [GR_ad_Ce],16
      nop.f         0
      br.cond.sptk  tgamma_from_0_to_2
};;

// here if 1 < x < 2
//--------------------------------------------------------------------
.align 32
tgammaf_from_1_to_2:
{ .mfi
      add           GR_ad_Co = 0x2A0,GR_ad_Data
      fms.s1        FR_r = f0,f0,FR_1mX
      shr           GR_TblOffs = GR_Arg,47
}
{ .mfi
      add           GR_ad_Ce = 0x2C0,GR_ad_Data
      nop.f         0
      mov           GR_TblOffsMask = 0x18
};;
{ .mfi
      nop.m         0
      nop.f         0
      and           GR_TblOffs = GR_TblOffs,GR_TblOffsMask
};;
{ .mfi
      shladd        GR_ad_Co = GR_TblOffs,3,GR_ad_Co
      nop.f         0
      nop.i         0
}
{ .mfi
      shladd        GR_ad_Ce = GR_TblOffs,3,GR_ad_Ce
      nop.f         0
      cmp.eq        p6,p7 = 8,GR_TblOffs
};;
{ .mmi
      ldfpd         FR_A7,FR_A6 = [GR_ad_Co],16
      ldfpd         FR_A5,FR_A4 = [GR_ad_Ce],16
      nop.i         0
};;
{ .mmi
      ldfpd         FR_A3,FR_A2 = [GR_ad_Co],16
      ldfpd         FR_A1,FR_A0 = [GR_ad_Ce],16
      nop.i         0
};;

.align 32
tgamma_from_0_to_2:
{ .mfi
      nop.m         0
(p6)  fms.s1        FR_r = FR_r,f1,FR_LocalMin
      nop.i         0
};;
{ .mfi
      nop.m         0
      // NR-iteration
(p10) fnma.s1       FR_Rcp2 = FR_Rcp1,FR_NormX,f1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_r2 = FR_r,FR_r,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r,FR_A6
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A5 = FR_A5,FR_r,FR_A4
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_r,FR_A2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A1,FR_r,FR_A0
      nop.i         0
};;
{ .mfi
      nop.m         0
      // NR-iteration
(p10) fma.s1        FR_Rcp2 = FR_Rcp1,FR_Rcp2,FR_Rcp1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r2,FR_A5
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_r4 = FR_r2,FR_r2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_r2,FR_A1
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fma.s1        FR_GAMMA = FR_A7,FR_r4,FR_A3
      nop.i         0
}
{ .mfi
      nop.m         0
(p11) fma.s.s0      f8 = FR_A7,FR_r4,FR_A3
      nop.i         0
};;
{ .mfb
      nop.m         0
(p10) fma.s.s0      f8 = FR_GAMMA,FR_Rcp2,f0
      br.ret.sptk   b0
};;


// overflow
//--------------------------------------------------------------------
.align 32
tgammaf_overflow_near0_bound:
.pred.rel "mutex",p14,p15
{ .mfi
	  mov           GR_fpsr = ar.fpsr
	  nop.f         0
(p15) mov           r8 = 0x7f8
}
{ .mfi
      nop.m         0
      nop.f         0
(p14) mov           r8 = 0xff8
};;
{ .mfi
	  nop.m         0
	  nop.f         0
	  shl           r8 = r8,20
};;
{ .mfi
      sub           r8 = r8,r0,1
      nop.f         0
	  extr.u        GR_fpsr = GR_fpsr,10,2 // rounding mode
};;
.pred.rel "mutex",p14,p15
{ .mfi
      // set p8 to 0 in case of overflow and to 1 otherwise
	  // for negative arg:
	  //    no overflow if rounding mode either Z or +Inf, i.e.
	  //    GR_fpsr > 1
(p14) cmp.lt        p8,p0 = 1,GR_fpsr
      nop.f         0
	  // for positive arg:
	  //    no overflow if rounding mode either Z or -Inf, i.e.
	  //    (GR_fpsr & 1) == 0
(p15) tbit.z        p0,p8 = GR_fpsr,0
};;
{ .mib
(p8)  setf.s        f8 = r8 // set result to 0x7f7fffff without
                            // OVERFLOW flag raising
      nop.i         0
(p8)  br.ret.sptk   b0
};;

.align 32
tgammaf_overflow:
{ .mfi
      nop.m         0
      nop.f         0
      mov           r8 = 0x1FFFE
};;
{ .mfi
      setf.exp      f9 = r8
      fmerge.s      FR_X = f8,f8
      nop.i         0
};;
.pred.rel "mutex",p14,p15
{ .mfi
      nop.m         0
(p14) fnma.s.s0     f8 = f9,f9,f0 // set I,O and -INF result
      mov           GR_TAG = 261 // overflow
}
{ .mfb
      nop.m         0
(p15) fma.s.s0      f8 = f9,f9,f0 // set I,O and +INF result
      br.cond.sptk  tgammaf_libm_err
};;

// x is negative integer or +/-0
//--------------------------------------------------------------------
.align 32
tgammaf_singularity:
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8
      mov           GR_TAG = 262 // negative
}
{ .mfb
      nop.m         0
      frcpa.s0      f8,p0 = f0,f0
      br.cond.sptk  tgammaf_libm_err
};;
// x is negative noninteger with big absolute value
//--------------------------------------------------------------------
.align 32
tgammaf_underflow:
{ .mfi
      mov           r8 = 0x00001
      nop.f         0
      tbit.z        p6,p7 = GR_Sig,0
};;
{ .mfi
      setf.exp      f9 = r8
      nop.f         0
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fms.s.s0      f8 = f9,f9,f9
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.s.s0      f8 = f9,f9,f9
      br.ret.sptk   b0
};;

//  x for natval, nan, +/-inf or +/-0
//--------------------------------------------------------------------
.align 32
tgammaf_spec_args:
{ .mfi
      nop.m         0
      fclass.m      p6,p0 =  f8,0x1E1 // Test x for natval, nan, +inf
      nop.i         0
};;
{ .mfi
      nop.m         0
      fclass.m      p7,p8 =  f8,0x7 // +/-0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8
      nop.i         0
}
{ .mfb
      nop.m         0
(p6)  fma.s.s0      f8 = f8,f1,f8
(p6)  br.ret.spnt   b0
};;
.pred.rel "mutex",p7,p8
{ .mfi
(p7)  mov           GR_TAG = 262 // negative
(p7)  frcpa.s0      f8,p0 = f1,f8
      nop.i         0
}
{ .mib
      nop.m         0
      nop.i         0
(p8)  br.cond.spnt  tgammaf_singularity
};;

.align 32
tgammaf_libm_err:
{ .mfi
      alloc        r32 = ar.pfs,1,4,4,0
      nop.f        0
      mov          GR_Parameter_TAG = GR_TAG
};;

GLOBAL_LIBM_END(tgammaf)
libm_alias_float_other (tgamma, tgamma)

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
        stfs [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfs [GR_Parameter_X] = FR_X           // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
        nop.b 0
}
{ .mib
        stfs [GR_Parameter_Y] = FR_RESULT      // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support# // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT]       // Get return result off stack
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
        mov   b0 = GR_SAVE_B0                  // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
};;

LOCAL_LIBM_END(__libm_error_region)
.type   __libm_error_support#,@function
.global __libm_error_support#
