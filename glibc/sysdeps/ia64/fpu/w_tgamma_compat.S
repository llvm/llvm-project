.file "tgamma.s"


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
// 10/12/01  Initial version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 02/10/03  Reordered header: .section, .global, .proc, .align
// 04/04/03  Changed error codes for overflow and negative integers
// 04/10/03  Changed code for overflow near zero handling
// 03/31/05  Reformatted delimiters between data tables
//
//*********************************************************************
//
//*********************************************************************
//
// Function: tgamma(x) computes the principle value of the GAMMA
// function of x.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8-f15
//                              f33-f87
//
//    General Purpose Registers:
//      r8-r11
//      r14-r28
//      r32-r36
//      r37-r40 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    tgamma(+inf) = +inf
//    tgamma(-inf) = QNaN
//    tgamma(+/-0) = +/-inf
//    tgamma(x<0, x - integer) = QNaN
//    tgamma(SNaN) = QNaN
//    tgamma(QNaN) = QNaN
//
//*********************************************************************
//
// Overview
//
// The method consists of three cases.
//
// If       2 <= x < OVERFLOW_BOUNDARY      use case tgamma_regular;
// else if    0 < x < 2                     use case tgamma_from_0_to_2;
// else    if  -(i+1) <  x < -i, i = 0...184 use case tgamma_negatives;
//
// Case 2 <= x < OVERFLOW_BOUNDARY
// -------------------------------
//   Here we use algorithm based on the recursive formula
//   GAMMA(x+1) = x*GAMMA(x). For that we subdivide interval
//   [2; OVERFLOW_BOUNDARY] into intervals [16*n; 16*(n+1)] and
//   approximate GAMMA(x) by polynomial of 22th degree on each
//   [16*n; 16*n+1], recursive formula is used to expand GAMMA(x)
//   to [16*n; 16*n+1]. In other words we need to find n, i and r
//   such that x = 16 * n + i + r where n and i are integer numbers
//   and r is fractional part of x. So GAMMA(x) = GAMMA(16*n+i+r) =
//   = (x-1)*(x-2)*...*(x-i)*GAMMA(x-i) =
//   = (x-1)*(x-2)*...*(x-i)*GAMMA(16*n+r) ~
//   ~ (x-1)*(x-2)*...*(x-i)*P22n(r).
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
//    P22n(r) = A22*(r^2+C01(n)*R+C00(n))*
//              *(r^2+C11(n)*R+C10(n))*...*(r^2+CA1(n)*R+CA0(n))
//
//   Step 3: Recursion
//   -----------------
//    In case when i > 0 we need to multiply P22n(r) by product
//    R(i)=(x-1)*(x-2)*...*(x-i). To reduce number of fp-instructions
//    we can calculate R as follow:
//    R(i) = ((x-1)*(x-2))*((x-3)*(x-4))*...*((x-(i-1))*(x-i)) if i is
//    even or R = ((x-1)*(x-2))*((x-3)*(x-4))*...*((x-(i-2))*(x-(i-1)))*
//    *(i-1) if i is odd. In both cases we need to calculate
//    R2(i) = (x^2-3*x+2)*(x^2-7*x+12)*...*(x^2+x+2*j*(2*j-1)) =
//    = (x^2-3*x+2)*(x^2-7*x+12)*...*((x^2+x)+2*j*(2*(j-1)+(1-2*x))) =
//    = (RA+2*(2-RB))*(RA+4*(4-RB))*...*(RA+2*j*(2*(j-1)+RB))
//    where j = 1..[i/2], RA = x^2+x, RB = 1-2*x.
//
//   Step 4: Reconstruction
//   ----------------------
//    Reconstruction is just simple multiplication i.e.
//    GAMMA(x) = P22n(r)*R(i)
//
// Case 0 < x < 2
// --------------
//    To calculate GAMMA(x) on this interval we do following
//        if 1 <= x < 1.25   than  GAMMA(x) = P15(x-1)
//        if 1.25 <= x < 1.5 than  GAMMA(x) = P15(x-x_min) where
//        x_min is point of local minimum on [1; 2] interval.
//        if 1.5  <= x < 2.0 than  GAMMA(x) = P15(x-1.5)
//    and
//        if 0 < x < 1 than GAMMA(x) = GAMMA(x+1)/x
//
// Case -(i+1) <  x < -i, i = 0...184
// ----------------------------------
//    Here we use the fact that GAMMA(-x) = PI/(x*GAMMA(x)*sin(PI*x)) and
//    so we need to calculate GAMMA(x), sin(PI*x)/PI. Calculation of
//    GAMMA(x) is described above.
//
//   Step 1: Reduction
//   -----------------
//    Note that period of sin(PI*x) is 2 and range reduction for
//    sin(PI*x) is like to range reduction for GAMMA(x)
//    i.e r = x - [x] with exception of cases
//    when r > 0.5 (in such cases r = 1 - (x - [x])).
//
//   Step 2: Approximation
//   ---------------------
//    To approximate sin(PI*x)/PI = sin(PI*(2*n+r))/PI =
//    = (-1)^n*sin(PI*r)/PI Taylor series is used.
//    sin(PI*r)/PI ~ S21(r).
//
//   Step 3: Division
//   ----------------
//    To calculate 1/(x*GAMMA(x)*S21(r)) we use frcpa instruction
//    with following Newton-Raphson interations.
//
//
//*********************************************************************

GR_Sig                  = r8
GR_TAG                  = r8
GR_ad_Data              = r9
GR_SigRqLin             = r10
GR_iSig                 = r11
GR_ExpOf1               = r11
GR_ExpOf8               = r11


GR_Sig2                 = r14
GR_Addr_Mask1           = r15
GR_Sign_Exp             = r16
GR_Tbl_Offs             = r17
GR_Addr_Mask2           = r18
GR_ad_Co                = r19
GR_Bit2                 = r19
GR_ad_Ce                = r20
GR_ad_Co7               = r21
GR_NzOvfBound           = r21
GR_ad_Ce7               = r22
GR_Tbl_Ind              = r23
GR_Tbl_16xInd           = r24
GR_ExpOf025             = r24
GR_ExpOf05              = r25
GR_0x30033              = r26
GR_10                   = r26
GR_12                   = r27
GR_185                  = r27
GR_14                   = r28
GR_2                    = r28
GR_fpsr                 = r28

GR_SAVE_B0              = r33
GR_SAVE_PFS             = r34
GR_SAVE_GP              = r35
GR_SAVE_SP              = r36

GR_Parameter_X          = r37
GR_Parameter_Y          = r38
GR_Parameter_RESULT     = r39
GR_Parameter_TAG        = r40



FR_X                    = f10
FR_Y                    = f1 // tgamma is single argument function
FR_RESULT               = f8

FR_AbsX                 = f9
FR_NormX                = f9
FR_r02                  = f11
FR_AbsXp1               = f12
FR_X2pX                 = f13
FR_1m2X                 = f14
FR_Rq1                  = f14
FR_Xt                   = f15

FR_r                    = f33
FR_OvfBound             = f34
FR_Xmin                 = f35
FR_2                    = f36
FR_Rcp1                 = f36
FR_Rcp3                 = f36
FR_4                    = f37
FR_5                    = f38
FR_6                    = f39
FR_8                    = f40
FR_10                   = f41
FR_12                   = f42
FR_14                   = f43
FR_GAMMA                = f43
FR_05                   = f44

FR_Rq2                  = f45
FR_Rq3                  = f46
FR_Rq4                  = f47
FR_Rq5                  = f48
FR_Rq6                  = f49
FR_Rq7                  = f50
FR_RqLin                = f51

FR_InvAn                = f52

FR_C01                  = f53
FR_A15                  = f53
FR_C11                  = f54
FR_A14                  = f54
FR_C21                  = f55
FR_A13                  = f55
FR_C31                  = f56
FR_A12                  = f56
FR_C41                  = f57
FR_A11                  = f57
FR_C51                  = f58
FR_A10                  = f58
FR_C61                  = f59
FR_A9                   = f59
FR_C71                  = f60
FR_A8                   = f60
FR_C81                  = f61
FR_A7                   = f61
FR_C91                  = f62
FR_A6                   = f62
FR_CA1                  = f63
FR_A5                   = f63
FR_C00                  = f64
FR_A4                   = f64
FR_rs2                  = f64
FR_C10                  = f65
FR_A3                   = f65
FR_rs3                  = f65
FR_C20                  = f66
FR_A2                   = f66
FR_rs4                  = f66
FR_C30                  = f67
FR_A1                   = f67
FR_rs7                  = f67
FR_C40                  = f68
FR_A0                   = f68
FR_rs8                  = f68
FR_C50                  = f69
FR_r2                   = f69
FR_C60                  = f70
FR_r3                   = f70
FR_C70                  = f71
FR_r4                   = f71
FR_C80                  = f72
FR_r7                   = f72
FR_C90                  = f73
FR_r8                   = f73
FR_CA0                  = f74
FR_An                   = f75

FR_S21                  = f76
FR_S19                  = f77
FR_Rcp0                 = f77
FR_Rcp2                 = f77
FR_S17                  = f78
FR_S15                  = f79
FR_S13                  = f80
FR_S11                  = f81
FR_S9                   = f82
FR_S7                   = f83
FR_S5                   = f84
FR_S3                   = f85

FR_iXt                  = f86
FR_rs                   = f87


// Data tables
//==============================================================
RODATA
.align 16

LOCAL_OBJECT_START(tgamma_data)
data8 0x406573FAE561F648 // overflow boundary (171.624376956302739927196)
data8 0x3FDD8B618D5AF8FE // point of local minium (0.461632144968362356785)
//
//[2; 3]
data8 0xEF0E85C9AE40ABE2,0x00004000 // C01
data8 0xCA2049DDB4096DD8,0x00004000 // C11
data8 0x99A203B4DC2D1A8C,0x00004000 // C21
data8 0xBF5D9D9C0C295570,0x00003FFF // C31
data8 0xE8DD037DEB833BAB,0x00003FFD // C41
data8 0xB6AE39A2A36AA03A,0x0000BFFE // C51
data8 0x804960DC2850277B,0x0000C000 // C61
data8 0xD9F3973841C09F80,0x0000C000 // C71
data8 0x9C198A676F8A2239,0x0000C001 // C81
data8 0xC98B7DAE02BE3226,0x0000C001 // C91
data8 0xE9CAF31AC69301BA,0x0000C001 // CA1
data8 0xFBBDD58608A0D172,0x00004000 // C00
data8 0xFDD0316D1E078301,0x00004000 // C10
data8 0x8630B760468C15E4,0x00004001 // C20
data8 0x93EDE20E47D9152E,0x00004001 // C30
data8 0xA86F3A38C77D6B19,0x00004001 // C40
//[16; 17]
data8 0xF87F757F365EE813,0x00004000 // C01
data8 0xECA84FBA92759DA4,0x00004000 // C11
data8 0xD4E0A55E07A8E913,0x00004000 // C21
data8 0xB0EB45E94C8A5F7B,0x00004000 // C31
data8 0x8050D6B4F7C8617D,0x00004000 // C41
data8 0x8471B111AA691E5A,0x00003FFF // C51
data8 0xADAF462AF96585C9,0x0000BFFC // C61
data8 0xD327C7A587A8C32B,0x0000BFFF // C71
data8 0xDEF5192B4CF5E0F1,0x0000C000 // C81
data8 0xBADD64BB205AEF02,0x0000C001 // C91
data8 0x9330A24AA67D6860,0x0000C002 // CA1
data8 0xF57EEAF36D8C47BE,0x00004000 // C00
data8 0x807092E12A251B38,0x00004001 // C10
data8 0x8C458F80DEE7ED1C,0x00004001 // C20
data8 0x9F30C731DC77F1A6,0x00004001 // C30
data8 0xBAC4E7E099C3A373,0x00004001 // C40
//[32; 33]
data8 0xC3059A415F142DEF,0x00004000 // C01
data8 0xB9C1DAC24664587A,0x00004000 // C11
data8 0xA7101D910992FFB2,0x00004000 // C21
data8 0x8A9522B8E4AA0AB4,0x00004000 // C31
data8 0xC76A271E4BA95DCC,0x00003FFF // C41
data8 0xC5D6DE2A38DB7FF2,0x00003FFE // C51
data8 0xDBA42086997818B2,0x0000BFFC // C61
data8 0xB8EDDB1424C1C996,0x0000BFFF // C71
data8 0xBF7372FB45524B5D,0x0000C000 // C81
data8 0xA03DDE759131580A,0x0000C001 // C91
data8 0xFDA6FC4022C1FFE3,0x0000C001 // CA1
data8 0x9759ABF797B2533D,0x00004000 // C00
data8 0x9FA160C6CF18CEC5,0x00004000 // C10
data8 0xB0EFF1E3530E0FCD,0x00004000 // C20
data8 0xCCD60D5C470165D1,0x00004000 // C30
data8 0xF5E53F6307B0B1C1,0x00004000 // C40
//[48; 49]
data8 0xAABE577FBCE37F5E,0x00004000 // C01
data8 0xA274CAEEB5DF7172,0x00004000 // C11
data8 0x91B90B6646C1B924,0x00004000 // C21
data8 0xF06718519CA256D9,0x00003FFF // C31
data8 0xAA9EE181C0E30263,0x00003FFF // C41
data8 0xA07BDB5325CB28D2,0x00003FFE // C51
data8 0x86C8B873204F9219,0x0000BFFD // C61
data8 0xB0192C5D3E4787D6,0x0000BFFF // C71
data8 0xB1E0A6263D4C19EF,0x0000C000 // C81
data8 0x93BA32A118EAC9AE,0x0000C001 // C91
data8 0xE942A39CD9BEE887,0x0000C001 // CA1
data8 0xE838B0957B0D3D0D,0x00003FFF // C00
data8 0xF60E0F00074FCF34,0x00003FFF // C10
data8 0x89869936AE00C2A5,0x00004000 // C20
data8 0xA0FE4E8AA611207F,0x00004000 // C30
data8 0xC3B1229CFF1DDAFE,0x00004000 // C40
//[64; 65]
data8 0x9C00DDF75CDC6183,0x00004000 // C01
data8 0x9446AE9C0F6A833E,0x00004000 // C11
data8 0x84ABC5083310B774,0x00004000 // C21
data8 0xD9BA3A0977B1ED83,0x00003FFF // C31
data8 0x989B18C99411D300,0x00003FFF // C41
data8 0x886E66402318CE6F,0x00003FFE // C51
data8 0x99028C2468F18F38,0x0000BFFD // C61
data8 0xAB72D17DCD40CCE1,0x0000BFFF // C71
data8 0xA9D9AC9BE42C2EF9,0x0000C000 // C81
data8 0x8C11D983AA177AD2,0x0000C001 // C91
data8 0xDC779E981C1F0F06,0x0000C001 // CA1
data8 0xC1FD4AC85965E8D6,0x00003FFF // C00
data8 0xCE3D2D909D389EC2,0x00003FFF // C10
data8 0xE7F79980AD06F5D8,0x00003FFF // C20
data8 0x88DD9F73C8680B5D,0x00004000 // C30
data8 0xA7D6CB2CB2D46F9D,0x00004000 // C40
//[80; 81]
data8 0x91C7FF4E993430D0,0x00004000 // C01
data8 0x8A6E7AB83E45A7E9,0x00004000 // C11
data8 0xF72D6382E427BEA9,0x00003FFF // C21
data8 0xC9E2E4F9B3B23ED6,0x00003FFF // C31
data8 0x8BEFEF56AE05D775,0x00003FFF // C41
data8 0xEE9666AB6A185560,0x00003FFD // C51
data8 0xA6AFAF5CEFAEE04D,0x0000BFFD // C61
data8 0xA877EAFEF1F9C880,0x0000BFFF // C71
data8 0xA45BD433048ECA15,0x0000C000 // C81
data8 0x86BD1636B774CC2E,0x0000C001 // C91
data8 0xD3721BE006E10823,0x0000C001 // CA1
data8 0xA97EFABA91854208,0x00003FFF // C00
data8 0xB4AF0AEBB3F97737,0x00003FFF // C10
data8 0xCC38241936851B0B,0x00003FFF // C20
data8 0xF282A6261006EA84,0x00003FFF // C30
data8 0x95B8E9DB1BD45BAF,0x00004000 // C40
//[96; 97]
data8 0x8A1FA3171B35A106,0x00004000 // C01
data8 0x830D5B8843890F21,0x00004000 // C11
data8 0xE98B0F1616677A23,0x00003FFF // C21
data8 0xBDF8347F5F67D4EC,0x00003FFF // C31
data8 0x825F15DE34EC055D,0x00003FFF // C41
data8 0xD4846186B8AAC7BE,0x00003FFD // C51
data8 0xB161093AB14919B1,0x0000BFFD // C61
data8 0xA65758EEA4800EF4,0x0000BFFF // C71
data8 0xA046B67536FA329C,0x0000C000 // C81
data8 0x82BBEC1BCB9E9068,0x0000C001 // C91
data8 0xCC9DE2B23BA91B0B,0x0000C001 // CA1
data8 0x983B16148AF77F94,0x00003FFF // C00
data8 0xA2A4D8EE90FEE5DD,0x00003FFF // C10
data8 0xB89446FA37FF481C,0x00003FFF // C20
data8 0xDC5572648485FB01,0x00003FFF // C30
data8 0x88CD5D7DB976129A,0x00004000 // C40
//[112; 113]
data8 0x8417098FD62AC5E3,0x00004000 // C01
data8 0xFA7896486B779CBB,0x00003FFF // C11
data8 0xDEC98B14AF5EEBD1,0x00003FFF // C21
data8 0xB48E153C6BF0B5A3,0x00003FFF // C31
data8 0xF597B038BC957582,0x00003FFE // C41
data8 0xBFC6F0884A415694,0x00003FFD // C51
data8 0xBA075A1392BDB5E5,0x0000BFFD // C61
data8 0xA4B79E01B44C7DB4,0x0000BFFF // C71
data8 0x9D12FA7711BFAB0F,0x0000C000 // C81
data8 0xFF24C47C8E108AB4,0x0000C000 // C91
data8 0xC7325EC86562606A,0x0000C001 // CA1
data8 0x8B47DCD9E1610938,0x00003FFF // C00
data8 0x9518B111B70F88B8,0x00003FFF // C10
data8 0xA9CC197206F68682,0x00003FFF // C20
data8 0xCB98294CC0D7A6A6,0x00003FFF // C30
data8 0xFE09493EA9165181,0x00003FFF // C40
//[128; 129]
data8 0xFE53D03442270D90,0x00003FFF // C01
data8 0xF0F857BAEC1993E4,0x00003FFF // C11
data8 0xD5FF6D70DBBC2FD3,0x00003FFF // C21
data8 0xACDAA5F4988B1074,0x00003FFF // C31
data8 0xE92E069F8AD75B54,0x00003FFE // C41
data8 0xAEBB64645BD94234,0x00003FFD // C51
data8 0xC13746249F39B43C,0x0000BFFD // C61
data8 0xA36B74F5B6297A1F,0x0000BFFF // C71
data8 0x9A77860DF180F6E5,0x0000C000 // C81
data8 0xF9F8457D84410A0C,0x0000C000 // C91
data8 0xC2BF44C649EB8597,0x0000C001 // CA1
data8 0x81225E7489BCDC0E,0x00003FFF // C00
data8 0x8A788A09CE0EED11,0x00003FFF // C10
data8 0x9E2E6F86D1B1D89C,0x00003FFF // C20
data8 0xBE6866B21CF6CCB5,0x00003FFF // C30
data8 0xEE94426EC1486AAE,0x00003FFF // C40
//[144; 145]
data8 0xF6113E09732A6497,0x00003FFF // C01
data8 0xE900D45931B04FC8,0x00003FFF // C11
data8 0xCE9FD58F745EBA5D,0x00003FFF // C21
data8 0xA663A9636C864C86,0x00003FFF // C31
data8 0xDEBF5315896CE629,0x00003FFE // C41
data8 0xA05FEA415EBD7737,0x00003FFD // C51
data8 0xC750F112BD9C4031,0x0000BFFD // C61
data8 0xA2593A35C51C6F6C,0x0000BFFF // C71
data8 0x9848E1DA7FB40C8C,0x0000C000 // C81
data8 0xF59FEE87A5759A4B,0x0000C000 // C91
data8 0xBF00203909E45A1D,0x0000C001 // CA1
data8 0xF1D8E157200127E5,0x00003FFE // C00
data8 0x81DD5397CB08D487,0x00003FFF // C10
data8 0x94C1DC271A8B766F,0x00003FFF // C20
data8 0xB3AFAF9B5D6EDDCF,0x00003FFF // C30
data8 0xE1FB4C57CA81BE1E,0x00003FFF // C40
//[160; 161]
data8 0xEEFFE5122AC72FFD,0x00003FFF // C01
data8 0xE22F70BB52AD54B3,0x00003FFF // C11
data8 0xC84FF021FE993EEA,0x00003FFF // C21
data8 0xA0DA2208EB5B2752,0x00003FFF // C31
data8 0xD5CDD2FCF8AD2DF5,0x00003FFE // C41
data8 0x940BEC6DCD811A59,0x00003FFD // C51
data8 0xCC954EF4FD4EBB81,0x0000BFFD // C61
data8 0xA1712E29A8C04554,0x0000BFFF // C71
data8 0x966B55DFB243521A,0x0000C000 // C81
data8 0xF1E6A2B9CEDD0C4C,0x0000C000 // C91
data8 0xBBC87BCC031012DB,0x0000C001 // CA1
data8 0xE43974E6D2818583,0x00003FFE // C00
data8 0xF5702A516B64C5B7,0x00003FFE // C10
data8 0x8CEBCB1B32E19471,0x00003FFF // C20
data8 0xAAC10F05BB77E0AF,0x00003FFF // C30
data8 0xD776EFCAB205CC58,0x00003FFF // C40
//[176; 177]
data8 0xE8DA614119811E5D,0x00003FFF // C01
data8 0xDC415E0288B223D8,0x00003FFF // C11
data8 0xC2D2243E44EC970E,0x00003FFF // C21
data8 0x9C086664B5307BEA,0x00003FFF // C31
data8 0xCE03D7A08B461156,0x00003FFE // C41
data8 0x894BE3BAAAB66ADC,0x00003FFD // C51
data8 0xD131EDD71A702D4D,0x0000BFFD // C61
data8 0xA0A907CDDBE10898,0x0000BFFF // C71
data8 0x94CC3CD9C765C808,0x0000C000 // C81
data8 0xEEA85F237815FC0D,0x0000C000 // C91
data8 0xB8FA04B023E43F91,0x0000C001 // CA1
data8 0xD8B2C7D9FCBD7EF9,0x00003FFE // C00
data8 0xE9566E93AAE7E38F,0x00003FFE // C10
data8 0x8646E78AABEF0255,0x00003FFF // C20
data8 0xA32AEDB62E304345,0x00003FFF // C30
data8 0xCE83E40280EE7DF0,0x00003FFF // C40
//
//[2; 3]
data8 0xC44FB47E90584083,0x00004001 // C50
data8 0xE863EE77E1C45981,0x00004001 // C60
data8 0x8AC15BE238B9D70E,0x00004002 // C70
data8 0xA5D94B6592350EF4,0x00004002 // C80
data8 0xC379DB3E20A148B3,0x00004002 // C90
data8 0xDACA49B73974F6C9,0x00004002 // CA0
data8 0x810E496A1AFEC895,0x00003FE1 // An
//[16; 17]
data8 0xE17C0357AAF3F817,0x00004001 // C50
data8 0x8BA8804750FBFBFE,0x00004002 // C60
data8 0xB18EAB3CB64BEBEE,0x00004002 // C70
data8 0xE90AB7015AF1C28F,0x00004002 // C80
data8 0xA0AB97CE9E259196,0x00004003 // C90
data8 0xF5E0E0A000C2D720,0x00004003 // CA0
data8 0xD97F0F87EC791954,0x00004005 // An
//[32; 33]
data8 0x980C293F3696040D,0x00004001 // C50
data8 0xC0DBFFBB948A9A4E,0x00004001 // C60
data8 0xFAB54625E9A588A2,0x00004001 // C70
data8 0xA7E08176D6050FBF,0x00004002 // C80
data8 0xEBAAEC4952270A9F,0x00004002 // C90
data8 0xB7479CDAD20550FE,0x00004003 // CA0
data8 0xAACD45931C3FF634,0x00004054 // An
//[48; 49]
data8 0xF5180F0000419AD5,0x00004000 // C50
data8 0x9D507D07BFBB2273,0x00004001 // C60
data8 0xCEB53F7A13A383E3,0x00004001 // C70
data8 0x8BAFEF9E0A49128F,0x00004002 // C80
data8 0xC58EF912D39E228C,0x00004002 // C90
data8 0x9A88118422BA208E,0x00004003 // CA0
data8 0xBD6C0E2477EC12CB,0x000040AC // An
//[64; 65]
data8 0xD410AC48BF7748DA,0x00004000 // C50
data8 0x89399B90AFEBD931,0x00004001 // C60
data8 0xB596DF8F77EB8560,0x00004001 // C70
data8 0xF6D9445A047FB4A6,0x00004001 // C80
data8 0xAF52F0DD65221357,0x00004002 // C90
data8 0x8989B45BFC881989,0x00004003 // CA0
data8 0xB7FCAE86E6E10D5A,0x0000410B // An
//[80; 81]
data8 0xBE759740E3B5AA84,0x00004000 // C50
data8 0xF8037B1B07D27609,0x00004000 // C60
data8 0xA4F6F6C7F0977D4F,0x00004001 // C70
data8 0xE131960233BF02C4,0x00004001 // C80
data8 0xA06DF43D3922BBE2,0x00004002 // C90
data8 0xFC266AB27255A360,0x00004002 // CA0
data8 0xD9F4B012EDAFEF2F,0x0000416F // An
//[96; 97]
data8 0xAEFC84CDA8E1EAA6,0x00004000 // C50
data8 0xE5009110DB5F3C8A,0x00004000 // C60
data8 0x98F5F48738E7B232,0x00004001 // C70
data8 0xD17EE64E21FFDC6B,0x00004001 // C80
data8 0x9596F7A7E36145CC,0x00004002 // C90
data8 0xEB64DBE50E125CAF,0x00004002 // CA0
data8 0xA090530D79E32D2E,0x000041D8 // An
//[112; 113]
data8 0xA33AEA22A16B2655,0x00004000 // C50
data8 0xD682B93BD7D7945C,0x00004000 // C60
data8 0x8FC854C6E6E30CC3,0x00004001 // C70
data8 0xC5754D828AFFDC7A,0x00004001 // C80
data8 0x8D41216B397139C2,0x00004002 // C90
data8 0xDE78D746848116E5,0x00004002 // CA0
data8 0xB8A297A2DC0630DB,0x00004244 // An
//[128; 129]
data8 0x99EB00F11D95E292,0x00004000 // C50
data8 0xCB005CB911EB779A,0x00004000 // C60
data8 0x8879AA2FDFF3A37A,0x00004001 // C70
data8 0xBBDA538AD40CAC2C,0x00004001 // C80
data8 0x8696D849D311B9DE,0x00004002 // C90
data8 0xD41E1C041481199F,0x00004002 // CA0
data8 0xEBA1A43D34EE61EE,0x000042B3 // An
//[144; 145]
data8 0x924F822578AA9F3D,0x00004000 // C50
data8 0xC193FAF9D3B36960,0x00004000 // C60
data8 0x827AE3A6B68ED0CA,0x00004001 // C70
data8 0xB3F52A27EED23F0B,0x00004001 // C80
data8 0x811A079FB3C94D79,0x00004002 // C90
data8 0xCB94415470B6F8D2,0x00004002 // CA0
data8 0x80A0260DCB3EC9AC,0x00004326 // An
//[160; 161]
data8 0x8BF24091E88B331D,0x00004000 // C50
data8 0xB9ADE01187E65201,0x00004000 // C60
data8 0xFAE4508F6E7625FE,0x00004000 // C70
data8 0xAD516668AD6D7367,0x00004001 // C80
data8 0xF8F5FF171154F637,0x00004001 // C90
data8 0xC461321268990C82,0x00004002 // CA0
data8 0xC3B693F344B0E6FE,0x0000439A // An
//
//[176; 177]
data8 0x868545EB42A258ED,0x00004000 // C50
data8 0xB2EF04ACE8BA0E6E,0x00004000 // C60
data8 0xF247D22C22E69230,0x00004000 // C70
data8 0xA7A1AB93E3981A90,0x00004001 // C80
data8 0xF10951733E2C697F,0x00004001 // C90
data8 0xBE3359BFAD128322,0x00004002 // CA0
data8 0x8000000000000000,0x00003fff
//
//[160; 161] for negatives
data8 0xA76DBD55B2E32D71,0x00003C63 // 1/An
//
// sin(pi*x)/pi
data8 0xBCBC4342112F52A2,0x00003FDE // S21
data8 0xFAFCECB86536F655,0x0000BFE3 // S19
data8 0x87E4C97F9CF09B92,0x00003FE9 // S17
data8 0xEA124C68E704C5CB,0x0000BFED // S15
data8 0x9BA38CFD59C8AA1D,0x00003FF2 // S13
data8 0x99C0B552303D5B21,0x0000BFF6 // S11
//
//[176; 177] for negatives
data8 0xBA5D5869211696FF,0x00003BEC // 1/An
//
// sin(pi*x)/pi
data8 0xD63402E79A853175,0x00003FF9 // S9
data8 0xC354723906DB36BA,0x0000BFFC // S7
data8 0xCFCE5A015E236291,0x00003FFE // S5
data8 0xD28D3312983E9918,0x0000BFFF // S3
//
//
// [1.0;1.25]
data8 0xA405530B067ECD3C,0x0000BFFC // A15
data8 0xF5B5413F95E1C282,0x00003FFD // A14
data8 0xC4DED71C782F76C8,0x0000BFFE // A13
data8 0xECF7DDDFD27C9223,0x00003FFE // A12
data8 0xFB73D31793068463,0x0000BFFE // A11
data8 0xFF173B7E66FD1D61,0x00003FFE // A10
data8 0xFFA5EF3959089E94,0x0000BFFE // A9
data8 0xFF8153BD42E71A4F,0x00003FFE // A8
data8 0xFEF9CAEE2CB5B533,0x0000BFFE // A7
data8 0xFE3F02E5EDB6811E,0x00003FFE // A6
data8 0xFB64074CED2658FB,0x0000BFFE // A5
data8 0xFB52882A095B18A4,0x00003FFE // A4
data8 0xE8508C7990A0DAC0,0x0000BFFE // A3
data8 0xFD32C611D8A881D0,0x00003FFE // A2
data8 0x93C467E37DB0C536,0x0000BFFE // A1
data8 0x8000000000000000,0x00003FFF // A0
//
// [1.25;1.5]
data8 0xD038092400619677,0x0000BFF7 // A15
data8 0xEA6DE925E6EB8C8F,0x00003FF3 // A14
data8 0xC53F83645D4597FC,0x0000BFF7 // A13
data8 0xE366DB2FB27B7ECD,0x00003FF7 // A12
data8 0xAC8FD5E11F6EEAD8,0x0000BFF8 // A11
data8 0xFB14010FB3697785,0x00003FF8 // A10
data8 0xB6F91CB5C371177B,0x0000BFF9 // A9
data8 0x85A262C6F8FEEF71,0x00003FFA // A8
data8 0xC038E6E3261568F9,0x0000BFFA // A7
data8 0x8F4BDE8883232364,0x00003FFB // A6
data8 0xBCFBBD5786537E9A,0x0000BFFB // A5
data8 0xA4C08BAF0A559479,0x00003FFC // A4
data8 0x85D74FA063E81476,0x0000BFFC // A3
data8 0xDB629FB9BBDC1C4E,0x00003FFD // A2
data8 0xF4F8FBC7C0C9D317,0x00003FC6 // A1
data8 0xE2B6E4153A57746C,0x00003FFE // A0
//
// [1.25;1.5]
data8 0x9533F9D3723B448C,0x0000BFF2 // A15
data8 0xF1F75D3C561CBBAF,0x00003FF5 // A14
data8 0xBA55A9A1FC883523,0x0000BFF8 // A13
data8 0xB5D5E9E5104FA995,0x00003FFA // A12
data8 0xFD84F35B70CD9AE2,0x0000BFFB // A11
data8 0x87445235F4688CC5,0x00003FFD // A10
data8 0xE7F236EBFB9F774E,0x0000BFFD // A9
data8 0xA6605F2721F787CE,0x00003FFE // A8
data8 0xCF579312AD7EAD72,0x0000BFFE // A7
data8 0xE96254A2407A5EAC,0x00003FFE // A6
data8 0xF41312A8572ED346,0x0000BFFE // A5
data8 0xF9535027C1B1F795,0x00003FFE // A4
data8 0xE7E82D0C613A8DE4,0x0000BFFE // A3
data8 0xFD23CD9741B460B8,0x00003FFE // A2
data8 0x93C30FD9781DBA88,0x0000BFFE // A1
data8 0xFFFFF1781FDBEE84,0x00003FFE // A0
LOCAL_OBJECT_END(tgamma_data)


//==============================================================
// Code
//==============================================================

.section .text
GLOBAL_LIBM_ENTRY(tgamma)
{ .mfi
      getf.exp      GR_Sign_Exp = f8
      fma.s1        FR_1m2X = f8,f1,f8 // 2x
      addl          GR_ad_Data = @ltoff(tgamma_data), gp
}
{ .mfi
      mov           GR_ExpOf8 = 0x10002 // 8
      fcvt.fx.trunc.s1 FR_iXt = f8 // [x]
      mov           GR_ExpOf05 = 0xFFFE // 0.5
};;
{ .mfi
      getf.sig      GR_Sig = f8
      fma.s1        FR_2 = f1,f1,f1 // 2
      mov           GR_Addr_Mask1 = 0x780
}
{ .mlx
      setf.exp      FR_8 = GR_ExpOf8
      movl          GR_10 = 0x4024000000000000
};;
{ .mfi
      ld8           GR_ad_Data = [GR_ad_Data]
      fcmp.lt.s1    p14,p15 = f8,f0
      tbit.z        p12,p13 = GR_Sign_Exp,0x10 // p13 if x >= 2
}
{ .mlx
      and           GR_Bit2 = 4,GR_Sign_Exp
      movl          GR_12 = 0x4028000000000000
};;
{ .mfi
      setf.d        FR_10 = GR_10
      fma.s1        FR_r02 = f8,f1,f0
      extr.u        GR_Tbl_Offs = GR_Sig,58,6
}
{ .mfi
(p12) mov           GR_Addr_Mask1 = r0
      fma.s1        FR_NormX = f8,f1,f0
      cmp.ne        p8,p0 = GR_Bit2,r0
};;
{ .mfi
(p8)  shladd        GR_Tbl_Offs = GR_Tbl_Offs,4,r0
      fclass.m      p10,p0 =  f8,0x1E7 // Test x for NaTVal, NaN, +/-0, +/-INF
      tbit.nz       p11,p0 = GR_Sign_Exp,1
}
{ .mlx
      add           GR_Addr_Mask2 = GR_Addr_Mask1,GR_Addr_Mask1
      movl          GR_14 = 0x402C000000000000
};;
.pred.rel "mutex",p14,p15
{ .mfi
      setf.d        FR_12 = GR_12
(p14) fma.s1        FR_1m2X = f1,f1,FR_1m2X // RB=1-2|x|
      tbit.nz       p8,p9 = GR_Sign_Exp,0
}
{ .mfi
      ldfpd         FR_OvfBound,FR_Xmin = [GR_ad_Data],16
(p15) fms.s1        FR_1m2X = f1,f1,FR_1m2X // RB=1-2|x|
(p11) shladd        GR_Tbl_Offs = GR_Tbl_Offs,2,r0
};;
.pred.rel "mutex",p9,p8
{ .mfi
      setf.d        FR_14 = GR_14
      fma.s1        FR_4 = FR_2,FR_2,f0
(p8)  and           GR_Tbl_Offs = GR_Tbl_Offs, GR_Addr_Mask1
}
{ .mfi
      setf.exp      FR_05 = GR_ExpOf05
      fma.s1        FR_6 = FR_2,FR_2,FR_2
(p9)  and           GR_Tbl_Offs = GR_Tbl_Offs, GR_Addr_Mask2
};;
.pred.rel "mutex",p9,p8
{ .mfi
(p8)  shladd        GR_ad_Co = GR_Tbl_Offs,1,GR_ad_Data
      fcvt.xf       FR_Xt = FR_iXt // [x]
(p15) tbit.z.unc    p11,p0 = GR_Sign_Exp,0x10 // p11 if 0 < x < 2
}
{ .mfi
(p9)  add           GR_ad_Co = GR_ad_Data,GR_Tbl_Offs
      fma.s1        FR_5 = FR_2,FR_2,f1
(p15) cmp.lt.unc    p7,p6 = GR_ExpOf05,GR_Sign_Exp // p7 if 0 < x < 1
};;
{ .mfi
      add           GR_ad_Ce = 16,GR_ad_Co
(p11) frcpa.s1      FR_Rcp0,p0 = f1,f8
      sub           GR_Tbl_Offs = GR_ad_Co,GR_ad_Data
}
{ .mfb
      ldfe          FR_C01 = [GR_ad_Co],32
(p7)  fms.s1        FR_r02 = FR_r02,f1,f1
      // jump if x is NaTVal, NaN, +/-0, +/-INF
(p10) br.cond.spnt  tgamma_spec
};;
.pred.rel "mutex",p14,p15
{ .mfi
      ldfe          FR_C11 = [GR_ad_Ce],32
(p14) fms.s1        FR_X2pX = f8,f8,f8 // RA=x^2+|x|
      shr           GR_Tbl_Ind = GR_Tbl_Offs,8
}
{ .mfb
      ldfe          FR_C21 = [GR_ad_Co],32
(p15) fma.s1        FR_X2pX = f8,f8,f8 // RA=x^2+x
      // jump if 0 < x < 2
(p11) br.cond.spnt  tgamma_from_0_to_2
};;
{ .mfi
      ldfe          FR_C31 = [GR_ad_Ce],32
      fma.s1        FR_Rq2 = FR_2,f1,FR_1m2X  // 2 + B
      cmp.ltu       p7,p0=0xB,GR_Tbl_Ind
}
{ .mfb
      ldfe          FR_C41 = [GR_ad_Co],32
      fma.s1        FR_Rq3 = FR_2,FR_2,FR_1m2X  // 4 + B
      // jump if GR_Tbl_Ind > 11, i.e |x| is more than 192
(p7)  br.cond.spnt  tgamma_spec_res
};;
{ .mfi
      ldfe          FR_C51 = [GR_ad_Ce],32
      fma.s1        FR_Rq4 = FR_6,f1,FR_1m2X  // 6 + B
      shr           GR_Tbl_Offs = GR_Tbl_Offs,1
}
{ .mfi
      ldfe          FR_C61 = [GR_ad_Co],32
      fma.s1        FR_Rq5 = FR_4,FR_2,FR_1m2X // 8 + B
      nop.i         0
};;
{ .mfi
      ldfe          FR_C71 = [GR_ad_Ce],32
(p14) fms.s1        FR_r = FR_Xt,f1,f8 // r = |x| - [|x|]
      shr           GR_Tbl_16xInd = GR_Tbl_Offs,3
}
{ .mfi
      ldfe          FR_C81 = [GR_ad_Co],32
(p15) fms.s1        FR_r = f8,f1,FR_Xt // r = x - [x]
      add           GR_ad_Data = 0xC00,GR_ad_Data
};;
{ .mfi
      ldfe          FR_C91 = [GR_ad_Ce],32
      fma.s1        FR_Rq6 = FR_5,FR_2,FR_1m2X  // 10 + B
(p14) mov           GR_0x30033 = 0x30033
}
{ .mfi
      ldfe          FR_CA1 = [GR_ad_Co],32
      fma.s1        FR_Rq7 = FR_6,FR_2,FR_1m2X // 12 + B
      sub           GR_Tbl_Offs = GR_Tbl_Offs,GR_Tbl_16xInd
};;
{ .mfi
      ldfe          FR_C00 = [GR_ad_Ce],32
      fma.s1        FR_Rq1 = FR_Rq1,FR_2,FR_X2pX // (x-1)*(x-2)
(p13) cmp.eq.unc    p8,p0 = r0,GR_Tbl_16xInd // index is 0 i.e. arg from [2;16)
}
{ .mfi
      ldfe          FR_C10 = [GR_ad_Co],32
(p14) fms.s1        FR_AbsX = f0,f0,FR_NormX // absolute value of argument
      add           GR_ad_Co7 = GR_ad_Data,GR_Tbl_Offs
};;
{ .mfi
      ldfe          FR_C20 = [GR_ad_Ce],32
      fma.s1        FR_Rq2 = FR_Rq2,FR_4,FR_X2pX // (x-3)*(x-4)
      add           GR_ad_Ce7 = 16,GR_ad_Co7
}
{ .mfi
      ldfe          FR_C30 = [GR_ad_Co],32
      fma.s1        FR_Rq3 = FR_Rq3,FR_6,FR_X2pX // (x-5)*(x-6)
      nop.i         0
};;
{ .mfi
      ldfe          FR_C40 = [GR_ad_Ce],32
      fma.s1        FR_Rq4 = FR_Rq4,FR_8,FR_X2pX // (x-7)*(x-8)
(p14) cmp.leu.unc   p7,p0 = GR_0x30033,GR_Sign_Exp
}
{ .mfb
      ldfe          FR_C50 = [GR_ad_Co7],32
      fma.s1        FR_Rq5 = FR_Rq5,FR_10,FR_X2pX // (x-9)*(x-10)
      // jump if x is less or equal to -2^52, i.e. x is big negative integer
(p7)  br.cond.spnt  tgamma_singularity
};;
{ .mfi
      ldfe          FR_C60 = [GR_ad_Ce7],32
      fma.s1        FR_C01 = FR_C01,f1,FR_r
      add           GR_ad_Ce = 0x560,GR_ad_Data
}
{ .mfi
      ldfe          FR_C70 = [GR_ad_Co7],32
      fma.s1        FR_rs = f0,f0,FR_r // reduced arg for sin(pi*x)
      add           GR_ad_Co = 0x550,GR_ad_Data
};;
{ .mfi
      ldfe          FR_C80 = [GR_ad_Ce7],32
      fma.s1        FR_C11 = FR_C11,f1,FR_r
      nop.i         0
}
{ .mfi
      ldfe          FR_C90 = [GR_ad_Co7],32
      fma.s1        FR_C21 = FR_C21,f1,FR_r
      nop.i         0
};;
.pred.rel "mutex",p12,p13
{ .mfi
(p13) getf.sig      GR_iSig = FR_iXt
      fcmp.lt.s1    p11,p0 = FR_05,FR_r
      mov           GR_185 = 185
}
{ .mfi
      nop.m         0
      fma.s1        FR_Rq6 = FR_Rq6,FR_12,FR_X2pX // (x-11)*(x-12)
      nop.i         0
};;
{ .mfi
      ldfe          FR_CA0 = [GR_ad_Ce7],32
      fma.s1        FR_C31 = FR_C31,f1,FR_r
(p12) mov           GR_iSig = 0
}
{ .mfi
      ldfe          FR_An = [GR_ad_Co7],0x80
      fma.s1        FR_C41 = FR_C41,f1,FR_r
      nop.i         0
};;
{ .mfi
(p14) getf.sig      GR_Sig = FR_r
      fma.s1        FR_C51 = FR_C51,f1,FR_r
(p14) sub           GR_iSig = r0,GR_iSig
}
{ .mfi
      ldfe          FR_S21 = [GR_ad_Co],32
      fma.s1        FR_C61 = FR_C61,f1,FR_r
      nop.i         0
};;
{ .mfi
      ldfe          FR_S19 = [GR_ad_Ce],32
      fma.s1        FR_C71 = FR_C71,f1,FR_r
      and           GR_SigRqLin = 0xF,GR_iSig
}
{ .mfi
      ldfe          FR_S17 = [GR_ad_Co],32
      fma.s1        FR_C81 = FR_C81,f1,FR_r
      mov           GR_2 = 2
};;
{ .mfi
(p14) ldfe          FR_InvAn = [GR_ad_Co7]
      fma.s1        FR_C91 = FR_C91,f1,FR_r
      // if significand of r is 0 tnan argument is negative integer
(p14) cmp.eq.unc    p12,p0 = r0,GR_Sig
}
{ .mfb
(p8)  sub           GR_SigRqLin = GR_SigRqLin,GR_2 // subtract 2 if 2 <= x < 16
      fma.s1        FR_CA1 = FR_CA1,f1,FR_r
      // jump if x is negative integer such that -2^52 < x < -185
(p12) br.cond.spnt  tgamma_singularity
};;
{ .mfi
      setf.sig      FR_Xt = GR_SigRqLin
(p11) fms.s1        FR_rs = f1,f1,FR_r
(p14) cmp.ltu.unc   p7,p0 = GR_185,GR_iSig
}
{ .mfb
      ldfe          FR_S15 = [GR_ad_Ce],32
      fma.s1        FR_Rq7 = FR_Rq7,FR_14,FR_X2pX // (x-13)*(x-14)
      // jump if x is noninteger such that -2^52 < x < -185
(p7)  br.cond.spnt  tgamma_underflow
};;
{ .mfi
      ldfe          FR_S13 = [GR_ad_Co],48
      fma.s1        FR_C01 = FR_C01,FR_r,FR_C00
      and           GR_Sig2 = 0xE,GR_SigRqLin
}
{ .mfi
      ldfe          FR_S11 = [GR_ad_Ce],48
      fma.s1        FR_C11 = FR_C11,FR_r,FR_C10
      nop.i         0
};;
{ .mfi
      ldfe          FR_S9 = [GR_ad_Co],32
      fma.s1        FR_C21 = FR_C21,FR_r,FR_C20
      // should we mul by polynomial of recursion?
      cmp.eq        p13,p12 = r0,GR_SigRqLin
}
{ .mfi
      ldfe          FR_S7 = [GR_ad_Ce],32
      fma.s1        FR_C31 = FR_C31,FR_r,FR_C30
      nop.i         0
};;
{ .mfi
      ldfe          FR_S5 = [GR_ad_Co],32
      fma.s1        FR_C41 = FR_C41,FR_r,FR_C40
      nop.i         0
}
{ .mfi
      ldfe          FR_S3 = [GR_ad_Ce],32
      fma.s1        FR_C51 = FR_C51,FR_r,FR_C50
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C61 = FR_C61,FR_r,FR_C60
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C71 = FR_C71,FR_r,FR_C70
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C81 = FR_C81,FR_r,FR_C80
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C91 = FR_C91,FR_r,FR_C90
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_CA1 = FR_CA1,FR_r,FR_CA0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C01 = FR_C01,FR_C11,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_C31,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_rs2 = FR_rs,FR_rs,f0
(p12) cmp.lt.unc    p7,p0 = 2,GR_Sig2 // should mul by FR_Rq2?
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C41 = FR_C41,FR_C51,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_Rq1 = FR_Rq1,FR_Rq2,f0
(p12) cmp.lt.unc    p9,p0 = 6,GR_Sig2 // should mul by FR_Rq4?
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C61 = FR_C61,FR_C71,f0
(p15) cmp.eq        p11,p0 = r0,r0
}
{ .mfi
      nop.m         0
(p9)  fma.s1        FR_Rq3 = FR_Rq3,FR_Rq4,f0
(p12) cmp.lt.unc    p8,p0 = 10,GR_Sig2 // should mul by FR_Rq6?
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C81 = FR_C81,FR_C91,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p8)  fma.s1        FR_Rq5 = FR_Rq5,FR_Rq6,f0
(p14) cmp.ltu       p0,p11 = 0x9,GR_Tbl_Ind
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_RqLin = FR_Xt
      nop.i         0
}
{ .mfi
      nop.m         0
(p11) fma.s1        FR_CA1 = FR_CA1,FR_An,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S21 = FR_S21,FR_rs2,FR_S19
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S17 = FR_S17,FR_rs2,FR_S15
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C01 = FR_C01,FR_C21,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_rs4 = FR_rs2,FR_rs2,f0
(p12) cmp.lt.unc    p8,p0 = 4,GR_Sig2 // should mul by FR_Rq3?
};;
{ .mfi
      nop.m         0
(p8)  fma.s1        FR_Rq1 = FR_Rq1,FR_Rq3,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S13 = FR_S13,FR_rs2,FR_S11
(p12) cmp.lt.unc    p9,p0 = 12,GR_Sig2 // should mul by FR_Rq7?
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C41 = FR_C41,FR_C61,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p9)  fma.s1        FR_Rq5 = FR_Rq5,FR_Rq7,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C81 = FR_C81,FR_CA1,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S9 = FR_S9,FR_rs2,FR_S7
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S5 = FR_S5,FR_rs2,FR_S3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_rs3 = FR_rs2,FR_rs,f0
(p12) tbit.nz.unc   p6,p0 = GR_SigRqLin,0
}
{ .mfi
      nop.m         0
      fma.s1        FR_rs8 = FR_rs4,FR_rs4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S21 = FR_S21,FR_rs4,FR_S17
      mov           GR_ExpOf1 = 0x2FFFF
}
{ .mfi
      nop.m         0
(p6)  fms.s1        FR_RqLin = FR_AbsX,f1,FR_RqLin
(p12) cmp.lt.unc    p8,p0 = 8,GR_Sig2 // should mul by FR_Rq5?
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C01 = FR_C01,FR_C41,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p8)  fma.s1        FR_Rq1 = FR_Rq1,FR_Rq5,f0
(p14) cmp.gtu.unc   p7,p0 = GR_Sign_Exp,GR_ExpOf1
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S13 = FR_S13,FR_rs4,FR_S9
      nop.i         0
}
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_C81 = FR_C81,FR_AbsX,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_AbsXp1 = f1,f1,FR_AbsX // |x|+1
      nop.i         0
}
{ .mfi
      nop.m         0
(p15) fcmp.lt.unc.s1 p0,p10 = FR_AbsX,FR_OvfBound // x >= overflow_boundary
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_rs7 = FR_rs4,FR_rs3,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S5 = FR_S5,FR_rs3,FR_rs
      nop.i         0
};;
{ .mib
(p14) cmp.lt        p13,p0 = r0,r0 // set p13 to 0 if x < 0
(p12) cmp.eq.unc    p8,p9 = 1,GR_SigRqLin
(p10) br.cond.spnt  tgamma_spec_res
};;
{ .mfi
      getf.sig      GR_Sig = FR_iXt
(p6)  fma.s1        FR_Rq1 = FR_Rq1,FR_RqLin,f0
      // should we mul by polynomial of recursion?
(p15) cmp.eq.unc    p0,p11 = r0,GR_SigRqLin
}
{ .mfb
      nop.m         0
      fma.s1        FR_GAMMA = FR_C01,FR_C81,f0
(p11) br.cond.spnt  tgamma_positives
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S21 = FR_S21,FR_rs8,FR_S13
      nop.i         0
}
{ .mfb
      nop.m         0
(p13) fma.d.s0      f8 = FR_C01,FR_C81,f0
(p13) br.ret.spnt   b0
};;
.pred.rel "mutex",p8,p9
{ .mfi
      nop.m         0
(p9)  fma.s1        FR_GAMMA = FR_GAMMA,FR_Rq1,f0
      tbit.z        p6,p7 = GR_Sig,0 // p6 if sin<0, p7 if sin>0
}
{ .mfi
      nop.m         0
(p8)  fma.s1        FR_GAMMA = FR_GAMMA,FR_RqLin,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S21 = FR_S21,FR_rs7,FR_S5
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fnma.s1       FR_GAMMA = FR_GAMMA,FR_S21,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_GAMMA = FR_GAMMA,FR_S21,f0
      mov           GR_Sig2 = 1
};;
{ .mfi
      nop.m         0
      frcpa.s1      FR_Rcp0,p0 = f1,FR_GAMMA
      cmp.ltu       p13,p0 = GR_Sign_Exp,GR_ExpOf1
};;
// NR method: ineration #1
{ .mfi
(p13) getf.exp      GR_Sign_Exp = FR_AbsX
      fnma.s1       FR_Rcp1 = FR_Rcp0,FR_GAMMA,f1 // t = 1 - r0*x
(p13) shl           GR_Sig2 = GR_Sig2,63
};;
{ .mfi
(p13) getf.sig      GR_Sig = FR_AbsX
      nop.f         0
(p13) mov           GR_NzOvfBound = 0xFBFF
};;
{ .mfi
(p13) cmp.ltu.unc   p8,p0 = GR_Sign_Exp,GR_NzOvfBound // p8 <- overflow
      nop.f         0
(p13) cmp.eq.unc    p9,p0 = GR_Sign_Exp,GR_NzOvfBound
};;
{ .mfb
      nop.m         0
(p13) fma.d.s0      FR_X = f1,f1,f8 // set deno & inexact flags
(p8)  br.cond.spnt  tgamma_ovf_near_0 //tgamma_neg_overflow
};;
{ .mib
      nop.m         0
(p9)  cmp.eq.unc    p8,p0 = GR_Sig,GR_Sig2
(p8)  br.cond.spnt  tgamma_ovf_near_0_boundary //tgamma_neg_overflow
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Rcp1 = FR_Rcp0,FR_Rcp1,FR_Rcp0
      nop.i         0
};;
// NR method: ineration #2
{ .mfi
      nop.m         0
      fnma.s1       FR_Rcp2 = FR_Rcp1,FR_GAMMA,f1 // t = 1 - r1*x
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Rcp2 = FR_Rcp1,FR_Rcp2,FR_Rcp1
      nop.i         0
};;
// NR method: ineration #3
{ .mfi
      nop.m         0
      fnma.s1       FR_Rcp3 = FR_Rcp2,FR_GAMMA,f1 // t = 1 - r2*x
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_Rcp2 = FR_Rcp2,FR_AbsXp1,f0
(p14) cmp.ltu       p10,p11 = 0x9,GR_Tbl_Ind
};;
.pred.rel "mutex",p10,p11
{ .mfi
      nop.m         0
(p10) fma.s1        FR_GAMMA = FR_Rcp2,FR_Rcp3,FR_Rcp2
      nop.i         0
}
{ .mfi
      nop.m         0
(p11) fma.d.s0      f8 = FR_Rcp2,FR_Rcp3,FR_Rcp2
      nop.i         0
};;
{ .mfb
      nop.m         0
(p10) fma.d.s0      f8 = FR_GAMMA,FR_InvAn,f0
      br.ret.sptk   b0
};;


// here if x >= 3
//--------------------------------------------------------------------
.align 32
tgamma_positives:
.pred.rel "mutex",p8,p9
{ .mfi
      nop.m         0
(p9)  fma.d.s0      f8 = FR_GAMMA,FR_Rq1,f0
      nop.i         0
}
{ .mfb
      nop.m         0
(p8)  fma.d.s0      f8 = FR_GAMMA,FR_RqLin,f0
      br.ret.sptk   b0
};;

// here if 0 < x < 1
//--------------------------------------------------------------------
.align 32
tgamma_from_0_to_2:
{ .mfi
      getf.exp      GR_Sign_Exp = FR_r02
      fms.s1        FR_r = FR_r02,f1,FR_Xmin
      mov           GR_ExpOf025 = 0xFFFD
}
{ .mfi
      add           GR_ad_Co = 0x1200,GR_ad_Data
(p6)  fnma.s1       FR_Rcp1 = FR_Rcp0,FR_NormX,f1  // t = 1 - r0*x
(p6)  mov           GR_Sig2 = 1
};;
{ .mfi
(p6)  getf.sig      GR_Sig = FR_NormX
      nop.f         0
(p6)  shl           GR_Sig2 = GR_Sig2,63
}
{ .mfi
      add           GR_ad_Ce = 0x1210,GR_ad_Data
      nop.f         0
(p6)  mov           GR_NzOvfBound = 0xFBFF
};;
{ .mfi
      cmp.eq        p8,p0 = GR_Sign_Exp,GR_ExpOf05 // r02 >= 1/2
      nop.f         0
      cmp.eq        p9,p10 = GR_Sign_Exp,GR_ExpOf025 // r02 >= 1/4
}
{ .mfi
(p6)  cmp.ltu.unc   p11,p0 = GR_Sign_Exp,GR_NzOvfBound // p11 <- overflow
      nop.f         0
(p6)  cmp.eq.unc    p12,p0 = GR_Sign_Exp,GR_NzOvfBound
};;
.pred.rel "mutex",p8,p9
{ .mfi
(p8)  add           GR_ad_Co = 0x200,GR_ad_Co
(p6)  fma.d.s0      FR_X = f1,f1,f8 // set deno & inexact flags
(p9)  add           GR_ad_Co = 0x100,GR_ad_Co
}
{ .mib
(p8)  add           GR_ad_Ce = 0x200,GR_ad_Ce
(p9)  add           GR_ad_Ce = 0x100,GR_ad_Ce
(p11) br.cond.spnt  tgamma_ovf_near_0 //tgamma_spec_res
};;
{ .mfi
      ldfe          FR_A15 = [GR_ad_Co],32
      nop.f         0
(p12) cmp.eq.unc    p13,p0 = GR_Sig,GR_Sig2
}
{ .mfb
      ldfe          FR_A14 = [GR_ad_Ce],32
      nop.f         0
(p13) br.cond.spnt  tgamma_ovf_near_0_boundary //tgamma_spec_res
};;
{ .mfi
      ldfe          FR_A13 = [GR_ad_Co],32
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfe          FR_A12 = [GR_ad_Ce],32
      nop.f         0
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mfi
      ldfe          FR_A11 = [GR_ad_Co],32
(p10) fma.s1        FR_r2 = FR_r02,FR_r02,f0
      nop.i         0
}
{ .mfi
      ldfe          FR_A10 = [GR_ad_Ce],32
(p9)  fma.s1        FR_r2 = FR_r,FR_r,f0
      nop.i         0
};;
{ .mfi
      ldfe          FR_A9 = [GR_ad_Co],32
(p6)  fma.s1        FR_Rcp1 = FR_Rcp0,FR_Rcp1,FR_Rcp0
      nop.i         0
}
{ .mfi
      ldfe          FR_A8 = [GR_ad_Ce],32
(p10) fma.s1        FR_r = f0,f0,FR_r02
      nop.i         0
};;
{ .mfi
      ldfe          FR_A7 = [GR_ad_Co],32
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfe          FR_A6 = [GR_ad_Ce],32
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfe          FR_A5 = [GR_ad_Co],32
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfe          FR_A4 = [GR_ad_Ce],32
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfe          FR_A3 = [GR_ad_Co],32
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfe          FR_A2 = [GR_ad_Ce],32
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfe          FR_A1 = [GR_ad_Co],32
      fma.s1        FR_r4 = FR_r2,FR_r2,f0
      nop.i         0
}
{ .mfi
      ldfe          FR_A0 = [GR_ad_Ce],32
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p6)  fnma.s1       FR_Rcp2 = FR_Rcp1,FR_NormX,f1 // t = 1 - r1*x
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r,FR_A14
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A11 = FR_A11,FR_r,FR_A10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_r8 = FR_r4,FR_r4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_Rcp2 = FR_Rcp1,FR_Rcp2,FR_Rcp1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r,FR_A6
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_r,FR_A2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r,FR_A13
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A11 = FR_A11,FR_r,FR_A9
      nop.i         0
};;
{ .mfi
      nop.m         0
(p6)  fnma.s1       FR_Rcp3 = FR_Rcp2,FR_NormX,f1 // t = 1 - r1*x
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r,FR_A5
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_r,FR_A1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r,FR_A12
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A11 = FR_A11,FR_r,FR_A8
      nop.i         0
};;
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_Rcp3 = FR_Rcp2,FR_Rcp3,FR_Rcp2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r,FR_A4
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_r,FR_A0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r4,FR_A11
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r4,FR_A3
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_A15 = FR_A15,FR_r8,FR_A7
      nop.i         0
}
{ .mfi
      nop.m         0
(p7)  fma.d.s0      f8 = FR_A15,FR_r8,FR_A7
      nop.i         0
};;
{ .mfb
      nop.m         0
(p6)  fma.d.s0      f8 = FR_A15,FR_Rcp3,f0
      br.ret.sptk   b0
};;

// overflow
//--------------------------------------------------------------------
.align 32
tgamma_ovf_near_0_boundary:
.pred.rel "mutex",p14,p15
{ .mfi
	  mov           GR_fpsr = ar.fpsr
	  nop.f         0
(p15) mov           r8 = 0x7ff
}
{ .mfi
      nop.m         0
      nop.f         0
(p14) mov           r8 = 0xfff
};;
{ .mfi
	  nop.m         0
	  nop.f         0
	  shl           r8 = r8,52
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
(p8)  setf.d        f8 = r8 // set result to 0x7fefffffffffffff without
                            // OVERFLOW flag raising
      nop.i         0
(p8)  br.ret.sptk   b0
};;
.align 32
tgamma_ovf_near_0:
{ .mfi
      mov           r8 = 0x1FFFE
      nop.f         0
      nop.i         0
};;
{ .mfi
      setf.exp      f9 = r8
      fmerge.s      FR_X = f8,f8
      mov           GR_TAG = 258 // overflow
};;
.pred.rel "mutex",p14,p15
{ .mfi
      nop.m         0
(p15) fma.d.s0      f8 = f9,f9,f0 // Set I,O and +INF result
      nop.i         0
}
{ .mfb
      nop.m         0
(p14) fnma.d.s0     f8 = f9,f9,f0 // Set I,O and -INF result
      br.cond.sptk  tgamma_libm_err
};;
// overflow or absolute value of x is too big
//--------------------------------------------------------------------
.align 32
tgamma_spec_res:
{ .mfi
      mov           GR_0x30033 = 0x30033
(p14) fcmp.eq.unc.s1 p10,p11 = f8,FR_Xt
(p15) mov           r8 = 0x1FFFE
};;
{ .mfi
(p15) setf.exp      f9 = r8
      nop.f         0
      nop.i         0
};;
{ .mfb
(p11) cmp.ltu.unc   p7,p8 = GR_0x30033,GR_Sign_Exp
      nop.f         0
(p10) br.cond.spnt  tgamma_singularity
};;
.pred.rel "mutex",p7,p8
{ .mbb
      nop.m         0
(p7)  br.cond.spnt  tgamma_singularity
(p8)  br.cond.spnt  tgamma_underflow
};;
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8
      mov           GR_TAG = 258 // overflow
}
{ .mfb
      nop.m         0
(p15) fma.d.s0      f8 = f9,f9,f0 // Set I,O and +INF result
      br.cond.sptk  tgamma_libm_err
};;

// x is negative integer or +/-0
//--------------------------------------------------------------------
.align 32
tgamma_singularity:
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8
      mov           GR_TAG = 259 // negative
}
{ .mfb
      nop.m         0
      frcpa.s0      f8,p0 = f0,f0
      br.cond.sptk  tgamma_libm_err
};;
// x is negative noninteger with big absolute value
//--------------------------------------------------------------------
.align 32
tgamma_underflow:
{ .mmi
      getf.sig      GR_Sig = FR_iXt
      mov           r11 = 0x00001
      nop.i         0
};;
{ .mfi
      setf.exp      f9 = r11
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
      nop.f         0
      tbit.z        p6,p7 = GR_Sig,0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fms.d.s0      f8 = f9,f9,f9
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.d.s0      f8 = f9,f9,f9
      br.ret.sptk   b0
};;

//  x for natval, nan, +/-inf or +/-0
//--------------------------------------------------------------------
.align 32
tgamma_spec:
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
(p6)  fma.d.s0      f8 = f8,f1,f8
(p6)  br.ret.spnt   b0
};;
.pred.rel "mutex",p7,p8
{ .mfi
(p7)  mov           GR_TAG = 259 // negative
(p7)  frcpa.s0      f8,p0 = f1,f8
      nop.i         0
}
{ .mib
      nop.m         0
      nop.i         0
(p8)  br.cond.spnt  tgamma_singularity
};;

.align 32
tgamma_libm_err:
{ .mfi
       alloc        r32 = ar.pfs,1,4,4,0
       nop.f        0
       mov          GR_Parameter_TAG = GR_TAG
};;

GLOBAL_LIBM_END(tgamma)
libm_alias_double_other (tgamma, tgamma)

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
        stfd [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfd [GR_Parameter_X] = FR_X                  // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#         // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfd  f8 = [GR_Parameter_RESULT]       // Get return result off stack
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
