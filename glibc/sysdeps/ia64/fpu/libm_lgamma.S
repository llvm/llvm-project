.file "libm_lgamma.s"


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
// 01/10/02  Initial version
// 01/25/02  Corrected error tag numbers
// 02/04/02  Added support of SIGN(GAMMA(x)) calculation
// 05/20/02  Cleaned up namespace and sf0 syntax
// 09/15/02  Fixed bug on the branch lgamma_negrecursion
// 10/21/02  Now it returns SIGN(GAMMA(x))=-1 for negative zero
// 02/10/03  Reordered header: .section, .global, .proc, .align
// 07/22/03  Reformatted some data tables
// 03/31/05  Reformatted delimiters between data tables
//
//*********************************************************************
//
//*********************************************************************
//
// Function: __libm_lgamma(double x, int* signgam, int szsigngam)
// computes the principle value of the logarithm of the GAMMA function
// of x. Signum of GAMMA(x) is stored to memory starting at the address
// specified by the signgam.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f6-f15
//                              f32-f122
//
//    General Purpose Registers:
//      r8-r11
//      r14-r31
//      r32-r36
//      r37-r40 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    __libm_lgamma(+inf) = +inf
//    __libm_lgamma(-inf) = QNaN
//    __libm_lgamma(+/-0) = +inf
//    __libm_lgamma(x<0, x - integer) = +inf
//    __libm_lgamma(SNaN) = QNaN
//    __libm_lgamma(QNaN) = QNaN
//
//*********************************************************************
//
// Overview
//
// The method consists of three cases.
//
// If      512 <= x < OVERFLOW_BOUNDARY   use case lgamma_pstirling;
// else if 1 < x < 512                    use case lgamma_regular;
// else if -17 < x < 1                    use case lgamma_negrecursion;
// else if -512 <  x < -17                use case lgamma_negpoly;
// else if x < -512                       use case lgamma_negstirling;
// else if x is close to negative
//         roots of ln(GAMMA(x))          use case lgamma_negroots;
//
//
// Case 512 <= x < OVERFLOW_BOUNDARY
// ---------------------------------
//   Here we use algorithm based on the Stirling formula:
//   ln(GAMMA(x)) = ln(sqrt(2*Pi)) + (x-0.5)ln(x) - x + (W2 + W4/x^2)/x
//
// Case 1 < x < 512
// ----------------
//   To calculate GAMMA(x) on this interval we use polynomial approximation
//   on following intervals [0.875; 1.25), [1.25; 1.75), [1.75, 2.25),
//   [2.25; 4), [2^i; 2^(i+1)), i=2..8
//
//   Following variants of approximation and argument reduction are used:
//    1. [0.875; 1.25)
//       ln(GAMMA(x)) ~ (x-1.0)*P17(x-1.0)
//
//    2. [1.25; 1.75)
//       ln(GAMMA(x)) ~ (x-LocalMinimun)*P17(x-LocalMinimun)
//
//    3. [1.75, 2.25)
//       ln(GAMMA(x)) ~ (x-2.0)*P17(x-2.0)
//
//    4. [2.25; 4)
//       ln(GAMMA(x)) ~ P22(x)
//
//    5. [2^i; 2^(i+1)), i=2..8
//       ln(GAMMA(x)) ~ P22((x-2^i)/2^i)
//
// Case -17 < x < 1
// ----------------
//   Here we use the recursive formula:
//   ln(GAMMA(x)) = ln(GAMMA(x+1)) - ln(x)
//
//   Using this formula we reduce argument to base interval [1.0; 2.0]
//
// Case -512 <  x < -17
// --------------------
//   Here we use the formula:
//   ln(GAMMA(-x)) = ln(Pi/(x*GAMMA(x)*sin(Pi*x))) =
//   = -ln(x) - ln((GAMMA(x)) - ln(sin(Pi*r)/(Pi*r)) - ln(|r|)
//   where r = x - rounded_to_nearest(x), i.e |r| <= 0.5 and
//   ln(sin(Pi*r)/(Pi*r)) is approximated by 14-degree polynomial of r^2
//
//
// Case x < -512
// -------------
//   Here we use algorithm based on the Stirling formula:
//   ln(GAMMA(-x)) = -ln(sqrt(2*Pi)) + (-x-0.5)ln(x) + x - (W2 + W4/x^2)/x -
//   - ln(sin(Pi*r)/(Pi*r)) - ln(|r|)
//   where r = x - rounded_to_nearest(x).
//
// Neighbourhoods of negative roots
// --------------------------------
//   Here we use polynomial approximation
//   ln(GAMMA(x-x0)) = ln(GAMMA(x0)) + (x-x0)*P14(x-x0),
//   where x0 is a root of ln(GAMMA(x)) rounded to nearest double
//   precision number.
//

//*********************************************************************

FR_X                   = f10
FR_Y                   = f1 // __libm_lgamma is single argument function
FR_RESULT              = f8

FR_B11                 = f6
FR_B10                 = f7

FR_int_N               = f9
FR_N                   = f10
FR_P5                  = f11
FR_P4                  = f12
FR_P3                  = f13
FR_P2                  = f14
FR_NormX               = f15

FR_Ln2                 = f32
FR_C01                 = f33
FR_A17                 = f33
FR_C00                 = f34
FR_Xp2                 = f34
FR_A00                 = f34
FR_A16                 = f34
FR_C11                 = f35
FR_A15                 = f35
FR_C10                 = f36
FR_Xp3                 = f36
FR_A14                 = f36
FR_B1                  = f36
FR_C21                 = f37
FR_A13                 = f37
FR_PR01                = f37
FR_C20                 = f38
FR_Xp6                 = f38
FR_A12                 = f38
FR_C31                 = f39
FR_Xp7                 = f39
FR_B0                  = f39
FR_A11                 = f39
FR_C30                 = f40
FR_Xp8                 = f40
FR_A10                 = f40
FR_PR00                = f40
FR_C41                 = f41
FR_Xp9                 = f41
FR_A9                  = f41
FR_PR11                = f41
FR_C40                 = f42
FR_A8                  = f42
FR_C51                 = f43
FR_Xp11                = f43
FR_A7                  = f43
FR_C50                 = f44
FR_C                   = f44
FR_Xp12                = f44
FR_A6                  = f44
FR_Xm2                 = f45
FR_Xp13                = f45
FR_A5                  = f45
FR_PR10                = f45
FR_C61                 = f46
FR_Xp14                = f46
FR_A4                  = f46
FR_PR21                = f46
FR_C60                 = f47
FR_Xp15                = f47
FR_A3                  = f47
FR_PR20                = f47
FR_C71                 = f48
FR_Xp16                = f48
FR_A2                  = f48
FR_PR31                = f48
FR_C70                 = f49
FR_Xp17                = f49
FR_A1                  = f49
FR_PR30                = f49
FR_C81                 = f50
FR_B17                 = f50
FR_A0                  = f50
FR_C80                 = f51
FR_B16                 = f51
FR_C91                 = f52
FR_B15                 = f52
FR_C90                 = f53
FR_B14                 = f53
FR_CA1                 = f54
FR_B13                 = f54
FR_CA0                 = f55
FR_B12                 = f55
FR_CN                  = f56
FR_Qlo                 = f56
FR_PRN                 = f56
FR_B7                  = f57
FR_B6                  = f58
FR_Qhi                 = f59
FR_x                   = f60
FR_x2                  = f61
FR_TpNxLn2             = f62
FR_W2                  = f63
FR_x4                  = f64
FR_r4                  = f64
FR_x8                  = f65
FR_r8                  = f65
FR_r05                 = f66
FR_Xm05                = f66
FR_B5                  = f66
FR_LnSqrt2Pi           = f67
FR_B4                  = f67
FR_InvX                = f68
FR_B3                  = f68
FR_InvX2               = f69
FR_B2                  = f69
FR_W4                  = f70
FR_OvfBound            = f71
FR_05                  = f72
FR_LocalMin            = f73
FR_tmp                 = f73
FR_LnX                 = f74
FR_Xf                  = f75
FR_InvXf               = f76
FR_rf                  = f77
FR_rf2                 = f78
FR_P54f                = f79
FR_P32f                = f80
FR_rf3                 = f81
FR_P10f                = f82
FR_TpNxLn2f            = f83
FR_Nf                  = f84
FR_LnXf                = f85
FR_int_Nf              = f86
FR_Tf                  = f87
FR_Xf2                 = f88
FR_Xp10                = f89
FR_w3                  = f90
FR_S28                 = f90
FR_w2                  = f91
FR_S26                 = f91
FR_w6                  = f92
FR_S24                 = f92
FR_w4                  = f93
FR_S22                 = f93
FR_w                   = f94
FR_S20                 = f94
FR_Q8                  = f95
FR_S18                 = f95
FR_Q7                  = f96
FR_S16                 = f96
FR_Q4                  = f97
FR_S14                 = f97
FR_Q3                  = f98
FR_S12                 = f98
FR_Q6                  = f99
FR_S10                 = f99
FR_Q5                  = f100
FR_S8                  = f100
FR_Q2                  = f101
FR_S6                  = f101
FR_Root                = f101
FR_S4                  = f102
FR_Q1                  = f102
FR_S2                  = f103
FR_Xp1                 = f104
FR_Xf4                 = f105
FR_Xf8                 = f106
FR_Xfr                 = f107
FR_Xf6                 = f108
FR_Ntrunc              = f109
FR_B9                  = f110
FR_2                   = f110
FR_B8                  = f111
FR_3                   = f111
FR_5                   = f112
FR_Xp4                 = f113
FR_Xp5                 = f114
FR_P54                 = f115
FR_P32                 = f116
FR_P10                 = f117
FR_r                   = f118
FR_r2                  = f119
FR_r3                  = f120
FR_T                   = f121
FR_int_Ntrunc          = f122

//===================================

GR_TAG                 = r8
GR_ExpMask             = r8
GR_ExpBias             = r9
GR_ad_Roots            = r9
GR_Expf                = r10
GR_Arg                 = r10
GR_SignExp             = r11
GR_ArgXfr              = r11

GR_Exp                 = r14
GR_Arg125              = r14
GR_RootInd             = r14
GR_ArgAsIs             = r15
GR_Arg175              = r15
GR_Sig                 = r16
GR_Ind                 = r17
GR_ad_Dx               = r17
GR_ad_1                = r18
GR_SignExp_w           = r19
GR_2_25                = r19
GR_Arg025              = r19
GR_Arg15               = r19
GR_Arg17               = r19
GR_Exp_w               = r19//21
GR_ad_2                = r20
GR_2xDx                = r21
GR_SignOfGamma         = r21
GR_fff9                = r22
GR_Offs                = r22
GR_ad_Co7              = r23
GR_Arg075              = r23
GR_Arg0875             = r23
GR_ad_T                = r24
GR_ad_Root             = r24
GR_Ind                 = r24
GR_ad_Co               = r25
GR_ad_Ce               = r26
GR_ad_Ce7              = r27
GR_Arg05               = r27
GR_Offs7               = r28
GR_ArgXfrAsIs          = r28
GR_ExpOf2              = r29
GR_ad_LnT              = r29
GR_Dx                  = r29
GR_ExpOf256            = r30
GR_0x30033             = r30
GR_Root                = r30
GR_PseudoRoot          = r30
GR_ad_Data             = r31
GR_ad_SignGam          = r31


GR_SAVE_B0             = r33
GR_SAVE_PFS            = r34
GR_SAVE_GP             = r35
GR_SAVE_SP             = r36

GR_Parameter_X         = r37
GR_Parameter_Y         = r38
GR_Parameter_RESULT    = r39
GR_Parameter_TAG       = r40



// Data tables
//==============================================================

RODATA
.align 16
LOCAL_OBJECT_START(lgamma_data)
// polynomial approximation of ln(GAMMA(x)), 2.25 <= x < 512
// [2.25; 4)
data8 0xF888E8D7892718A2,0xC001 // C01
data8 0xF62F273BA12A4639,0x3FFD // C11
data8 0xA93AC50A37EC8D38,0xBFFC // C21
data8 0xB4CC43D2C161E057,0xBFFF // C31
data8 0xC6AC672F0C1392C7,0xC000 // C41
data8 0xA292B9AE3276942E,0xC001 // C51
data8 0xE554E4CCCA6C7B7B,0xC001 // C61
data8 0x92F0F55FBC87F860,0xC002 // C71
data8 0xAF60D0112843F6C1,0xC002 // C81
data8 0xC5956500FA3D92E7,0xC002 // C91
data8 0xD3B22CCBD8587750,0xC002 // CA1
data8 0xD888B6CF34159B54,0x4001 // C00
data8 0xBCB79C8329FD9F44,0x3FFE // C10
data8 0xCB8896FAD69C455D,0x4000 // C20
data8 0xE510A424639EBF5E,0x4001 // C30
data8 0xC65ED41B097486B3,0x4002 // C40
// [4; 8)
data8 0x9F1F3C822D03080E,0xC001 // C01
data8 0x941CACFA9C0FA8A6,0xC001 // C11
data8 0xFE34336391D99CB7,0xC000 // C21
data8 0xC40BAEAA165F81A1,0xC000 // C31
data8 0xFE3AE166E9B4DE8F,0xBFFF // C41
data8 0xD744F91AF7DAF873,0xBFFE // C51
data8 0x87871851E9C32D02,0x3FFD // C61
data8 0x9C93C03C502E808F,0x3FFF // C71
data8 0xF78BED07501D6A8E,0x3FFF // C81
data8 0x92FE41BA8BEADF70,0x4000 // C91
data8 0xA021878E1903A2C6,0x3FFF // CA1
data8 0xC85EFAC379FAFEE2,0x4001 // C00
data8 0xC10D7AAB7CEC7FF2,0x4001 // C10
data8 0xB3537BDF603E454C,0x4001 // C20
data8 0xA0D44E3D5BBE44C4,0x4001 // C30
data8 0x8B9C229B6241E7B3,0x4001 // C40
// [8; 16)
data8 0xD16AB33AEC220DF6,0x3FFF // C01
data8 0x987483646E150BCD,0x4000 // C11
data8 0x80C10A24C863999B,0x4000 // C21
data8 0xA39A8EB6F8AACE75,0x3FFF // C31
data8 0x93E04A1379BEC764,0x3FFD // C41
data8 0xD9F59C4BD3A69BD1,0xBFFE // C51
data8 0x82094EC891179B1A,0xC000 // C61
data8 0xC90CFE3A24F70659,0xC000 // C71
data8 0x827984EA7C155184,0xC001 // C81
data8 0x981BFDF79D1E0D80,0xC001 // C91
data8 0xA37209A8B97D230D,0xC001 // CA1
data8 0xAA1989737D6BA66D,0x3FFE // C00
data8 0xDBC013A351630AF8,0x3FFF // C10
data8 0x8B8D47698299389D,0x4000 // C20
data8 0xACCDD1315DE06EB0,0x4000 // C30
data8 0xD3414A5AC81BBB2D,0x4000 // C40
// [16; 32)
data8 0xECB2B0BE75C5F995,0x3FFF // C01
data8 0x9DD28BD6DBC96500,0x4000 // C11
data8 0x8521431B99C6244F,0x4000 // C21
data8 0xA95F92612B8413C3,0x3FFF // C31
data8 0x9C76E643B22D9544,0x3FFD // C41
data8 0xDD90EA99417C8038,0xBFFE // C51
data8 0x84EA6B6D32E5F906,0xC000 // C61
data8 0xCDBFE499E05AA622,0xC000 // C71
data8 0x8594A7DE35427100,0xC001 // C81
data8 0x9BC1CB2C10DC702F,0xC001 // C91
data8 0xA7602268762666B0,0xC001 // CA1
data8 0xDA082BCC6BDB8F7B,0x3FFE // C00
data8 0xEEBFE1C99322B85E,0x3FFF // C10
data8 0x96FED4C785361946,0x4000 // C20
data8 0xB9E3A7207C16B2FE,0x4000 // C30
data8 0xE1E8170CED48E2C7,0x4000 // C40
// [32; 64)
data8 0xFD481EB9AEDD53E7,0x3FFF // C01
data8 0xA216FB66AC8C53E1,0x4000 // C11
data8 0x885FF935787553BA,0x4000 // C21
data8 0xAD471CD89A313327,0x3FFF // C31
data8 0x9FF13FBA139D21E0,0x3FFD // C41
data8 0xE25E1663A6EE0266,0xBFFE // C51
data8 0x87BE51DD5D262FA2,0xC000 // C61
data8 0xD211A9D4CCE55696,0xC000 // C71
data8 0x885BEFC29FDED3C9,0xC001 // C81
data8 0x9EFA48E6367A67F6,0xC001 // C91
data8 0xAAD3978FC0791297,0xC001 // CA1
data8 0xF96D210DF37A0AEA,0x3FFE // C00
data8 0xFE11DC6783917C82,0x3FFF // C10
data8 0x9FFCD928291B7DDE,0x4000 // C20
data8 0xC4518F4A80E09AE1,0x4000 // C30
data8 0xEDDFE9E0FD297C63,0x4000 // C40
// [64; 128)
data8 0x840E2E62609B0AD3,0x4000 // C01
data8 0xA5275A0DD0D3DDF8,0x4000 // C11
data8 0x8AADC6ABFC441731,0x4000 // C21
data8 0xB041C6696BE90E50,0x3FFF // C31
data8 0xA4A8C9153F4B037E,0x3FFD // C41
data8 0xE3C6A461A7B86736,0xBFFE // C51
data8 0x89047681C6DE7673,0xC000 // C61
data8 0xD42DF77A480092DF,0xC000 // C71
data8 0x89C25D17F086FB20,0xC001 // C81
data8 0xA09F907D02E34EC7,0xC001 // C91
data8 0xAC998A9CB79805B7,0xC001 // CA1
data8 0x875CC9B69AE964CC,0x3FFF // C00
data8 0x847836BA85DD4C12,0x4000 // C10
data8 0xA5F3CB2B32E74936,0x4000 // C20
data8 0xCAE2197C96CB5A0F,0x4000 // C30
data8 0xF50F7EB60DE5CD09,0x4000 // C40
// [128; 256)
data8 0x87D9065DD1876926,0x4000 // C01
data8 0xA781C28FDAD7CC25,0x4000 // C11
data8 0x8C6A4FCE35A7EC8D,0x4000 // C21
data8 0xB27BA081728354F9,0x3FFF // C31
data8 0xA82FEA7124B0EB2B,0x3FFD // C41
data8 0xE4C996E42ECBF77A,0xBFFE // C51
data8 0x89F1A92C84FA538F,0xC000 // C61
data8 0xD5B6CFF7DB7F6070,0xC000 // C71
data8 0x8AC6B561FAE38B66,0xC001 // C81
data8 0xA1D1505C438D8F46,0xC001 // C91
data8 0xADE2DC1C924FEC81,0xC001 // CA1
data8 0x8EF6CC62A7E0EB5A,0x3FFF // C00
data8 0x88A2FFC0ABCB00C0,0x4000 // C10
data8 0xAA6EA8FCB75B065B,0x4000 // C20
data8 0xCFC4B82B3D5C9363,0x4000 // C30
data8 0xFA60FD85DE861771,0x4000 // C40
// [256; 512)
data8 0x8AAA7CE4ED5C1EFD,0x4000 // C01
data8 0xA9679234FB56F1E1,0x4000 // C11
data8 0x8DCE02287789D841,0x4000 // C21
data8 0xB44328EF30A8DE7E,0x3FFF // C31
data8 0xAB0DC564BFA1AB12,0x3FFD // C41
data8 0xE5882B16FCF2D3CB,0xBFFE // C51
data8 0x8AA7F48993006A86,0xC000 // C61
data8 0xD6E63752D192750D,0xC000 // C71
data8 0x8B90080B17853295,0xC001 // C81
data8 0xA2BDD4253128D1AB,0xC001 // C91
data8 0xAEE1A042F96B8121,0xC001 // CA1
data8 0x94A9C37A42E43BA7,0x3FFF // C00
data8 0x8BFA54E703878F5A,0x4000 // C10
data8 0xADFA426DDF14647B,0x4000 // C20
data8 0xD39C7F7B3958EAF0,0x4000 // C30
data8 0xFE8C3987853C01E3,0x4000 // C40
//
// [2.25; 4)
data8 0x943AF77763601441,0x4003 // C50
data8 0xC8A93F9ECB06E891,0x4003 // C60
data8 0xFC2E5A4AD33DE19D,0x4003 // C70
data8 0x9526B75B38670119,0x4004 // C80
data8 0xA7675879D68B587E,0x4004 // C90
data8 0xB31DFA672D7FB8C0,0x4004 // CA0
data8 0x83A27775D86F9A81,0xBFD7 // CN
// [4; 8)
data8 0xEB8049BA5E79ADA3,0x4000 // C50
data8 0xC20C95EA99037228,0x4000 // C60
data8 0x9D4A8C864053CEB8,0x4000 // C70
data8 0xFC7716544AB0C5C9,0x3FFF // C80
data8 0xC7EB985259EABA5F,0x3FFF // C90
data8 0xC042FB3B4C95096D,0x3FFD // CA0
data8 0xCC2A7F930856177B,0x3FEE // CN
// [8; 16)
data8 0xFE1903679D078C7A,0x4000 // C50
data8 0x957C221AB90171F1,0x4001 // C60
data8 0xAB2C53B2A78F4031,0x4001 // C70
data8 0xBE080AE6063AE387,0x4001 // C80
data8 0xCC019A0311605CB9,0x4001 // C90
data8 0xD3739D85A12C8ADF,0x4001 // CA0
data8 0x81FA4D2B7BD7A82D,0x3FEF // CN
// [16; 32)
data8 0x871F69E2DD221F02,0x4001 // C50
data8 0x9E3EF2D477442A9C,0x4001 // C60
data8 0xB48733582B3C82C5,0x4001 // C70
data8 0xC7DB9B3C25854A2A,0x4001 // C80
data8 0xD628B87975BE898F,0x4001 // C90
data8 0xDDC569C321FF119C,0x4001 // CA0
data8 0xB27B65560DF7ADA7,0x3FEF // CN
// [32; 64)
data8 0x8DE4127349719B22,0x4001 // C50
data8 0xA5C30A7760F5FBB2,0x4001 // C60
data8 0xBCB4096055AA2A4E,0x4001 // C70
data8 0xD08F5F2FB4E7B899,0x4001 // C80
data8 0xDF39ED39DC91F9CF,0x4001 // C90
data8 0xE7063E45322F072E,0x4001 // CA0
data8 0x85A9E11DDDDE67C8,0x3FF0 // CN
// [64; 128)
data8 0x91CA191EB80E8893,0x4001 // C50
data8 0xA9F1D5A55397334A,0x4001 // C60
data8 0xC1222710295094E3,0x4001 // C70
data8 0xD52FFABBA6CBE5C6,0x4001 // C80
data8 0xE3FD9D5282052E1D,0x4001 // C90
data8 0xEBDBE47BB662F3EF,0x4001 // CA0
data8 0xEF889F489D88FD31,0x3FF0 // CN
// [128; 256)
data8 0x94AA029C2286F8D2,0x4001 // C50
data8 0xAD0549E55A72389F,0x4001 // C60
data8 0xC4628899DAF94BA4,0x4001 // C70
data8 0xD89432A4161C72CB,0x4001 // C80
data8 0xE77ABA75E9C38F3A,0x4001 // C90
data8 0xEF65BFFFF71347FF,0x4001 // CA0
data8 0xE2627460064D918D,0x3FF1 // CN
// [256; 512)
data8 0x96E9890D722C2FC1,0x4001 // C50
data8 0xAF6C2236F6A1CEC4,0x4001 // C60
data8 0xC6EBB8C9F987D20D,0x4001 // C70
data8 0xDB38CEFD5EF328CC,0x4001 // C80
data8 0xEA3265DC66C9A0B4,0x4001 // C90
data8 0xF2272D6B368C70B1,0x4001 // CA0
data8 0xDBFF93ECEBCEF1F3,0x3FF2 // CN
//
data8 0x3FDD8B618D5AF8FE // point of local minimum on [1;2]
data8 0x3FE0000000000000 // 0.5
data8 0xBFC5555DA7212371 // P5
data8 0x3FC999A19EEF5826 // P4
data8 0xb17217f7d1cf79ac,0x3ffe // ln(2)
data8 0xEB3F8E4325F5A535,0x3FFE // ln(sqrt(4*arcsin(1)))
//
data8 0xBFCFFFFFFFFEF009 // P3
data8 0x3FD555555554ECB2 // P2
data8 0xBF66C16C16C16C17 // W4=B4/12=-1/360
data8 0x7F5754D9278B51A8 // overflow boundary (first inf result)
data8 0xAAAAAAAAAAAAAAAB,0x3FFB // W2=B2/2=1/12
//
data8 0x3FBC756AC654273B // Q8
data8 0xBFC001A42489AB4D // Q7
data8 0x3FC99999999A169B // Q4
data8 0xBFD00000000019AC // Q3
data8 0x3FC2492479AA0DF8 // Q6
data8 0xBFC5555544986F52 // Q5
data8 0x3FD5555555555555 // Q2
data8 0xBFE0000000000000 // Q1, P1 = -0.5
//
data8 0x80200aaeac44ef38,0x3ff6 // ln(1/frcpa(1+  0/2^-8))
data8 0xc09090a2c35aa070,0x3ff7 // ln(1/frcpa(1+  1/2^-8))
data8 0xa0c94fcb41977c75,0x3ff8 // ln(1/frcpa(1+  2/2^-8))
data8 0xe18b9c263af83301,0x3ff8 // ln(1/frcpa(1+  3/2^-8))
data8 0x8d35c8d6399c30ea,0x3ff9 // ln(1/frcpa(1+  4/2^-8))
data8 0xadd4d2ecd601cbb8,0x3ff9 // ln(1/frcpa(1+  5/2^-8))
data8 0xce95403a192f9f01,0x3ff9 // ln(1/frcpa(1+  6/2^-8))
data8 0xeb59392cbcc01096,0x3ff9 // ln(1/frcpa(1+  7/2^-8))
data8 0x862c7d0cefd54c5d,0x3ffa // ln(1/frcpa(1+  8/2^-8))
data8 0x94aa63c65e70d499,0x3ffa // ln(1/frcpa(1+  9/2^-8))
data8 0xa54a696d4b62b382,0x3ffa // ln(1/frcpa(1+ 10/2^-8))
data8 0xb3e4a796a5dac208,0x3ffa // ln(1/frcpa(1+ 11/2^-8))
data8 0xc28c45b1878340a9,0x3ffa // ln(1/frcpa(1+ 12/2^-8))
data8 0xd35c55f39d7a6235,0x3ffa // ln(1/frcpa(1+ 13/2^-8))
data8 0xe220f037b954f1f5,0x3ffa // ln(1/frcpa(1+ 14/2^-8))
data8 0xf0f3389b036834f3,0x3ffa // ln(1/frcpa(1+ 15/2^-8))
data8 0xffd3488d5c980465,0x3ffa // ln(1/frcpa(1+ 16/2^-8))
data8 0x87609ce2ed300490,0x3ffb // ln(1/frcpa(1+ 17/2^-8))
data8 0x8ede9321e8c85927,0x3ffb // ln(1/frcpa(1+ 18/2^-8))
data8 0x96639427f2f8e2f4,0x3ffb // ln(1/frcpa(1+ 19/2^-8))
data8 0x9defad3e8f73217b,0x3ffb // ln(1/frcpa(1+ 20/2^-8))
data8 0xa582ebd50097029c,0x3ffb // ln(1/frcpa(1+ 21/2^-8))
data8 0xac06dbe75ab80fee,0x3ffb // ln(1/frcpa(1+ 22/2^-8))
data8 0xb3a78449b2d3ccca,0x3ffb // ln(1/frcpa(1+ 23/2^-8))
data8 0xbb4f79635ab46bb2,0x3ffb // ln(1/frcpa(1+ 24/2^-8))
data8 0xc2fec93a83523f3f,0x3ffb // ln(1/frcpa(1+ 25/2^-8))
data8 0xc99af2eaca4c4571,0x3ffb // ln(1/frcpa(1+ 26/2^-8))
data8 0xd1581106472fa653,0x3ffb // ln(1/frcpa(1+ 27/2^-8))
data8 0xd8002560d4355f2e,0x3ffb // ln(1/frcpa(1+ 28/2^-8))
data8 0xdfcb43b4fe508632,0x3ffb // ln(1/frcpa(1+ 29/2^-8))
data8 0xe67f6dff709d4119,0x3ffb // ln(1/frcpa(1+ 30/2^-8))
data8 0xed393b1c22351280,0x3ffb // ln(1/frcpa(1+ 31/2^-8))
data8 0xf5192bff087bcc35,0x3ffb // ln(1/frcpa(1+ 32/2^-8))
data8 0xfbdf4ff6dfef2fa3,0x3ffb // ln(1/frcpa(1+ 33/2^-8))
data8 0x81559a97f92f9cc7,0x3ffc // ln(1/frcpa(1+ 34/2^-8))
data8 0x84be72bce90266e8,0x3ffc // ln(1/frcpa(1+ 35/2^-8))
data8 0x88bc74113f23def2,0x3ffc // ln(1/frcpa(1+ 36/2^-8))
data8 0x8c2ba3edf6799d11,0x3ffc // ln(1/frcpa(1+ 37/2^-8))
data8 0x8f9dc92f92ea08b1,0x3ffc // ln(1/frcpa(1+ 38/2^-8))
data8 0x9312e8f36efab5a7,0x3ffc // ln(1/frcpa(1+ 39/2^-8))
data8 0x968b08643409ceb6,0x3ffc // ln(1/frcpa(1+ 40/2^-8))
data8 0x9a062cba08a1708c,0x3ffc // ln(1/frcpa(1+ 41/2^-8))
data8 0x9d845b3abf95485c,0x3ffc // ln(1/frcpa(1+ 42/2^-8))
data8 0xa06fd841bc001bb4,0x3ffc // ln(1/frcpa(1+ 43/2^-8))
data8 0xa3f3a74652fbe0db,0x3ffc // ln(1/frcpa(1+ 44/2^-8))
data8 0xa77a8fb2336f20f5,0x3ffc // ln(1/frcpa(1+ 45/2^-8))
data8 0xab0497015d28b0a0,0x3ffc // ln(1/frcpa(1+ 46/2^-8))
data8 0xae91c2be6ba6a615,0x3ffc // ln(1/frcpa(1+ 47/2^-8))
data8 0xb189d1b99aebb20b,0x3ffc // ln(1/frcpa(1+ 48/2^-8))
data8 0xb51cced5de9c1b2c,0x3ffc // ln(1/frcpa(1+ 49/2^-8))
data8 0xb819bee9e720d42f,0x3ffc // ln(1/frcpa(1+ 50/2^-8))
data8 0xbbb2a0947b093a5d,0x3ffc // ln(1/frcpa(1+ 51/2^-8))
data8 0xbf4ec1505811684a,0x3ffc // ln(1/frcpa(1+ 52/2^-8))
data8 0xc2535bacfa8975ff,0x3ffc // ln(1/frcpa(1+ 53/2^-8))
data8 0xc55a3eafad187eb8,0x3ffc // ln(1/frcpa(1+ 54/2^-8))
data8 0xc8ff2484b2c0da74,0x3ffc // ln(1/frcpa(1+ 55/2^-8))
data8 0xcc0b1a008d53ab76,0x3ffc // ln(1/frcpa(1+ 56/2^-8))
data8 0xcfb6203844b3209b,0x3ffc // ln(1/frcpa(1+ 57/2^-8))
data8 0xd2c73949a47a19f5,0x3ffc // ln(1/frcpa(1+ 58/2^-8))
data8 0xd5daae18b49d6695,0x3ffc // ln(1/frcpa(1+ 59/2^-8))
data8 0xd8f08248cf7e8019,0x3ffc // ln(1/frcpa(1+ 60/2^-8))
data8 0xdca7749f1b3e540e,0x3ffc // ln(1/frcpa(1+ 61/2^-8))
data8 0xdfc28e033aaaf7c7,0x3ffc // ln(1/frcpa(1+ 62/2^-8))
data8 0xe2e012a5f91d2f55,0x3ffc // ln(1/frcpa(1+ 63/2^-8))
data8 0xe600064ed9e292a8,0x3ffc // ln(1/frcpa(1+ 64/2^-8))
data8 0xe9226cce42b39f60,0x3ffc // ln(1/frcpa(1+ 65/2^-8))
data8 0xec4749fd97a28360,0x3ffc // ln(1/frcpa(1+ 66/2^-8))
data8 0xef6ea1bf57780495,0x3ffc // ln(1/frcpa(1+ 67/2^-8))
data8 0xf29877ff38809091,0x3ffc // ln(1/frcpa(1+ 68/2^-8))
data8 0xf5c4d0b245cb89be,0x3ffc // ln(1/frcpa(1+ 69/2^-8))
data8 0xf8f3afd6fcdef3aa,0x3ffc // ln(1/frcpa(1+ 70/2^-8))
data8 0xfc2519756be1abc7,0x3ffc // ln(1/frcpa(1+ 71/2^-8))
data8 0xff59119f503e6832,0x3ffc // ln(1/frcpa(1+ 72/2^-8))
data8 0x8147ce381ae0e146,0x3ffd // ln(1/frcpa(1+ 73/2^-8))
data8 0x82e45f06cb1ad0f2,0x3ffd // ln(1/frcpa(1+ 74/2^-8))
data8 0x842f5c7c573cbaa2,0x3ffd // ln(1/frcpa(1+ 75/2^-8))
data8 0x85ce471968c8893a,0x3ffd // ln(1/frcpa(1+ 76/2^-8))
data8 0x876e8305bc04066d,0x3ffd // ln(1/frcpa(1+ 77/2^-8))
data8 0x891012678031fbb3,0x3ffd // ln(1/frcpa(1+ 78/2^-8))
data8 0x8a5f1493d766a05f,0x3ffd // ln(1/frcpa(1+ 79/2^-8))
data8 0x8c030c778c56fa00,0x3ffd // ln(1/frcpa(1+ 80/2^-8))
data8 0x8da85df17e31d9ae,0x3ffd // ln(1/frcpa(1+ 81/2^-8))
data8 0x8efa663e7921687e,0x3ffd // ln(1/frcpa(1+ 82/2^-8))
data8 0x90a22b6875c6a1f8,0x3ffd // ln(1/frcpa(1+ 83/2^-8))
data8 0x91f62cc8f5d24837,0x3ffd // ln(1/frcpa(1+ 84/2^-8))
data8 0x93a06cfc3857d980,0x3ffd // ln(1/frcpa(1+ 85/2^-8))
data8 0x94f66d5e6fd01ced,0x3ffd // ln(1/frcpa(1+ 86/2^-8))
data8 0x96a330156e6772f2,0x3ffd // ln(1/frcpa(1+ 87/2^-8))
data8 0x97fb3582754ea25b,0x3ffd // ln(1/frcpa(1+ 88/2^-8))
data8 0x99aa8259aad1bbf2,0x3ffd // ln(1/frcpa(1+ 89/2^-8))
data8 0x9b0492f6227ae4a8,0x3ffd // ln(1/frcpa(1+ 90/2^-8))
data8 0x9c5f8e199bf3a7a5,0x3ffd // ln(1/frcpa(1+ 91/2^-8))
data8 0x9e1293b9998c1daa,0x3ffd // ln(1/frcpa(1+ 92/2^-8))
data8 0x9f6fa31e0b41f308,0x3ffd // ln(1/frcpa(1+ 93/2^-8))
data8 0xa0cda11eaf46390e,0x3ffd // ln(1/frcpa(1+ 94/2^-8))
data8 0xa22c8f029cfa45aa,0x3ffd // ln(1/frcpa(1+ 95/2^-8))
data8 0xa3e48badb7856b34,0x3ffd // ln(1/frcpa(1+ 96/2^-8))
data8 0xa5459a0aa95849f9,0x3ffd // ln(1/frcpa(1+ 97/2^-8))
data8 0xa6a79c84480cfebd,0x3ffd // ln(1/frcpa(1+ 98/2^-8))
data8 0xa80a946d0fcb3eb2,0x3ffd // ln(1/frcpa(1+ 99/2^-8))
data8 0xa96e831a3ea7b314,0x3ffd // ln(1/frcpa(1+100/2^-8))
data8 0xaad369e3dc544e3b,0x3ffd // ln(1/frcpa(1+101/2^-8))
data8 0xac92e9588952c815,0x3ffd // ln(1/frcpa(1+102/2^-8))
data8 0xadfa035aa1ed8fdc,0x3ffd // ln(1/frcpa(1+103/2^-8))
data8 0xaf6219eae1ad6e34,0x3ffd // ln(1/frcpa(1+104/2^-8))
data8 0xb0cb2e6d8160f753,0x3ffd // ln(1/frcpa(1+105/2^-8))
data8 0xb2354249ad950f72,0x3ffd // ln(1/frcpa(1+106/2^-8))
data8 0xb3a056e98ef4a3b4,0x3ffd // ln(1/frcpa(1+107/2^-8))
data8 0xb50c6dba52c6292a,0x3ffd // ln(1/frcpa(1+108/2^-8))
data8 0xb679882c33876165,0x3ffd // ln(1/frcpa(1+109/2^-8))
data8 0xb78c07429785cedc,0x3ffd // ln(1/frcpa(1+110/2^-8))
data8 0xb8faeb8dc4a77d24,0x3ffd // ln(1/frcpa(1+111/2^-8))
data8 0xba6ad77eb36ae0d6,0x3ffd // ln(1/frcpa(1+112/2^-8))
data8 0xbbdbcc915e9bee50,0x3ffd // ln(1/frcpa(1+113/2^-8))
data8 0xbd4dcc44f8cf12ef,0x3ffd // ln(1/frcpa(1+114/2^-8))
data8 0xbec0d81bf5b531fa,0x3ffd // ln(1/frcpa(1+115/2^-8))
data8 0xc034f19c139186f4,0x3ffd // ln(1/frcpa(1+116/2^-8))
data8 0xc14cb69f7c5e55ab,0x3ffd // ln(1/frcpa(1+117/2^-8))
data8 0xc2c2abbb6e5fd56f,0x3ffd // ln(1/frcpa(1+118/2^-8))
data8 0xc439b2c193e6771e,0x3ffd // ln(1/frcpa(1+119/2^-8))
data8 0xc553acb9d5c67733,0x3ffd // ln(1/frcpa(1+120/2^-8))
data8 0xc6cc96e441272441,0x3ffd // ln(1/frcpa(1+121/2^-8))
data8 0xc8469753eca88c30,0x3ffd // ln(1/frcpa(1+122/2^-8))
data8 0xc962cf3ce072b05c,0x3ffd // ln(1/frcpa(1+123/2^-8))
data8 0xcadeba8771f694aa,0x3ffd // ln(1/frcpa(1+124/2^-8))
data8 0xcc5bc08d1f72da94,0x3ffd // ln(1/frcpa(1+125/2^-8))
data8 0xcd7a3f99ea035c29,0x3ffd // ln(1/frcpa(1+126/2^-8))
data8 0xcef93860c8a53c35,0x3ffd // ln(1/frcpa(1+127/2^-8))
data8 0xd0192f68a7ed23df,0x3ffd // ln(1/frcpa(1+128/2^-8))
data8 0xd19a201127d3c645,0x3ffd // ln(1/frcpa(1+129/2^-8))
data8 0xd2bb92f4061c172c,0x3ffd // ln(1/frcpa(1+130/2^-8))
data8 0xd43e80b2ee8cc8fc,0x3ffd // ln(1/frcpa(1+131/2^-8))
data8 0xd56173601fc4ade4,0x3ffd // ln(1/frcpa(1+132/2^-8))
data8 0xd6e6637efb54086f,0x3ffd // ln(1/frcpa(1+133/2^-8))
data8 0xd80ad9f58f3c8193,0x3ffd // ln(1/frcpa(1+134/2^-8))
data8 0xd991d1d31aca41f8,0x3ffd // ln(1/frcpa(1+135/2^-8))
data8 0xdab7d02231484a93,0x3ffd // ln(1/frcpa(1+136/2^-8))
data8 0xdc40d532cde49a54,0x3ffd // ln(1/frcpa(1+137/2^-8))
data8 0xdd685f79ed8b265e,0x3ffd // ln(1/frcpa(1+138/2^-8))
data8 0xde9094bbc0e17b1d,0x3ffd // ln(1/frcpa(1+139/2^-8))
data8 0xe01c91b78440c425,0x3ffd // ln(1/frcpa(1+140/2^-8))
data8 0xe14658f26997e729,0x3ffd // ln(1/frcpa(1+141/2^-8))
data8 0xe270cdc2391e0d23,0x3ffd // ln(1/frcpa(1+142/2^-8))
data8 0xe3ffce3a2aa64922,0x3ffd // ln(1/frcpa(1+143/2^-8))
data8 0xe52bdb274ed82887,0x3ffd // ln(1/frcpa(1+144/2^-8))
data8 0xe6589852e75d7df6,0x3ffd // ln(1/frcpa(1+145/2^-8))
data8 0xe786068c79937a7d,0x3ffd // ln(1/frcpa(1+146/2^-8))
data8 0xe91903adad100911,0x3ffd // ln(1/frcpa(1+147/2^-8))
data8 0xea481236f7d35bb0,0x3ffd // ln(1/frcpa(1+148/2^-8))
data8 0xeb77d48c692e6b14,0x3ffd // ln(1/frcpa(1+149/2^-8))
data8 0xeca84b83d7297b87,0x3ffd // ln(1/frcpa(1+150/2^-8))
data8 0xedd977f4962aa158,0x3ffd // ln(1/frcpa(1+151/2^-8))
data8 0xef7179a22f257754,0x3ffd // ln(1/frcpa(1+152/2^-8))
data8 0xf0a450d139366ca7,0x3ffd // ln(1/frcpa(1+153/2^-8))
data8 0xf1d7e0524ff9ffdb,0x3ffd // ln(1/frcpa(1+154/2^-8))
data8 0xf30c29036a8b6cae,0x3ffd // ln(1/frcpa(1+155/2^-8))
data8 0xf4412bc411ea8d92,0x3ffd // ln(1/frcpa(1+156/2^-8))
data8 0xf576e97564c8619d,0x3ffd // ln(1/frcpa(1+157/2^-8))
data8 0xf6ad62fa1b5f172f,0x3ffd // ln(1/frcpa(1+158/2^-8))
data8 0xf7e499368b55c542,0x3ffd // ln(1/frcpa(1+159/2^-8))
data8 0xf91c8d10abaffe22,0x3ffd // ln(1/frcpa(1+160/2^-8))
data8 0xfa553f7018c966f3,0x3ffd // ln(1/frcpa(1+161/2^-8))
data8 0xfb8eb13e185d802c,0x3ffd // ln(1/frcpa(1+162/2^-8))
data8 0xfcc8e3659d9bcbed,0x3ffd // ln(1/frcpa(1+163/2^-8))
data8 0xfe03d6d34d487fd2,0x3ffd // ln(1/frcpa(1+164/2^-8))
data8 0xff3f8c7581e9f0ae,0x3ffd // ln(1/frcpa(1+165/2^-8))
data8 0x803e029e280173ae,0x3ffe // ln(1/frcpa(1+166/2^-8))
data8 0x80dca10cc52d0757,0x3ffe // ln(1/frcpa(1+167/2^-8))
data8 0x817ba200632755a1,0x3ffe // ln(1/frcpa(1+168/2^-8))
data8 0x821b05f3b01d6774,0x3ffe // ln(1/frcpa(1+169/2^-8))
data8 0x82bacd623ff19d06,0x3ffe // ln(1/frcpa(1+170/2^-8))
data8 0x835af8c88e7a8f47,0x3ffe // ln(1/frcpa(1+171/2^-8))
data8 0x83c5f8299e2b4091,0x3ffe // ln(1/frcpa(1+172/2^-8))
data8 0x8466cb43f3d87300,0x3ffe // ln(1/frcpa(1+173/2^-8))
data8 0x850803a67c80ca4b,0x3ffe // ln(1/frcpa(1+174/2^-8))
data8 0x85a9a1d11a23b461,0x3ffe // ln(1/frcpa(1+175/2^-8))
data8 0x864ba644a18e6e05,0x3ffe // ln(1/frcpa(1+176/2^-8))
data8 0x86ee1182dcc432f7,0x3ffe // ln(1/frcpa(1+177/2^-8))
data8 0x875a925d7e48c316,0x3ffe // ln(1/frcpa(1+178/2^-8))
data8 0x87fdaa109d23aef7,0x3ffe // ln(1/frcpa(1+179/2^-8))
data8 0x88a129ed4becfaf2,0x3ffe // ln(1/frcpa(1+180/2^-8))
data8 0x89451278ecd7f9cf,0x3ffe // ln(1/frcpa(1+181/2^-8))
data8 0x89b29295f8432617,0x3ffe // ln(1/frcpa(1+182/2^-8))
data8 0x8a572ac5a5496882,0x3ffe // ln(1/frcpa(1+183/2^-8))
data8 0x8afc2d0ce3b2dadf,0x3ffe // ln(1/frcpa(1+184/2^-8))
data8 0x8b6a69c608cfd3af,0x3ffe // ln(1/frcpa(1+185/2^-8))
data8 0x8c101e106e899a83,0x3ffe // ln(1/frcpa(1+186/2^-8))
data8 0x8cb63de258f9d626,0x3ffe // ln(1/frcpa(1+187/2^-8))
data8 0x8d2539c5bd19e2b1,0x3ffe // ln(1/frcpa(1+188/2^-8))
data8 0x8dcc0e064b29e6f1,0x3ffe // ln(1/frcpa(1+189/2^-8))
data8 0x8e734f45d88357ae,0x3ffe // ln(1/frcpa(1+190/2^-8))
data8 0x8ee30cef034a20db,0x3ffe // ln(1/frcpa(1+191/2^-8))
data8 0x8f8b0515686d1d06,0x3ffe // ln(1/frcpa(1+192/2^-8))
data8 0x90336bba039bf32f,0x3ffe // ln(1/frcpa(1+193/2^-8))
data8 0x90a3edd23d1c9d58,0x3ffe // ln(1/frcpa(1+194/2^-8))
data8 0x914d0de2f5d61b32,0x3ffe // ln(1/frcpa(1+195/2^-8))
data8 0x91be0c20d28173b5,0x3ffe // ln(1/frcpa(1+196/2^-8))
data8 0x9267e737c06cd34a,0x3ffe // ln(1/frcpa(1+197/2^-8))
data8 0x92d962ae6abb1237,0x3ffe // ln(1/frcpa(1+198/2^-8))
data8 0x9383fa6afbe2074c,0x3ffe // ln(1/frcpa(1+199/2^-8))
data8 0x942f0421651c1c4e,0x3ffe // ln(1/frcpa(1+200/2^-8))
data8 0x94a14a3845bb985e,0x3ffe // ln(1/frcpa(1+201/2^-8))
data8 0x954d133857f861e7,0x3ffe // ln(1/frcpa(1+202/2^-8))
data8 0x95bfd96468e604c4,0x3ffe // ln(1/frcpa(1+203/2^-8))
data8 0x9632d31cafafa858,0x3ffe // ln(1/frcpa(1+204/2^-8))
data8 0x96dfaabd86fa1647,0x3ffe // ln(1/frcpa(1+205/2^-8))
data8 0x9753261fcbb2a594,0x3ffe // ln(1/frcpa(1+206/2^-8))
data8 0x9800c11b426b996d,0x3ffe // ln(1/frcpa(1+207/2^-8))
data8 0x9874bf4d45ae663c,0x3ffe // ln(1/frcpa(1+208/2^-8))
data8 0x99231f5ee9a74f79,0x3ffe // ln(1/frcpa(1+209/2^-8))
data8 0x9997a18a56bcad28,0x3ffe // ln(1/frcpa(1+210/2^-8))
data8 0x9a46c873a3267e79,0x3ffe // ln(1/frcpa(1+211/2^-8))
data8 0x9abbcfc621eb6cb6,0x3ffe // ln(1/frcpa(1+212/2^-8))
data8 0x9b310cb0d354c990,0x3ffe // ln(1/frcpa(1+213/2^-8))
data8 0x9be14cf9e1b3515c,0x3ffe // ln(1/frcpa(1+214/2^-8))
data8 0x9c5710b8cbb73a43,0x3ffe // ln(1/frcpa(1+215/2^-8))
data8 0x9ccd0abd301f399c,0x3ffe // ln(1/frcpa(1+216/2^-8))
data8 0x9d7e67f3bdce8888,0x3ffe // ln(1/frcpa(1+217/2^-8))
data8 0x9df4ea81a99daa01,0x3ffe // ln(1/frcpa(1+218/2^-8))
data8 0x9e6ba405a54514ba,0x3ffe // ln(1/frcpa(1+219/2^-8))
data8 0x9f1e21c8c7bb62b3,0x3ffe // ln(1/frcpa(1+220/2^-8))
data8 0x9f956593f6b6355c,0x3ffe // ln(1/frcpa(1+221/2^-8))
data8 0xa00ce1092e5498c3,0x3ffe // ln(1/frcpa(1+222/2^-8))
data8 0xa0c08309c4b912c1,0x3ffe // ln(1/frcpa(1+223/2^-8))
data8 0xa1388a8c6faa2afa,0x3ffe // ln(1/frcpa(1+224/2^-8))
data8 0xa1b0ca7095b5f985,0x3ffe // ln(1/frcpa(1+225/2^-8))
data8 0xa22942eb47534a00,0x3ffe // ln(1/frcpa(1+226/2^-8))
data8 0xa2de62326449d0a3,0x3ffe // ln(1/frcpa(1+227/2^-8))
data8 0xa357690f88bfe345,0x3ffe // ln(1/frcpa(1+228/2^-8))
data8 0xa3d0a93f45169a4b,0x3ffe // ln(1/frcpa(1+229/2^-8))
data8 0xa44a22f7ffe65f30,0x3ffe // ln(1/frcpa(1+230/2^-8))
data8 0xa500c5e5b4c1aa36,0x3ffe // ln(1/frcpa(1+231/2^-8))
data8 0xa57ad064eb2ebbc2,0x3ffe // ln(1/frcpa(1+232/2^-8))
data8 0xa5f5152dedf4384e,0x3ffe // ln(1/frcpa(1+233/2^-8))
data8 0xa66f9478856233ec,0x3ffe // ln(1/frcpa(1+234/2^-8))
data8 0xa6ea4e7cca02c32e,0x3ffe // ln(1/frcpa(1+235/2^-8))
data8 0xa765437325341ccf,0x3ffe // ln(1/frcpa(1+236/2^-8))
data8 0xa81e21e6c75b4020,0x3ffe // ln(1/frcpa(1+237/2^-8))
data8 0xa899ab333fe2b9ca,0x3ffe // ln(1/frcpa(1+238/2^-8))
data8 0xa9157039c51ebe71,0x3ffe // ln(1/frcpa(1+239/2^-8))
data8 0xa991713433c2b999,0x3ffe // ln(1/frcpa(1+240/2^-8))
data8 0xaa0dae5cbcc048b3,0x3ffe // ln(1/frcpa(1+241/2^-8))
data8 0xaa8a27ede5eb13ad,0x3ffe // ln(1/frcpa(1+242/2^-8))
data8 0xab06de228a9e3499,0x3ffe // ln(1/frcpa(1+243/2^-8))
data8 0xab83d135dc633301,0x3ffe // ln(1/frcpa(1+244/2^-8))
data8 0xac3fb076adc7fe7a,0x3ffe // ln(1/frcpa(1+245/2^-8))
data8 0xacbd3cbbe47988f1,0x3ffe // ln(1/frcpa(1+246/2^-8))
data8 0xad3b06b1a5dc57c3,0x3ffe // ln(1/frcpa(1+247/2^-8))
data8 0xadb90e94af887717,0x3ffe // ln(1/frcpa(1+248/2^-8))
data8 0xae3754a218f7c816,0x3ffe // ln(1/frcpa(1+249/2^-8))
data8 0xaeb5d9175437afa2,0x3ffe // ln(1/frcpa(1+250/2^-8))
data8 0xaf349c322e9c7cee,0x3ffe // ln(1/frcpa(1+251/2^-8))
data8 0xafb39e30d1768d1c,0x3ffe // ln(1/frcpa(1+252/2^-8))
data8 0xb032df51c2c93116,0x3ffe // ln(1/frcpa(1+253/2^-8))
data8 0xb0b25fd3e6035ad9,0x3ffe // ln(1/frcpa(1+254/2^-8))
data8 0xb1321ff67cba178c,0x3ffe // ln(1/frcpa(1+255/2^-8))
//
data8 0xC7DC2985D3B44557,0x3FCA // A00
//
// polynomial approximation of ln(GAMMA(x)), 1 <= x < 2.25
// [0.875,1.25)
data8 0xBF9A04F7E40C8498,0x3FAB79D8D9380F03 // C17,C16
data8 0xBFB3B63609CA0CBD,0x3FB5564EA1675539 // C13,C12
data8 0xBFBC806766F48C41,0x3FC010B36CDA773A // C9,C8
data8 0xD45CE0BD54BE3D67,0xBFFC // C5
data8 0xCD26AADF559676D0,0xBFFD // C3
data8 0x93C467E37DB0C7A7,0xBFFE // C1
data8 0xBFB10C251723B123,0x3FB2669DAD69A12D // C15,C14
data8 0xBFB748A3CFCE4717,0x3FB9A01DEE29966A // C11,C10
data8 0xBFC2703A1D85497E,0x3FC5B40CB0FD353C // C7,C6
data8 0x8A8991563ECBBA5D,0x3FFD // C4
data8 0xD28D3312983E9844,0x3FFE // C2
data8 0,0                       // C0
// [1.25,1.75)
data8 0xBF12680486396DE6,0x3F23C51FC332CD9D // C17,C16
data8 0xBF422633DA3A1496,0x3F4CC70680768857 // C13,C12
data8 0xBF6E2F1A1F804B5D,0x3F78FCE02A032428 // C9,C8
data8 0x864D46FA895985C1,0xBFFA // C5
data8 0x97213C6E35E12043,0xBFFC // C3
data8 0x8A8A42A401D979B7,0x3FC7 // C1
data8 0xBF2E098A8A2332A8,0x3F370E61B73B205C // C15,C14
data8 0xBF56F9849D3BC6CC,0x3F6283126F58D7F4 // C11,C10
data8 0xBF851F9F9516A98F,0x3F9266E797A1433F // C7,C6
data8 0x845A14A6A81B0638,0x3FFB // C4
data8 0xF7B95E4771C55C99,0x3FFD // C2
data8 0xF8CDCDE61C520E0F,0xBFFB // C0
// [1.75,2.25)
data8 0xBEA01D7AFA5D8F52,0x3EB1010986E60253 // C17,C16
data8 0xBEE3CBEDB4C918AA,0x3EF580F6D9D0F72D // C13,C12
data8 0xBF2D3FD4C7F68563,0x3F40B36AF884AE9A // C9,C8
data8 0xF2027E10C7B051EC,0xBFF7 // C5
data8 0x89F000D2ABB03401,0xBFFB // C3
data8 0xD8773039049E70B6,0x3FFD // C1
data8 0xBEC112CD07CFC31A,0x3ED2528A428D30E1 // C15,C14
data8 0xBF078DE5618D8C9F,0x3F1A127AD811A53D // C11,C10
data8 0xBF538AC5C2BF540D,0x3F67ADD6EADB5718 // C7,C6
data8 0xA8991563EC243383,0x3FF9 // C4
data8 0xA51A6625307D3230,0x3FFD // C2
data8 0,0                       // C0
//
// polynomial approximation of ln(sin(Pi*x)/(Pi*x)), 9 <= x <= 0.5
data8 0xBFDC1BF0931AE591,0x3FD36D6D6CE263D7 //S28,S26
data8 0xBFBD516F4FD9FB18,0xBFBBE1703F315086 //S20,S18
data8 0xAAB5A3CCEFCD3628,0xBFFC //S12
data8 0x80859B5C318E19A5,0xBFFD //S8
data8 0x8A8991563EC7EB33,0xBFFE //S4
data8 0xBFD23AB9E6CC88AC,0xBF9957F5146FC7AF //S24,S22
data8 0xBFC007B324E23040,0xBFC248DEC29CAC4A //S16,S14
data8 0xCD00EFF2F8F86899,0xBFFC //S10
data8 0xADA06587FACD668B,0xBFFD //S6
data8 0xD28D3312983E98A0,0xBFFF //S2
//
data8 0x8090F777D7942F73,0x4001 // PR01
data8 0xE5B521193CF61E63,0x4000 // PR11
data8 0xC02C000000001939 // (-15;-14)
data8 0x0000000000000233 // (-15;-14)
data8 0xC02A000000016124 // (-14;-13)
data8 0x0000000000002BFB // (-14;-13)
data8 0xC02800000011EED9 // (-13;-12)
data8 0x0000000000025CBB // (-13;-12)
data8 0xC026000000D7322A // (-12;-11)
data8 0x00000000001E1095 // (-12;-11)
data8 0xC0240000093F2777 // (-11;-10)
data8 0x00000000013DD3DC // (-11;-10)
data8 0xC02200005C7768FB // (-10;-9)
data8 0x000000000C9539B9 // (-10;-9)
data8 0xC02000034028B3F9 // (-9;-8)
data8 0x000000007570C565 // (-9;-8)
data8 0xC01C0033FDEDFE1F // (-8;-7)
data8 0x00000007357E670E // (-8;-7)
data8 0xC018016B25897C8D // (-7;-6)
data8 0x000000346DC5D639 // (-7;-6)
data8 0xC014086A57F0B6D9 // (-6;-5)
data8 0x0000010624DD2F1B // (-6;-5)
data8 0xC010284E78599581 // (-5;-4)
data8 0x0000051EB851EB85 // (-5;-4)
data8 0xC009260DBC9E59AF // (-4;-3)
data8 0x000028F5C28F5C29 // (-4;-3)
data8 0xC003A7FC9600F86C // (-3;-2)
data8 0x0000666666666666 // (-3;-2)
data8 0xCC15879606130890,0x4000 // PR21
data8 0xB42FE3281465E1CC,0x4000 // PR31
//
data8 0x828185F0B95C9916,0x4001 // PR00
//
data8 0xD4D3C819E4E5654B,0x4000 // PR10
data8 0xA82FBBA4FCC75298,0x4000 // PR20
data8 0xC02DFFFFFFFFFE52 // (-15;-14)
data8 0x000000000000001C // (-15;-14)
data8 0xC02BFFFFFFFFE6C7 // (-14;-13)
data8 0x00000000000001A6 // (-14;-13)
data8 0xC029FFFFFFFE9EDC // (-13;-12)
data8 0x0000000000002BFB // (-13;-12)
data8 0xC027FFFFFFEE1127 // (-12;-11)
data8 0x000000000001EEC8 // (-12;-11)
data8 0xC025FFFFFF28CDD4 // (-11;-10)
data8 0x00000000001E1095 // (-11;-10)
data8 0xC023FFFFF6C0D7C0 // (-10;-9)
data8 0x000000000101B2B3 // (-10;-9)
data8 0xC021FFFFA3884BD0 // (-9;-8)
data8 0x000000000D6BF94D // (-9;-8)
data8 0xC01FFFF97F8159CF // (-8;-7)
data8 0x00000000C9539B89 // (-8;-7)
data8 0xC01BFFCBF76B86F0 // (-7;-6)
data8 0x00000007357E670E // (-7;-6)
data8 0xC017FE92F591F40D // (-6;-5)
data8 0x000000346DC5D639 // (-6;-5)
data8 0xC013F7577A6EEAFD // (-5;-4)
data8 0x00000147AE147AE1 // (-5;-4)
data8 0xC00FA471547C2FE5 // (-4;-3)
data8 0x00000C49BA5E353F // (-4;-3)
data8 0xC005FB410A1BD901 // (-3;-2)
data8 0x000053F7CED91687 // (-3;-2)
data8 0x80151BB918A293AA,0x4000 // PR30
data8 0xB3C9F8F47422A314,0x400B // PRN
//
// right negative roots
//(-3;-2)
data8 0x40BFCF8B90BE7F6B,0x40B237623345EFC3 // A15,A14
data8 0x407A92EFB03B281E,0x40728700C7819759 // A11,A10
data8 0x403809F04EF4D0F2,0x4038D32F682D9593 // A7,A6
data8 0xB4A5302C53C2F2D8,0x3FFF // A3
data8 0xC1FF4B357A9B0383,0x3FFF // A1
data8 0x409C46632EB4B2D3,0x4091A72AFA2148F5 // A13,A12
data8 0x4059297AC79A88DB,0x40548EAA7BE7FA6B // A9,A8
data8 0x4017339FE04B227F,0x4021718D7CA09E02 // A5,A4
data8 0x9B775D8017AAE668,0x4001 // A2
data8 0x8191DB68FF4366A1,0x3FC9 // A0
//(-4;-3)
data8 0x425260910D35307B,0x422668F5BE7983BB // A15,A14
data8 0x41A4454DBE4BEE43,0x41799CA93F6EA817 // A11,A10
data8 0x40FBB97AA1400F31,0x40D293C3F7ADAB15 // A7,A6
data8 0xE089B8926AE4517B,0x4005 // A3
data8 0xF90532F97D630C69,0x4001 // A1
data8 0x41F9F0CF98C5F2EA,0x41D026336C6BF394 // A13,A12
data8 0x415057F61156D5B8,0x41251EA3055CB754 // A9,A8
data8 0x40A99A6337D9FC2B,0x408267203D776151 // A5,A4
data8 0xCEA694BB8A8827A9,0x4003 // A2
data8 0xF4B02F1D73D30EED,0x3FCD // A0
//(-5;-4)
data8 0x4412365489340979,0x43C86441BAFDEE39 // A15,A14
data8 0x42ED68FCB19352DD,0x42A45FCE3905CD6F // A11,A10
data8 0x41CD14FE49FD4FCA,0x41855E3DBFA89744 // A7,A6
data8 0xAACD88D954E0EC16,0x400B // A3
data8 0xD652E7A490B0DCDF,0x4003 // A1
data8 0x437F52608E0E752A,0x433560E0633E33D5 // A13,A12
data8 0x425C83998976DE3D,0x421433DCCD3B473B // A9,A8
data8 0x4140261EB5732106,0x40F96D18E21AE6CC // A5,A4
data8 0xA220AE6C09FA8A0E,0x4007 // A2
data8 0xCC1682D17A2B5A58,0xBFCF // A0
//(-6;-5)
data8 0x4630E41D6386CF5A,0x45C2E7992C628C8C // A15,A14
data8 0x447AABEC714F913A,0x440EDCAB45339F3A // A11,A10
data8 0x42C9A8D00C97E3CE,0x425F7D8D5BEAB44D // A7,A6
data8 0x929EC2B1FB95BB5B,0x4012 // A3
data8 0xF6B970414D717D38,0x4005 // A1
data8 0x45545E578976F6A2,0x44E738288DD52686 // A13,A12
data8 0x43A20921FEC49492,0x433557FD7C6A41B3 // A9,A8
data8 0x41F3E01773761DB4,0x418A225DF2DA6C47 // A5,A4
data8 0xE7661976117F9312,0x400B // A2
data8 0xC33C13FEE07494DE,0x3FCF // A0
//(-7;-6)
data8 0x4898F1E6133305AD,0x4802C5306FE4A850 // A15,A14
data8 0x463FD37946B44094,0x45A8D489B784C2DD // A11,A10
data8 0x43E9500995815F06,0x4354F21E2FEE6DF5 // A7,A6
data8 0xEF281D1E1BBE10BD,0x4019 // A3
data8 0xB4EF24F1D78C2029,0x4008 // A1
data8 0x476AB1D5930011E5,0x46D4867E77BFB622 // A13,A12
data8 0x45139151ECDEF7C5,0x447F3A2BC6BF466F // A9,A8
data8 0x42C1D3D50713FA40,0x422F9C7B52556A1B // A5,A4
data8 0xFE711A4267CEA83A,0x4010 // A2
data8 0xD11E91B3FF8F4B94,0xBFD2 // A0
//(-8;-7)
data8 0x4B39E57569811B6E,0x4A7656073EB1FA21 // A15,A14
data8 0x482C9B24A516B0BB,0x47698FF55139C62B // A11,A10
data8 0x452393E2BC8E8D04,0x44628E1C710DA478 // A7,A6
data8 0x9F2A95AF1B7A773F,0x4022 // A3
data8 0x9DA03D51C303C918,0x400B // A1
data8 0x49B24C241A3D5BCB,0x48F01CB936ECDA67 // A13,A12
data8 0x46A712B3425C6797,0x45E5164114BD6DA1 // A9,A8
data8 0x43A216A356069D01,0x42E25E42A45E2108 // A5,A4
data8 0xC1F42ED57BBC2529,0x4016 // A2
data8 0xB1C7B615A7DCA8A9,0xBFD7 // A0
//(-9;-8)
data8 0x4E09D478E5EE857D,0x4D1647782106E9AB // A15,A14
data8 0x4A3C7F4D51927548,0x49497954796D743A // A11,A10
data8 0x467387BD6AF0CBDF,0x4582843E134111D2 // A7,A6
data8 0x9F003C6DE9666513,0x402B // A3
data8 0x9D8447F6BF99950A,0x400E // A1
data8 0x4C22364D238C61A9,0x4B300B18050AB940 // A13,A12
data8 0x4857004D64215772,0x4765074E448C3C9A // A9,A8
data8 0x44920E9EA07BF624,0x43A257BEC94BBF48 // A5,A4
data8 0xC1D1C49AC5B2A4B4,0x401C // A2
data8 0x9A749AF9F2D2E688,0x3FDB // A0
//(-10;-9)
data8 0x5102C7C43EA26C83,0x4FDCD174DEB0426B // A15,A14
data8 0x4C6A036195CD5BAD,0x4B44ABB52B65628A // A11,A10
data8 0x47D6439374B98FED,0x46B2C3903EF44D7D // A7,A6
data8 0xE25BAF73AB8A7DB3,0x4034 // A3
data8 0xB130901CA6D81B61,0x4011 // A1
data8 0x4EB50BB0726AE206,0x4D907A96E6D2B6E2 // A13,A12
data8 0x4A20975D78EAF01A,0x48FAF79C9C3E7908 // A9,A8
data8 0x459044144129A247,0x446D6043FA3150A3 // A5,A4
data8 0xF547997E083D9BA7,0x4022 // A2
data8 0x977AF525A6ECA1BC,0x3FDC // A0
//(-11;-10)
data8 0x5420A5D5E90C6D73,0x52C4710A503DC67A // A15,A14
data8 0x4EB2ED07BA88D2A8,0x4D581001ED9A5ECE // A11,A10
data8 0x494A8A28E9E3DFEF,0x47F1E4E1E476793E // A7,A6
data8 0xDD0C97E12D4A3378,0x403E // A3
data8 0xDD7C12D5182FD543,0x4014 // A1
data8 0x5167ED536877A072,0x500DF9AF21DDC0B6 // A13,A12
data8 0x4BFEE6F04BC34FF8,0x4AA4175CEF736A5E // A9,A8
data8 0x4698D1B4388FEC78,0x4541EDE7607A600D // A5,A4
data8 0xBF9F645F282AC552,0x4029 // A2
data8 0xAE1BBE4D3CDACCF4,0x3FE1 // A0
//(-12;-11)
data8 0x575F0EEF5FB7D4C0,0x55CBB7302B211A7C // A15,A14
data8 0x5113A4F1825C7CB2,0x4F822A0D46E0605A // A11,A10
data8 0x4ACED38FC8BE069A,0x493E3B56D2649F18 // A7,A6
data8 0x8FA8FF5DF8B72D5E,0x4049 // A3
data8 0x9845417E8598D642,0x4018 // A1
data8 0x5437780541C3F2D3,0x52A56279B563C1B2 // A13,A12
data8 0x4DF0F71A48C50188,0x4C600B358988DEBF // A9,A8
data8 0x47AE7EE95BDA3DE9,0x46200599DC16B18F // A5,A4
data8 0xB5249F914932E55D,0x4030 // A2
data8 0xEAE760CD2C086094,0x3FE5 // A0
//(-13;-12)
data8 0x5ABA5848651F6D18,0x58EF60D8A817650B // A15,A14
data8 0x538A8CA86E13EFB1,0x51C05DBD4D01076D // A11,A10
data8 0x4C607594C339D259,0x4A9585BD5BF932BB // A7,A6
data8 0xF26D282C36EC3611,0x4053 // A3
data8 0xE467DF4810EE7EEE,0x401B // A1
data8 0x5721D9BA485E8CC3,0x5555AF2CCFB2104D // A13,A12
data8 0x4FF4619A17B14EA6,0x4E29B2F29EB9F8C4 // A9,A8
data8 0x48CCF27629D46E79,0x47044715F991A63D // A5,A4
data8 0xCBC92FB9BDAA95A9,0x4037 // A2
data8 0xFB743A426163665B,0xBFE6 // A0
//(-14;-13)
data8 0x5E3295B24B353EAA,0x5C2B447E29796F20 // A15,A14
data8 0x5615A35CB5EAFAE5,0x54106AB089C95CAF // A11,A10
data8 0x4DFEC7D93501900A,0x4BF8C4C685F01B83 // A7,A6
data8 0x820899603D9A74D5,0x405F // A3
data8 0xB9949919933821CB,0x401F // A1
data8 0x5A23373DB9A995AC,0x581CBA0AF7F53009 // A13,A12
data8 0x520929836BB304CD,0x500386409A7076DA // A9,A8
data8 0x49F480173FEAF90B,0x47F1ACB14B810793 // A5,A4
data8 0x86881B8674DBF205,0x403F // A2
data8 0x8CF3CC35AA2C5F90,0x3FED // A0
//(-15;-14)
data8 0x61C37D53BE0029D6,0x5F80667CD9D68354 // A15,A14
data8 0x58B3F01898E6605B,0x567149652116DB6A // A11,A10
data8 0x4FA82FA4F5D35B00,0x4D663DB00832DF8F // A7,A6
data8 0xAE426731C9B94996,0x406A // A3
data8 0xA264C84BE3708F3F,0x4023 // A1
data8 0x5D3B254BC1C806A8,0x5AF72E736048B553 // A13,A12
data8 0x542E476505104BB0,0x51EAD96CDC4FB48F // A9,A8
data8 0x4B25095F498DB134,0x48E4B9FDEBFE24AB // A5,A4
data8 0xCE076A5A116C1D34,0x4046 // A2
data8 0x940013871A15050B,0x3FF1 // A0
//
// left negative roots
//(-3;-2)
data8 0x41AEB7998DBE2B2C,0xC19053D8FAC05DF7 // A16,A15
data8 0x4133197BF1ADEAF9,0xC1150728B9B82072 // A12,A11
data8 0x40BDBA65E74F4526,0xC0A12239BEEF8F72 // A8,A7
data8 0xFA8256664F99E2AA,0x4004 // A4
data8 0x9933F9E132D2A5DB,0x4002 // A2
data8 0x416FFB167B85F77C,0xC15166AE0ACCF87C // A14,A13
data8 0x40F75815106322C0,0xC0DA2D23C59C348D // A10,A9
data8 0x4084373F7CC42043,0xC0685884581F8C61 // A6,A5
data8 0xA0C2D6186460FF9D,0xC003 // A3
data8 0xF5096D48258CA0AD,0xBFFF // A1
//(-4;-3)
data8 0xC3E5BD233016D4B9,0x43A084DAD2D94AB1 // A15,A14
data8 0xC2CCFFF5E5AED722,0x4286D143AC7D29A6 // A11,A10
data8 0xC1B7DBBE0680D07B,0x4173E8F3ABB79CED // A7,A6
data8 0xE929ACEA59799BAF,0xC00A // A3
data8 0xA5CCECB362B21E1C,0xC003 // A1
data8 0xC357EED873871B81,0x43128E0B873204FC // A13,A12
data8 0xC242225FA76E8450,0x41FD2F76AE7386CE // A9,A8
data8 0xC13116F7806D0C7A,0x40EE8F829F141025 // A5,A4
data8 0xFBB6F57021B5B397,0x4006 // A2
data8 0xEEE019B4C05AC269,0xBFCB // A0
//(-5;-4)
data8 0xC626A52FE8AAA100,0x45B9FD1F4DDFE31E // A15,A14
data8 0xC473812A5675F08B,0x440738530AECC254 // A11,A10
data8 0xC2C5068B3F94AC27,0x425A8C5C539A500B // A7,A6
data8 0x869FBFF732F20C3A,0xC012 // A3
data8 0xE91251F7CF25A655,0xC005 // A1
data8 0xC54C18CB48E5DA0F,0x44E07BD36FF561DF // A13,A12
data8 0xC39BEC120D2FEBEA,0x4330FFA5388435BE // A9,A8
data8 0xC1F13D5D163B7FB5,0x418752A6F5AC0F39 // A5,A4
data8 0xDA99E33C51D360F0,0x400B // A2
data8 0x9F47A66A2F53D9B9,0x3FD1 // A0
//(-6;-5)
data8 0xC8970DAC16B6D59E,0x480170728306FD76 // A15,A14
data8 0xC63E0E5030604CF3,0x45A7924D74D57C65 // A11,A10
data8 0xC3E8684E41730FC6,0x43544D54EA2E5B9A // A7,A6
data8 0xEB7404450C47C5F4,0xC019 // A3
data8 0xB30FB521D2C19F8B,0xC008 // A1
data8 0xC768F34D35DF6320,0x46D348B3BB2E68B8 // A13,A12
data8 0xC512AC2FE5EA638E,0x447DF44BC7FC5E17 // A9,A8
data8 0xC2C15EA6B0AAFEF9,0x422EF5D308DBC420 // A5,A4
data8 0xFBCEE5BCA70FD3A3,0x4010 // A2
data8 0x8589A7CFFE0A3E86,0xBFD5 // A0
//(-7;-6)
data8 0xCB3995A0CC961E5A,0x4A7615C6C7116ADD // A15,A14
data8 0xC82C5AFE0BF9C427,0x47695BD2F367668B // A11,A10
data8 0xC52377E70BA14CF5,0x4462775E859E4392 // A7,A6
data8 0x9EC8ED6E4C3D4DBE,0xC022 // A3
data8 0x9D5FBD2E75520E65,0xC00B // A1
data8 0xC9B21BB881A4DDF8,0x48EFEAB06FBA0207 // A13,A12
data8 0xC6A6E8550CBC188F,0x45E4F3D26238B099 // A9,A8
data8 0xC3A20427DF1B110A,0x42E24F3D636F2E4E // A5,A4
data8 0xC1A4D12A82280CFB,0x4016 // A2
data8 0xEF46D8DCCA9E8197,0x3FD2 // A0
//(-8;-7)
data8 0xCE0946982B27DE5B,0x4D15DBC6664E2DD2 // A15,A14
data8 0xCA3C769F6B3B2B93,0x49497251CD0C4363 // A11,A10
data8 0xC67384066C47F489,0x458281393433AB28 // A7,A6
data8 0x9EF3459926D0F14F,0xC02B // A3
data8 0x9D7BB7F2600DFF0B,0xC00E // A1
data8 0xCC22351326C939A7,0x4B3009431C4F1D3F // A13,A12
data8 0xC856FAADDD48815D,0x476502BC3ECA040C // A9,A8
data8 0xC4920C2A84173810,0x43A255C052525F99 // A5,A4
data8 0xC1C73B6554011EFA,0x401C // A2
data8 0x954612700ADF8317,0xBFD8 // A0
//(-9;-8)
data8 0xD102F5CC7B590D3A,0x4FDD0F1C30E4EB22 // A15,A14
data8 0xCC6A02912B0DF650,0x4B44AB18E4FCC159 // A11,A10
data8 0xC7D64314B4A2FAAB,0x46B2C334AE5E2D34 // A7,A6
data8 0xE2598724F7E28E99,0xC034 // A3
data8 0xB12F6FE2E195452C,0xC011 // A1
data8 0xCEB507747AF9356A,0x4D907802C08BA48F // A13,A12
data8 0xCA2096E3DC29516F,0x48FAF6ED046A1DB7 // A9,A8
data8 0xC59043D21BA5EE56,0x446D5FE468B30450 // A5,A4
data8 0xF5460A8196B59C83,0x4022 // A2
data8 0xB108F35A8EDA92D5,0xBFDD // A0
//(-10;-9)
data8 0xD420430D91F8265B,0x52C406CAAAC9E0EE // A15,A14
data8 0xCEB2ECDDDAA3DAD1,0x4D580FDA97F92E3A // A11,A10
data8 0xC94A8A192341B5D4,0x47F1E4D8C690D07B // A7,A6
data8 0xDD0C5F920C2F0D2B,0xC03E // A3
data8 0xDD7BED3631657B48,0xC014 // A1
data8 0xD167F410E64E90A4,0x500DFFED20F714A7 // A13,A12
data8 0xCBFEE6D9043169E9,0x4AA4174F64B40AA7 // A9,A8
data8 0xC698D1A9AF0AB9C2,0x4541EDE14987A887 // A5,A4
data8 0xBF9F43D461B3DE6E,0x4029 // A2
data8 0xF3891A50642FAF26,0x3FE1 // A0
//(-11;-10)
data8 0xD75F0EEAF769D42A,0x55CBB72C8869183A // A15,A14
data8 0xD113A4EF80394F77,0x4F822A0B96B3ECA9 // A11,A10
data8 0xCACED38DC75763CB,0x493E3B5522D2D028 // A7,A6
data8 0x8FA8FB5C92533701,0xC049 // A3
data8 0x98453EDB9339C24E,0xC018 // A1
data8 0xD43778026CCD4B20,0x52A5627753273B9B // A13,A12
data8 0xCDF0F718DD7E1214,0x4C600B34582911EB // A9,A8
data8 0xC7AE7EE7F112362C,0x46200599439C264F // A5,A4
data8 0xB5249C335342B5BC,0x4030 // A2
data8 0x881550711D143475,0x3FE4 // A0
//(-12;-11)
data8 0xDAB9C724EEEE2BBB,0x58EEC971340EDDBA // A15,A14
data8 0xD38A8C8AE63BD8BF,0x51C05DB21CEE00D3 // A11,A10
data8 0xCC607594C311C12D,0x4A9585BD5BE6AB57 // A7,A6
data8 0xF26D282C36EC0E66,0xC053 // A3
data8 0xE467DF1FA674BFAE,0xC01B // A1
data8 0xD721DE506999AA9C,0x5555B34F71B45132 // A13,A12
data8 0xCFF4619A476BF76F,0x4E29B2F2BBE7A67E // A9,A8
data8 0xC8CCF27629D48EDC,0x47044715F991AB46 // A5,A4
data8 0xCBC92FB9BDAA928D,0x4037 // A2
data8 0xCE27C4F01CF53284,0xBFE6 // A0
//(-13;-12)
data8 0xDE3295B24355C5A1,0x5C2B447E298B562D // A15,A14
data8 0xD615A35CB5E92103,0x54106AB089C95E8C // A11,A10
data8 0xCDFEC7D935019005,0x4BF8C4C685F01B83 // A7,A6
data8 0x820899603D9A74D5,0xC05F // A3
data8 0xB9949916F8DF4AC4,0xC01F // A1
data8 0xDA23373DBA0B7548,0x581CBA0AF7F45C01 // A13,A12
data8 0xD20929836BB30934,0x500386409A7076D6 // A9,A8
data8 0xC9F480173FEAF90B,0x47F1ACB14B810793 // A5,A4
data8 0x86881B8674DBF205,0x403F // A2
data8 0x8CFAFA9A142C1FF0,0x3FED // A0
//(-14;-13)
data8 0xE1C33F356FA2C630,0x5F8038B8AA919DD7 // A15,A14
data8 0xD8B3F0167E14982D,0x5671496400BAE0DB // A11,A10
data8 0xCFA82FA4F5D25C3E,0x4D663DB008328C58 // A7,A6
data8 0xAE426731C9B94980,0xC06A // A3
data8 0xA264C84BB8A66F86,0xC023 // A1
data8 0xDD3B26E34762ED1E,0x5AF72F76E3C1B793 // A13,A12
data8 0xD42E476507E3D06E,0x51EAD96CDD881DFA // A9,A8
data8 0xCB25095F498DB15F,0x48E4B9FDEBFE24B5 // A5,A4
data8 0xCE076A5A116C1D32,0x4046 // A2
data8 0x94001BF5A24966F5,0x3FF1 // A0
//(-15;-14)
data8 0xE56DB8B72D7156FF,0x62EAB0CDB22539BE // A15,A14
data8 0xDB63D76B0D3457E7,0x58E254823D0AE4FF // A11,A10
data8 0xD15F060BF548404A,0x4EDE65C20CD4E961 // A7,A6
data8 0x900DA565ED76C19D,0xC076 // A3
data8 0x9868C809852DA712,0xC027 // A1
data8 0xE067CCDA0408AAF0,0x5DE5A79C5C5C54AF // A13,A12
data8 0xD6611ADBF5958ED0,0x53E0294092BE9677 // A9,A8
data8 0xCC5EA28D90EE8C5D,0x49E014930EF336EE // A5,A4
data8 0xB57930DCE7A61AE8,0x404E // A2
data8 0x976BEC1F30DF151C,0x3FF5 // A0
LOCAL_OBJECT_END(lgamma_data)


.section .text
GLOBAL_LIBM_ENTRY(__libm_lgamma)

{ .mfi
      getf.exp      GR_SignExp = f8
      frcpa.s1      FR_C,p9 = f1,f8
      mov           GR_ExpMask = 0x1ffff
}
{ .mfi
      addl          GR_ad_Data = @ltoff(lgamma_data),gp
      fcvt.fx.s1    FR_int_N = f8
      mov           GR_2_25 = 0x4002 // 2.25
};;
{ .mfi
      getf.d        GR_ArgAsIs = f8
      fclass.m      p13,p0 = f8,0x1EF // is x NaTVal, NaN,
                                      // +/-0, +/-INF or +/-deno?
      mov           GR_ExpBias = 0xFFFF
}
{ .mfi
      ld8           GR_ad_Data = [GR_ad_Data]
      fcvt.fx.trunc.s1 FR_int_Ntrunc = f8
      mov           GR_ExpOf256 = 0x10007
};;
{ .mfi
      mov           GR_ExpOf2 = 0x10000
      fcmp.lt.s1    p14,p15 = f8,f0 // p14 if x<0
      dep.z         GR_Ind = GR_SignExp,8,4
}
{ .mfi
      and           GR_Exp = GR_SignExp,GR_ExpMask
      fma.s1        FR_2 = f1,f1,f1
      cmp.lt        p10,p0 = GR_SignExp,GR_ExpBias
};;
{ .mfi
      add           GR_ad_1 = 0xB80,GR_ad_Data
      fnorm.s1      FR_NormX = f8
      shr.u         GR_Arg = GR_ArgAsIs,48
}
{ .mib
      add           GR_ad_Co = GR_Ind,GR_ad_Data
      add           GR_ad_Ce = 0x10,GR_ad_Data
      // jump if the input argument is NaTVal, NaN, +/-0, +/-INF or +/-deno
(p13) br.cond.spnt  lgamma_spec
};;
lgamma_common:
{ .mfi
      ldfpd         FR_LocalMin,FR_05 = [GR_ad_1],16
      fmerge.se     FR_x = f1,f8
      add           GR_ad_2 = 0xBC0,GR_ad_Data
}
{ .mfb
      add           GR_ad_Ce = GR_Ind,GR_ad_Ce
      fms.s1        FR_w = f8,f1,f1 // x-1
      // jump if the input argument is positive and  less than 1.0
(p10) br.cond.spnt  lgamma_0_1
};;
{ .mfi
      ldfe          FR_C01 = [GR_ad_Co],32
      fnma.s1       FR_InvX = FR_C,f8,f1 // NR iteration #1
(p15) cmp.lt.unc    p8,p0 = GR_ExpOf256,GR_SignExp
}
{ .mib
      ldfe          FR_C11 = [GR_ad_Ce],32
(p15) cmp.lt.unc    p11,p0 = GR_Arg,GR_2_25
      // jump if the input argument isn't less than 512.0
(p8)  br.cond.spnt  lgamma_pstirling
};;
{ .mfi
      ldfe          FR_C21 = [GR_ad_Co],32
(p14) fms.s1        FR_r = FR_C,f8,f1 // reduced arg for log(x)
(p14) cmp.lt.unc    p0,p9 = GR_Exp,GR_ExpOf256
}
{ .mib
      ldfe          FR_C31 = [GR_ad_Ce],32
      add           GR_ad_Co7 = 0x12C0,GR_ad_2
      // jump if the input argument is from range [1.0; 2.25)
(p11) br.cond.spnt  lgamma_1_2
};;
{ .mfi
      ldfe          FR_C41 = [GR_ad_Co],32
      fcvt.xf       FR_N = FR_int_N
      add           GR_ad_Ce7 = 0x1310,GR_ad_2
}
{ .mfb
      ldfe          FR_C51 = [GR_ad_Ce],32
(p14) fma.s1        FR_5 = FR_2,FR_2,f1
      // jump if the input argument is less or equal to -512.0
(p9)  br.cond.spnt  lgamma_negstirling
};;
{ .mfi
      ldfe          FR_C61 = [GR_ad_Co],32
(p14) fcvt.xf       FR_Ntrunc = FR_int_Ntrunc
      shr           GR_Ind = GR_Ind,4
}
{ .mfi
      ldfe          FR_C71 = [GR_ad_Ce],32
(p14) fma.s1        FR_Xp1 = f1,f1,FR_NormX // x+1
      cmp.eq        p6,p7 = GR_ExpOf2,GR_SignExp
};;
.pred.rel "mutex",p6,p7
{ .mfi
      ldfe          FR_C81 = [GR_ad_Co],32
(p6)  fma.s1        FR_x = f0,f0,FR_NormX
      shladd        GR_Offs7 = GR_Ind,2,GR_Ind // (ind*16)*5
}
{ .mfi
      ldfe          FR_C91 = [GR_ad_Ce],32
(p7)  fms.s1        FR_x = FR_x,f1,f1
      add           GR_ad_Co7 = 0x800,GR_ad_Data
};;
{ .mfi
      ldfe          FR_CA1 = [GR_ad_Co],32
(p14) fma.s1        FR_3 = f1,f1,FR_2
      shladd        GR_Offs7 = GR_Ind,1,GR_Offs7 // (ind*16)*7
}
{ .mfi
      ldfe          FR_C00 = [GR_ad_Ce],32
(p14) fma.s1        FR_Xp4 = FR_2,FR_2,FR_NormX
      add           GR_ad_Ce7 = 0x810,GR_ad_Data
};;
{ .mfi
      ldfe          FR_C10 = [GR_ad_Co],32
(p6)  fms.s1        FR_Xm2 = FR_w,f1,f1
      add           GR_ad_Co7 = GR_ad_Co7,GR_Offs7
}
{ .mfi
      ldfe          FR_C20 = [GR_ad_Ce],32
(p14) fma.s1        FR_r2 = FR_r,FR_r,f0 // log(x)
      add           GR_ad_Ce7 = GR_ad_Ce7,GR_Offs7
};;
{ .mfi
      ldfe          FR_C30 = [GR_ad_Co],32
(p14) fms.s1        FR_Xf = FR_NormX,f1,FR_N  // xf = x - [x]
(p14) mov           GR_Arg17 = 0xC031 // -17
}
{ .mfi
      ldfe          FR_C40 = [GR_ad_Ce],32
(p14) fma.s1        FR_Xp5 = FR_5,f1,FR_NormX
(p14) sub           GR_Exp = GR_Exp,GR_ExpBias
};;
{ .mfi
      ldfe          FR_C50 = [GR_ad_Co7],32
(p14) fms.s1        FR_Xfr = FR_Xp1,f1,FR_Ntrunc // xfr = (x+1) - [x]
(p14) cmp.lt.unc    p13,p0 = GR_Arg,GR_Arg17
}
{ .mfb
      ldfe          FR_C60 = [GR_ad_Ce7],32
(p14) fma.s1        FR_Xp10 = FR_5,FR_2,FR_NormX
      // jump if the input argument is negative and great than -17.0
(p13) br.cond.spnt  lgamma_negrecursion
};;
{ .mfi
      ldfe          FR_C70 = [GR_ad_Co7],32
      fma.s1        FR_C01 = FR_x,f1,FR_C01
(p14) add           GR_ad_Ce = 0x1310,GR_ad_2
}
{ .mfi
      ldfe          FR_C80 = [GR_ad_Ce7],32
      fma.s1        FR_C11 = FR_x,f1,FR_C11
(p14) add           GR_ad_Co = 0x12C0,GR_ad_2
};;
{ .mfi
      ldfe          FR_C90 = [GR_ad_Co7],32
      fma.s1        FR_C21 = FR_x,f1,FR_C21
      nop.i         0
}
{ .mfi
      ldfe          FR_CA0 = [GR_ad_Ce7],32
      fma.s1        FR_C31 = FR_x,f1,FR_C31
      nop.i         0
};;
{ .mfi
      ldfe          FR_CN = [GR_ad_Co7],32
      fma.s1        FR_C41 = FR_x,f1,FR_C41
      nop.i         0
}
{ .mfi
(p14) ldfpd         FR_P5,FR_P4 = [GR_ad_1],16
      fma.s1        FR_C51 = FR_x,f1,FR_C51
      nop.i         0
};;
{ .mfi
(p14) ldfpd         FR_P3,FR_P2 = [GR_ad_2],16
      fma.s1        FR_C61 = FR_x,f1,FR_C61
      nop.i         0
}
{ .mfi
(p14) ldfe          FR_Ln2 = [GR_ad_1]
      fma.s1        FR_C71 = FR_x,f1,FR_C71
      nop.i         0
};;
{ .mfi
(p14) ldfpd         FR_S28,FR_S26 = [GR_ad_Co],16
      fma.s1        FR_C81 = FR_x,f1,FR_C81
      add           GR_ad_2 = 0x60,GR_ad_2
}
{ .mfi
(p14) ldfpd         FR_S24,FR_S22 = [GR_ad_Ce],16
      fma.s1        FR_C91 = FR_x,f1,FR_C91
      nop.i         0
};;
{ .mfi
(p14) ldfpd         FR_S20,FR_S18 = [GR_ad_Co],16
      fma.s1        FR_CA1 = FR_x,f1,FR_CA1
      nop.i         0
}
{ .mfi
(p14) ldfpd         FR_S16,FR_S14 = [GR_ad_Ce],16
      fma.s1        FR_C01 = FR_C01,FR_x,FR_C00
      nop.i         0
};;
{ .mfi
(p14) getf.exp      GR_SignExp = FR_Xf
      fma.s1        FR_C11 = FR_C11,FR_x,FR_C10
      nop.i         0
}
{ .mfi
(p14) ldfe          FR_S12 = [GR_ad_Co],16
      fma.s1        FR_C21 = FR_C21,FR_x,FR_C20
      nop.i         0
};;
{ .mfi
(p14) getf.sig      GR_Sig = FR_Xf
(p14) frcpa.s1      FR_InvXf,p0 = f1,FR_Xf
      nop.i         0
}
{ .mfi
(p14) ldfe          FR_S10 = [GR_ad_Ce],16
      fma.s1        FR_C41 = FR_C41,FR_x,FR_C40
      nop.i         0
};;
{ .mfi
(p14) ldfe          FR_S8 = [GR_ad_Co],16
      fma.s1        FR_C51 = FR_C51,FR_x,FR_C50
      nop.i         0
}
{ .mfi
(p14) ldfe          FR_S6 = [GR_ad_Ce],16
      fma.s1        FR_C61 = FR_C61,FR_x,FR_C60
(p14) and           GR_Expf = GR_SignExp,GR_ExpMask
};;
{ .mfi
(p14) sub           GR_Expf = GR_Expf,GR_ExpBias
      fma.s1        FR_C71 = FR_C71,FR_x,FR_C70
(p14) shl           GR_Ind = GR_Sig,1
}
{ .mfi
(p14) ldfe          FR_S4 = [GR_ad_Co],16
      fma.s1        FR_C81 = FR_C81,FR_x,FR_C80
(p14) cmp.eq.unc    p8,p0 = 0,GR_Sig
};;
{ .mfi
(p14) setf.sig      FR_int_Nf = GR_Expf
      fma.s1        FR_C91 = FR_C91,FR_x,FR_C90
(p14) shr.u         GR_Ind = GR_Ind,56
}
{ .mfb
(p14) ldfe          FR_S2 = [GR_ad_Ce],16
      fma.s1        FR_CA1 = FR_CA1,FR_x,FR_CA0
      // jump if the input argument is integer number from range (-512.0;-17.0]
(p8)  br.cond.spnt  lgamma_singularity
};;
{ .mfi
(p14) getf.sig      GR_Sig = FR_int_Ntrunc
      fma.s1        FR_C01 = FR_C01,FR_C11,f0
      nop.i         0
}
{ .mfi
(p14) shladd        GR_ad_T = GR_Ind,4,GR_ad_2
      fma.s1        FR_C31 = FR_C31,FR_x,FR_C30
      nop.i         0
};;
{ .mfi
(p14) ldfe          FR_Tf = [GR_ad_T]
(p14) fms.s1        FR_rf = FR_InvXf,FR_Xf,f1 // reduced arg for log({x})
(p14) extr.u        GR_Ind = GR_ArgAsIs,44,8
}
{ .mfi
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
      fma.s1        FR_C21 = FR_C21,FR_C41,f0
      mov           GR_SignOfGamma = 1
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C51 = FR_C51,FR_C61,f0
(p14) tbit.z.unc    p8,p0 = GR_Sig,0
}
{ .mfi
(p14) shladd        GR_ad_T = GR_Ind,4,GR_ad_2
(p6)  fma.s1        FR_CN = FR_CN,FR_Xm2,f0
      nop.i         0
};;
{ .mfi
(p14) setf.sig      FR_int_N = GR_Exp
      fma.s1        FR_C71 = FR_C71,FR_C81,f0
(p8)  sub           GR_SignOfGamma = r0,GR_SignOfGamma
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_Xf2 = FR_Xf,FR_Xf,f0
      nop.i         0
};;
{ .mfi
(p14) ldfe          FR_T = [GR_ad_T]
      fma.s1        FR_C91 = FR_C91,FR_CA1,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_r2 = FR_r,FR_r,f0
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
      fma.s1        FR_C01 = FR_C01,FR_C31,f0
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
(p14) fma.s1        FR_P54 = FR_P5,FR_r,FR_P4
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
}
{ .mfb
      nop.m         0
(p14) fma.s1        FR_P54f = FR_P5,FR_rf,FR_P4
      // jump if the input argument is non-integer from range (-512.0;-17.0]
(p14) br.cond.spnt  lgamma_negpoly
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_C51,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C71 = FR_C71,FR_C91,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_CN  = FR_C01,FR_CN,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_C71,f0
      nop.i         0
};;
{ .mfb
      nop.m         0
      fma.d.s0      f8 = FR_C21,FR_CN,f0
      br.ret.sptk   b0 // exit for arguments from range [2.25; 512.0)
};;
// branch for calculating of ln(GAMMA(x)) for -512 < x < -17
//---------------------------------------------------------------------
.align 32
lgamma_negpoly:
{ .mfi
      nop.m         0
      fma.s1        FR_Xf4 = FR_Xf2,FR_Xf2,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf2,FR_S26
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S24 = FR_S24,FR_Xf2,FR_S22
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S20 = FR_S20,FR_Xf2,FR_S18
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S16 = FR_S16,FR_Xf2,FR_S14
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S12 = FR_S12,FR_Xf2,FR_S10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S8 = FR_S8,FR_Xf2,FR_S6
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S4 = FR_S4,FR_Xf2,FR_S2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_rf2 = FR_rf,FR_rf,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P32f = FR_P3,FR_rf,FR_P2 // log(x)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_r3 = FR_r2,FR_r,f0 // log(x)
      nop.i         0
}
{ .mfi
      nop.m         0
      fcvt.xf       FR_Nf = FR_int_Nf // log({x})
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf4,FR_S24
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_Xf8 = FR_Xf4,FR_Xf4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S20 = FR_S20,FR_Xf4,FR_S16
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_C51,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S12 = FR_S12,FR_Xf4,FR_S8
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C71 = FR_C71,FR_C91,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_P10 = FR_r2,FR_05,FR_r // log(x)
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P54 = FR_P54,FR_r2,FR_P32 // log(x)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_P10f = FR_rf2,FR_05,FR_rf // log({x})
      nop.i         0
}
{ .mfi
      nop.m         0
      fcvt.xf       FR_N = FR_int_N // log(x)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_rf3 = FR_rf2,FR_rf,f0 // log({x})
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P54f = FR_P54f,FR_rf2,FR_P32f // log({x})
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf8,FR_S20
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_TpNxLn2f = FR_Nf,FR_Ln2,FR_Tf // log({x})
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_CN  = FR_C01,FR_CN,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_C21 = FR_C21,FR_C71,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P54 = FR_P54,FR_r3,FR_P10 // log(x)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_TpNxLn2 = FR_N,FR_Ln2,FR_T // log(x)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P54f = FR_P54f,FR_rf3,FR_P10f // log({x})
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf8,FR_S12
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_C21 = FR_C21,FR_CN,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnX = FR_TpNxLn2,f1,FR_P54 // log(x)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnXf = FR_TpNxLn2f,f1,FR_P54f // log({x})
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf4,FR_S4
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnX = FR_LnX,f1,FR_LnXf
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_S28 = FR_S28,FR_Xf2,FR_C21
      nop.i         0
};;
{ .mfb
      nop.m         0
      fms.d.s0      f8 = FR_S28,f1,FR_LnX
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for x >= 512
//---------------------------------------------------------------------
.align 32
lgamma_pstirling:
{ .mfi
      ldfpd         FR_P5,FR_P4 = [GR_ad_1],16
      nop.f         0
      and           GR_Exp = GR_SignExp,GR_ExpMask
}
{ .mfi
      ldfpd         FR_P3,FR_P2 = [GR_ad_2],16
      fma.s1        FR_InvX = FR_C,FR_InvX,FR_C // NR iteration #1
      mov           GR_ExpBias = 0xffff
};;
{ .mfi
      ldfe          FR_Ln2 = [GR_ad_1],16
      nop.f         0
      sub           GR_Exp = GR_Exp,GR_ExpBias
};;
{ .mfi
      ldfpd         FR_W4,FR_OvfBound = [GR_ad_2],16
      nop.f         0
      nop.i         0
};;
{ .mfi
      setf.sig      FR_int_N = GR_Exp
      fms.s1        FR_r = FR_C,f8,f1
      nop.i         0
};;
{ .mmf
      getf.sig      GR_Sig = FR_NormX
      ldfe          FR_LnSqrt2Pi = [GR_ad_1],16
      nop.f         0
};;
{ .mmf
      ldfe          FR_W2 = [GR_ad_2],16
      nop.m         0
      fnma.s1       FR_InvX2 = FR_InvX,FR_NormX,f1 // NR iteration #2
};;
{ .mfi
      add           GR_ad_2 = 0x40,GR_ad_2
      nop.f         0
      shl           GR_Ind = GR_Sig,1
};;
{ .mfi
      mov           GR_SignOfGamma = 1
      nop.f         0
      shr.u         GR_Ind = GR_Ind,56
};;
{ .mfi
      shladd        GR_ad_2 = GR_Ind,4,GR_ad_2
      fma.s1        FR_r2 = FR_r,FR_r,f0
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
};;
{ .mfi
      ldfe          FR_T = [GR_ad_2]
      fma.s1        FR_P54 = FR_P5,FR_r,FR_P4
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcmp.le.s1    p6,p0 = FR_OvfBound,FR_NormX
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_InvX2 = FR_InvX,FR_InvX2,FR_InvX // NR iteration #2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_N = FR_int_N
      nop.i         0
}
{ .mfb
      nop.m         0
      nop.f         0
      // jump if x is great than OVERFLOW_BOUNDARY
(p6)  br.cond.spnt  lgamma_overflow
};;
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
      fma.s1        FR_r3 = FR_r2,FR_r,f0
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
      fnma.s1       FR_P10 = FR_r2,FR_05,FR_r
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P54 = FR_P54,FR_r2,FR_P32
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_InvX = FR_InvX2,FR_NormX,f1 // NR iteration #3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_Xm05 = FR_NormX,f1,FR_05 // (x-1/2)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_TpNxLn2 = FR_N,FR_Ln2,FR_T
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P54 = FR_P54,FR_r3,FR_P10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_InvX = FR_InvX2,FR_InvX,FR_InvX2 // NR iteration #3
      nop.i         0
}
{ .mfi
      nop.m         0
      fms.s1        FR_LnSqrt2Pi = FR_LnSqrt2Pi,f1,FR_NormX // ln(sqrt(2*Pi))-x
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnX = FR_TpNxLn2,f1,FR_P54
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_InvX2 = FR_InvX,FR_InvX,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      // (x-1/2)*ln(x)+ln(sqrt(2*Pi))-x
      fma.s1        FR_LnX = FR_LnX,FR_Xm05,FR_LnSqrt2Pi
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_W2 = FR_W4,FR_InvX2,FR_W2 // W2 + W4/x^2
      nop.i         0
};;
{ .mfb
      nop.m         0
      fma.d.s0      f8 = FR_InvX,FR_W2,FR_LnX
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for x < -512
//---------------------------------------------------------------------
.align 32
lgamma_negstirling:
{ .mfi
      ldfpd         FR_P5,FR_P4 = [GR_ad_1],16
      fms.s1        FR_Xf = FR_NormX,f1,FR_N  // xf = x - [x]
      and           GR_Exp = GR_SignExp,GR_ExpMask
}
{ .mfi
      ldfpd         FR_P3,FR_P2 = [GR_ad_2],16
      fma.s1        FR_InvX = FR_C,FR_InvX,FR_C // NR iteration #1
      mov           GR_0x30033 = 0x30033
};;
{ .mfi
      ldfe          FR_Ln2 = [GR_ad_1],16
      nop.f         0
      extr.u        GR_Ind = GR_ArgAsIs,44,8
}
{ .mib
      ldfd          FR_W4 = [GR_ad_2],16
      // jump if x is less or equal to -2^52, i.e. x is big negative integer
      cmp.leu.unc   p7,p0 = GR_0x30033,GR_SignExp
(p7)  br.cond.spnt  lgamma_singularity
};;
{ .mfi
      ldfpd         FR_S28,FR_S26 = [GR_ad_Co7],16
      nop.f         0
      add           GR_ad_LnT = 0x50,GR_ad_2
}
{ .mfi
      ldfpd         FR_S24,FR_S22 = [GR_ad_Ce7],16
      nop.f         0
      mov           GR_ExpBias = 0xffff
};;
{ .mfi
      ldfpd         FR_S20,FR_S18 = [GR_ad_Co7],16
      nop.f         0
      shladd        GR_ad_T = GR_Ind,4,GR_ad_LnT
}
{ .mfi
      ldfpd         FR_S16,FR_S14 = [GR_ad_Ce7],16
      nop.f         0
      sub           GR_Exp = GR_Exp,GR_ExpBias
};;
{ .mfi
      ldfe          FR_S12 = [GR_ad_Co7],16
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfe          FR_S10 = [GR_ad_Ce7],16
      fms.s1        FR_r = FR_C,f8,f1
      nop.i         0
};;
{ .mmf
      ldfe          FR_S8 = [GR_ad_Co7],16
      ldfe          FR_S6 = [GR_ad_Ce7],16
      nop.f         0
};;
{ .mfi
      ldfe          FR_S4 = [GR_ad_Co7],16
      fma.s1        FR_Xf2 = FR_Xf,FR_Xf,f0
      nop.i         0
}
{ .mfi
      ldfe          FR_S2 = [GR_ad_Ce7],16
      fnma.s1       FR_InvX2 = FR_InvX,FR_NormX,f1 // NR iteration #2
      nop.i         0
};;
{ .mfi
      setf.sig      FR_int_N = GR_Exp
      frcpa.s1      FR_InvXf,p9 = f1,FR_Xf // 1/xf
      nop.i         0
}
{ .mfi
      ldfe          FR_LnSqrt2Pi = [GR_ad_1],16
      nop.f         0
      nop.i         0
};;
{ .mfi
      getf.exp      GR_SignExp = FR_Xf
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfe          FR_W2 = [GR_ad_2],16
      nop.f         0
      nop.i         0
};;
{ .mfi
      getf.sig      GR_Sig = FR_Xf
      fma.s1        FR_P54 = FR_P5,FR_r,FR_P4
      nop.i         0
}
{ .mfi
      ldfe          FR_T = [GR_ad_T]
      fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
};;
{ .mfi
      and           GR_Exp = GR_SignExp,GR_ExpMask
      fma.s1        FR_r2 = FR_r,FR_r,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fms.s1        FR_Xm05 = FR_NormX,f1,FR_05 // (x-1/2)
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_InvX2 = FR_InvX,FR_InvX2,FR_InvX // NR iteration #2
      extr.u        GR_Ind = GR_Sig,55,8
}
{ .mfi
      sub           GR_Exp = GR_Exp,GR_ExpBias
      fma.s1        FR_Xf4 = FR_Xf2,FR_Xf2,f0
      cmp.eq        p6,p0 = 0,GR_Sig
};;
{ .mfi
      setf.sig      FR_int_Nf = GR_Exp
      fma.s1        FR_S28 = FR_S28,FR_Xf2,FR_S26
      shladd        GR_ad_T = GR_Ind,4,GR_ad_LnT
}
{ .mfb
      nop.m         0
      fma.s1        FR_S24 = FR_S24,FR_Xf2,FR_S22
      // jump if the input argument is integer number from range (-512.0;-17.0]
(p6)  br.cond.spnt  lgamma_singularity
};;
{ .mfi
      getf.sig      GR_Sig = FR_int_Ntrunc
      fma.s1        FR_S20 = FR_S20,FR_Xf2,FR_S18
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S16 = FR_S16,FR_Xf2,FR_S14
      nop.i         0
};;
{ .mfi
      ldfe          FR_Tf = [GR_ad_T]
      fma.s1        FR_S12 = FR_S12,FR_Xf2,FR_S10
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S8 = FR_S8,FR_Xf2,FR_S6
      mov           GR_SignOfGamma = 1
};;
{ .mfi
      nop.m         0
      fms.s1        FR_rf = FR_InvXf,FR_Xf,f1 // reduced arg rf
      tbit.z        p8,p0 = GR_Sig,0
}
{ .mfi
      nop.m         0
      fma.s1        FR_r3 = FR_r2,FR_r,f0
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_N = FR_int_N
(p8)  sub           GR_SignOfGamma = r0,GR_SignOfGamma
}
{ .mfi
      nop.m         0
      fnma.s1       FR_InvX = FR_InvX2,FR_NormX,f1 // NR iteration #3
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
      fma.s1        FR_P54 = FR_P54,FR_r2,FR_P32
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
      fnma.s1       FR_P10 = FR_r2,FR_05,FR_r
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Xf8 = FR_Xf4,FR_Xf4,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf4,FR_S24
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S20 = FR_S20,FR_Xf4,FR_S16
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S12 = FR_S12,FR_Xf4,FR_S8
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_rf2 = FR_rf,FR_rf,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P54f = FR_P5,FR_rf,FR_P4
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P32f = FR_P3,FR_rf,FR_P2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_InvX = FR_InvX2,FR_InvX,FR_InvX2 // NR iteration #3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_Nf = FR_int_Nf
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_LnSqrt2Pi = FR_NormX,f1,FR_LnSqrt2Pi // x+ln(sqrt(2*Pi))
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P54 = FR_P54,FR_r3,FR_P10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf8,FR_S20
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_rf3 = FR_rf2,FR_rf,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fnma.s1       FR_P10f = FR_rf2,FR_05,FR_rf
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_TpNxLn2 = FR_N,FR_Ln2,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P54f = FR_P54f,FR_rf2,FR_P32f
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_InvX2 = FR_InvX,FR_InvX,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf8,FR_S12
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S4 = FR_S4,FR_Xf2,FR_S2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P54f = FR_P54f,FR_rf3,FR_P10f
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_TpNxLn2f = FR_Nf,FR_Ln2,FR_Tf
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnX = FR_TpNxLn2,f1,FR_P54
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_W2 = FR_W4,FR_InvX2,FR_W2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S28 = FR_S28,FR_Xf4,FR_S4
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnXf = FR_TpNxLn2f,f1,FR_P54f
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_LnX = FR_LnX,FR_Xm05,FR_LnSqrt2Pi
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_LnX = FR_InvX,FR_W2,FR_LnX
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_LnX = FR_S28,FR_Xf2,FR_LnX
      nop.i         0
};;
{ .mfb
      nop.m         0
      fms.d.s0      f8 = FR_LnX,f1,FR_LnXf
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for 0 <= x < 1
//---------------------------------------------------------------------
.align 32
lgamma_0_1:
{ .mfi
      ldfpd         FR_P5,FR_P4 = [GR_ad_1],16
      fms.s1        FR_x = FR_NormX,f1,f0 // x
      mov           GR_Arg025 = 0x3FD0
}
{ .mfi
      ldfpd         FR_P3,FR_P2 = [GR_ad_2],16
      nop.f         0
      add           GR_ad_Co = 0x1C40,GR_ad_Data
};;
{ .mfi
      ldfe          FR_Ln2 = [GR_ad_1],0x50
      nop.f         0
      // p6 if arg < 0.25
      cmp.lt        p6,p9 = GR_Arg,GR_Arg025
}
{ .mfi
      add           GR_ad_2 = 0x40,GR_ad_2
      nop.f         0
      mov           GR_Arg075 = 0x3FE8
};;
{ .mfi
      ldfpd         FR_Q8,FR_Q7 = [GR_ad_1],16
      fma.s1        FR_w2 = FR_w,FR_w,f0
      // p7 if 0.25 <= arg < 0.75
      // p8 if 0.75 <= arg < 1.0
(p9)  cmp.lt.unc    p7,p8 = GR_Arg,GR_Arg075
}
{ .mfi
      mov           GR_Arg0875 = 0x3FEC
      nop.f         0
      sub           GR_Exp = GR_Exp,GR_ExpBias
};;
{ .mfi
      ldfpd         FR_Q6,FR_Q5 = [GR_ad_2],16
      nop.f         0
(p8)  cmp.lt        p9,p0 = GR_Arg,GR_Arg0875
}
{ .mfi
      ldfpd         FR_Q4,FR_Q3 = [GR_ad_1],16
      nop.f         0
      add           GR_ad_Ce = 0x60,GR_ad_Co
};;
.pred.rel "mutex",p7,p8
{ .mfi
      ldfd          FR_Q2 = [GR_ad_2],16
      fms.s1        FR_r = FR_C,f8,f1
(p7)  mov           GR_Offs = 0xC0
}
{ .mfi
      setf.sig      FR_int_N = GR_Exp
      nop.f         0
(p8)  mov           GR_Offs = 0x180
};;
.pred.rel "mutex",p6,p7
{ .mfi
(p9)  add           GR_ad_Co = GR_Offs,GR_ad_Co
(p8)  fms.s1        FR_x = FR_NormX,f1,f1 // x-1
      nop.i         0
}
{ .mfi
(p9)  add           GR_ad_Ce = GR_Offs,GR_ad_Ce
(p7)  fms.s1        FR_x = FR_NormX,f1,FR_LocalMin // x-LocalMin
      cmp.lt        p10,p0 = GR_Arg,GR_Arg0875
};;
lgamma_common_0_2:
{ .mfi
      ldfpd         FR_A17,FR_A16 = [GR_ad_Co],16
      nop.f         0
      nop.i         0
}
{ .mfi
      ldfpd         FR_A15,FR_A14 = [GR_ad_Ce],16
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfpd         FR_A13,FR_A12 = [GR_ad_Co],16
      nop.f         0
(p10) extr.u        GR_Ind = GR_ArgAsIs,44,8
}
{ .mfi
      ldfpd         FR_A11,FR_A10 = [GR_ad_Ce],16
      nop.f         0
      nop.i         0
};;
{ .mfi
      ldfpd         FR_A9,FR_A8 = [GR_ad_Co],16
(p10) fnma.s1       FR_Q1 = FR_05,FR_w2,FR_w
      nop.i         0
}
{ .mfi
      ldfpd         FR_A7,FR_A6 = [GR_ad_Ce],16
(p10) fma.s1        FR_w3 = FR_w2,FR_w,f0
      nop.i         0
};;
{ .mfi
(p10) getf.exp      GR_SignExp_w = FR_w
(p10) fma.s1        FR_w4 = FR_w2,FR_w2,f0
      nop.i         0
}
{ .mfi
(p10) shladd        GR_ad_2 = GR_Ind,4,GR_ad_2
(p10) fma.s1        FR_r2 = FR_r,FR_r,f0
      nop.i         0
};;
{ .mfi
(p10) ldfe          FR_T = [GR_ad_2]
(p10) fma.s1        FR_P54 = FR_P5,FR_r,FR_P4
      nop.i         0
}
{ .mfi
      ldfe          FR_A5 = [GR_ad_Co],16
(p10) fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
};;
{ .mfi
      ldfe          FR_A4 = [GR_ad_Ce],16
      fma.s1        FR_x2 = FR_x,FR_x,f0
(p10) and           GR_Exp_w = GR_ExpMask, GR_SignExp_w
}
{ .mfi
      ldfe          FR_A3 = [GR_ad_Co],16
      nop.f         0
(p10) mov           GR_fff9 = 0xfff9
};;
//    p13 <== large w __libm_lgamma
//    p14 <== small w __libm_lgamma
{ .mfi
      ldfe          FR_A2 = [GR_ad_Ce],16
(p10) fma.s1        FR_Q8 = FR_Q8,FR_w,FR_Q7
(p10) cmp.ge.unc    p13,p14 = GR_Exp_w,GR_fff9
}
{ .mfi
      ldfe          FR_A1 = [GR_ad_Co],16
(p10) fma.s1        FR_Q6 = FR_Q6,FR_w,FR_Q5
      nop.i         0
};;
{ .mfi
      ldfe          FR_A0 = [GR_ad_Ce],16
(p10) fma.s1        FR_Q4 = FR_Q4,FR_w,FR_Q3
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_Q2 = FR_Q2,FR_w3,FR_Q1
      nop.i         0
};;
{ .mfi
      // set p11 if signgum is 32-bit int
      // set p12 if signgum is 64-bit int
      cmp.eq        p12,p11 = 8,r34
(p10) fma.s1        FR_r3 = FR_r2,FR_r,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fnma.s1       FR_P10 = FR_r2,FR_05,FR_r
      mov           GR_SignOfGamma = 1
};;
.pred.rel "mutex",p11,p12
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p11) st4           [r33] = GR_SignOfGamma
      fma.s1        FR_A17 = FR_A17,FR_x,FR_A16
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p12) st8           [r33] = GR_SignOfGamma
      fma.s1        FR_A15 = FR_A15,FR_x,FR_A14
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fcvt.xf       FR_N = FR_int_N
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_P54 = FR_P54,FR_r2,FR_P32
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A13 = FR_A13,FR_x,FR_A12
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A11 = FR_A11,FR_x,FR_A10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A9 = FR_A9,FR_x,FR_A8
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_x,FR_A6
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fma.s1        FR_Qlo = FR_Q8,FR_w2,FR_Q6
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_w6 = FR_w3,FR_w3,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fma.s1        FR_Qhi = FR_Q4,FR_w4,FR_Q2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A5 = FR_A5,FR_x,FR_A4
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fma.s1        FR_TpNxLn2 = FR_N,FR_Ln2,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_x,FR_A2
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fma.s1        FR_P54 = FR_P54,FR_r3,FR_P10
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A1,FR_x,FR_A0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A17 = FR_A17,FR_x2,FR_A15
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A13 = FR_A13,FR_x2,FR_A11
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A9 = FR_A9,FR_x2,FR_A7
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_LnX = FR_Qlo,FR_w6,FR_Qhi
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A5 = FR_A5,FR_x2,FR_A3
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_LnX = FR_TpNxLn2,f1,FR_P54
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A17 = FR_A17,FR_x4,FR_A13
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_x8 = FR_x4,FR_x4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A9 = FR_A9,FR_x4,FR_A5
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A17 = FR_A17,FR_x8,FR_A9
      nop.i         0
};;
{ .mfi
      nop.m         0
(p10) fms.s1        FR_A1 = FR_A1,f1,FR_LnX
      nop.i         0
};;
{ .mfb
      nop.m         0
      fma.d.s0      f8 = FR_A17,FR_x2,FR_A1
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for 1.0 <= x < 2.25
//---------------------------------------------------------------------
.align 32
lgamma_1_2:
{ .mfi
      add           GR_ad_Co = 0x10B0,GR_ad_1
      fcmp.eq.s1    p12,p0 = f1,FR_w
      mov           GR_Arg125 = 0x3FF4
}
{ .mfi
      add           GR_ad_Ce = 0x1110,GR_ad_1
      nop.f         0
      mov           GR_Arg175 = 0x3FFC
};;
{ .mfi
      mov           GR_SignOfGamma = 1
      fcmp.eq.s1    p13,p0 = f1,FR_NormX
      cmp.lt        p6,p9 = GR_Arg,GR_Arg125 // 1.0 <= x < 1.25
}
{ .mfi
      // set p10 if signgum is 32-bit int
      // set p11 if signgum is 64-bit int
      cmp.eq        p11,p10 = 8,r34
      nop.f         0
      cmp.ge        p8,p0 = GR_Arg,GR_Arg175 // x >= 1.75
};;
.pred.rel "mutex",p10,p11
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p10) st4           [r33] = GR_SignOfGamma
(p12) fma.d.s0      f8 = f0,f0,f0
(p9)  cmp.lt.unc    p7,p0 = GR_Arg,GR_Arg175 // 1.25 <= x < 1.75
}
{ .mib
      // store sign of gamma(x) as 64-bit int
(p11) st8           [r33] = GR_SignOfGamma
      mov           GR_Offs = 0
(p12) br.ret.spnt   b0 // fast exit for 2.0
};;
.pred.rel "mutex",p7,p8
{ .mfi
(p7)  mov           GR_Offs = 0xC0
(p7)  fms.s1        FR_x = FR_w,f1,FR_LocalMin
      nop.i         0
}
{ .mfb
(p8)  mov           GR_Offs = 0x180
(p13) fma.d.s0      f8 = f0,f0,f0
(p13) br.ret.spnt   b0 // fast exit for 1.0
};;
.pred.rel "mutex",p6,p8
{ .mfi
      add           GR_ad_Co = GR_ad_Co,GR_Offs
(p8)  fms.s1        FR_x = FR_w,f1,f1
      cmp.eq        p0,p10 = r0,r0
}
{ .mfb
      add           GR_ad_Ce = GR_ad_Ce,GR_Offs
(p6)  fma.s1        FR_x = f0,f0,FR_w
      br.cond.sptk  lgamma_common_0_2
};;
// branch for calculating of ln(GAMMA(x)) for -17 < x < 0
//---------------------------------------------------------------------
.align 32
lgamma_negrecursion:
{ .mfi
      getf.d        GR_ArgXfrAsIs = FR_Xfr
      fma.s1        FR_Xp2 = FR_2,f1,FR_NormX
      mov           GR_Arg05 = 0x3FE
}
{ .mfi
      add           GR_ad_Roots = 0x1390,GR_ad_1
      fma.s1        FR_NormX = FR_NormX,FR_Xfr,f0
      mov           GR_Arg075 = 0x3FE8
};;
{ .mfi
      getf.sig      GR_Sig = FR_int_Ntrunc
      fma.s1        FR_Xp3 = FR_2,f1,FR_Xp1
      shl           GR_Arg05 = GR_Arg05,52
}
{ .mfi
      mov           GR_Arg025 = 0x3FD0
      fma.s1        FR_Xp6 = FR_5,f1,FR_Xp1
      add           GR_ad_Co = 0x1C40,GR_ad_Data
};;
{ .mfi
      add           GR_ad_Dx = 8,GR_ad_Roots
      fma.s1        FR_Xp7 = FR_2,f1,FR_Xp5
      shr.u         GR_ArgXfr = GR_ArgXfrAsIs,48
}
{ .mfi
      add           GR_ad_Ce = 0x60,GR_ad_Co
      fma.s1        FR_Xp8 = FR_3,f1,FR_Xp5
      cmp.lt        p6,p0 = GR_ArgXfrAsIs,GR_Arg05
};;
{ .mfi
      and           GR_RootInd = 0xF,GR_Sig
      fma.s1        FR_Xp9 = FR_2,FR_2,FR_Xp5
      // p10 if arg < 0.25
      cmp.lt        p10,p14 = GR_ArgXfr,GR_Arg025
}
{ .mfi
(p6)  add           GR_ad_Roots = 0x120,GR_ad_Roots
      fma.s1        FR_Xp11 = f1,f1,FR_Xp10
(p6)  add           GR_ad_Dx = 0x120,GR_ad_Dx
};;
{ .mfi
      shladd        GR_ad_Root = GR_RootInd,4,GR_ad_Roots
      fma.s1        FR_Xp12 = FR_2,f1,FR_Xp10
      // p11 if 0.25 <= arg < 0.75
      // p12 if 0.75 <= arg < 1.0
(p14) cmp.lt.unc    p11,p12 = GR_ArgXfr,GR_Arg075
}
{ .mfi
      shladd        GR_ad_Dx = GR_RootInd,4,GR_ad_Dx
      fma.s1        FR_Xp13 = FR_3,f1,FR_Xp10
      cmp.eq        p0,p13 = 0,GR_Sig
};;
{ .mfi
      ld8           GR_Root = [GR_ad_Root]
      fma.s1        FR_Xp14 = FR_2,FR_2,FR_Xp10
(p12) mov           GR_Offs = 0x180
}
{ .mfi
      ldfd          FR_Root = [GR_ad_Root]
      fma.s1        FR_Xp15 = FR_5,f1,FR_Xp10
      and           GR_Sig = 0xF,GR_Sig
};;
{ .mfi
      ld8           GR_Dx = [GR_ad_Dx]
      fma.s1        FR_Xp16 = FR_3,FR_2,FR_Xp10
(p13) cmp.ge.unc    p6,p0 = 0xD,GR_Sig
}
{ .mfi
(p11) mov           GR_Offs = 0xC0
(p13) fma.s1        FR_NormX = FR_NormX,FR_Xp1,f0
(p13) cmp.ge.unc    p7,p0 = 0xB,GR_Sig
};;
{ .mfi
(p14) add           GR_ad_Co = GR_Offs,GR_ad_Co
(p6)  fma.s1        FR_Xp2 = FR_Xp2,FR_Xp3,f0
(p13) cmp.ge.unc    p8,p0 = 0x9,GR_Sig
}
{ .mfi
(p14) add           GR_ad_Ce = GR_Offs,GR_ad_Ce
(p7)  fma.s1        FR_Xp4 = FR_Xp4,FR_Xp5,f0
(p13) cmp.ge.unc    p9,p0 = 0x7,GR_Sig
};;
{ .mfi
      ldfpd         FR_B17,FR_B16 = [GR_ad_Co],16
(p8)  fma.s1        FR_Xp6 = FR_Xp6,FR_Xp7,f0
(p13) cmp.ge.unc    p6,p0 = 0x5,GR_Sig
}
{ .mfi
      ldfpd         FR_B15,FR_B14 = [GR_ad_Ce],16
(p9)  fma.s1        FR_Xp8 = FR_Xp8,FR_Xp9,f0
(p13) cmp.ge.unc    p7,p0 = 0x3,GR_Sig
};;
{ .mfi
      ldfpd         FR_B13,FR_B12 = [GR_ad_Co],16
(p6)  fma.s1        FR_Xp10 = FR_Xp10,FR_Xp11,f0
(p13) cmp.ge.unc    p8,p0 = 0x1,GR_Sig
}
{ .mfi
      ldfpd         FR_B11,FR_B10 = [GR_ad_Ce],16
(p7)  fma.s1        FR_Xp12 = FR_Xp12,FR_Xp13,f0
(p13) cmp.eq.unc    p9,p0 = 0,GR_Sig
};;
{ .mfi
      ldfpd         FR_B9,FR_B8 = [GR_ad_Co],16
(p8)  fma.s1        FR_Xp14 = FR_Xp14,FR_Xp15,f0
      mov           GR_Arg15 = 0xC02E // -15
}
{ .mfi
      ldfpd         FR_B7,FR_B6 = [GR_ad_Ce],16
      fcmp.eq.s1    p15,p0 = f0,FR_Xf
(p13) cmp.ge.unc    p6,p0 = 0xC,GR_Sig
};;
{ .mfi
      ldfe          FR_B5 = [GR_ad_Co],16
(p9)  fma.s1        FR_NormX = FR_NormX,FR_Xp16,f0
      sub           GR_Root = GR_ArgAsIs,GR_Root
}
{ .mfi
      sub           GR_RootInd = 0xE,GR_RootInd
(p11) fms.s1        FR_x = FR_Xfr,f1,FR_LocalMin // x-LocalMin
(p13) cmp.ge.unc    p7,p0 = 0x8,GR_Sig
};;
.pred.rel "mutex",p10,p12
{ .mfi
      ldfe          FR_B4 = [GR_ad_Ce],16
(p10) fms.s1        FR_x = FR_Xfr,f1,f0 // x
      add           GR_Root = GR_Root,GR_Dx
}
{ .mfb
      cmp.gtu       p14,p0 = 0xE,GR_RootInd
(p12) fms.s1        FR_x = FR_Xfr,f1,f1 // x-1
(p15) br.cond.spnt  lgamma_singularity
};;
{ .mfi
      ldfe          FR_B3 = [GR_ad_Co],16
(p6)  fma.s1        FR_Xp2 = FR_Xp2,FR_Xp4,f0
(p14) cmp.lt.unc    p11,p0 = GR_Arg,GR_Arg15
}
{ .mfi
      ldfe          FR_B2 = [GR_ad_Ce],16
(p7)  fma.s1        FR_Xp6 = FR_Xp6,FR_Xp8,f0
      add           GR_2xDx = GR_Dx,GR_Dx
};;
{ .mfi
      ldfe          FR_B1 = [GR_ad_Co],16
      fms.s1        FR_r = f8,f1,FR_Root
(p13) cmp.ge.unc    p6,p0 = 0x4,GR_Sig
}
{ .mib
      ldfe          FR_B0 = [GR_ad_Ce],16
(p11) cmp.leu.unc   p10,p0 = GR_Root,GR_2xDx
(p10) br.cond.spnt  lgamma_negroots
};;
{ .mfi
      ldfpd         FR_P5,FR_P4 = [GR_ad_1],16
(p6)  fma.s1        FR_Xp10 = FR_Xp10,FR_Xp12,f0
      tbit.z        p14,p15 = GR_Sig,0
}
{ .mfi
      ldfpd         FR_P3,FR_P2 = [GR_ad_2],16
      fnma.d.s0     FR_T = f1,f1,f8 //      nop.f         0

(p13) cmp.ge.unc    p7,p0 = 0x2,GR_Sig
};;
{ .mfi
      ldfe          FR_Ln2 = [GR_ad_1],0x50
(p7)  fma.s1        FR_NormX = FR_NormX,FR_Xp14,f0
      mov           GR_PseudoRoot = 0xBFFBC
}
{ .mlx
      add           GR_ad_2 = 0x40,GR_ad_2
      movl          GR_2xDx = 0x00002346DC5D6389
};;
{ .mfi
      ldfpd         FR_Q8,FR_Q7 = [GR_ad_1],16
      fma.s1        FR_x2 = FR_x,FR_x,f0
      shl           GR_PseudoRoot = GR_PseudoRoot,44
}
{ .mfi
      ldfpd         FR_Q6,FR_Q5 = [GR_ad_2],16
      fma.s1        FR_B17 = FR_B17,FR_x,FR_B16
(p13) cmp.ge.unc    p6,p0 = 0xA,GR_Sig
};;
{ .mfi
      ldfpd         FR_Q4,FR_Q3 = [GR_ad_1],16
(p6)  fma.s1        FR_Xp2 = FR_Xp2,FR_Xp6,f0
      sub           GR_PseudoRoot = GR_ArgAsIs,GR_PseudoRoot
}
{ .mfi
      ldfpd         FR_Q2,FR_Q1 = [GR_ad_2],16
      fma.s1        FR_B15 = FR_B15,FR_x,FR_B14
(p13) cmp.ge.unc    p7,p0 = 0x6,GR_Sig
};;
{ .mfi
      add           GR_ad_Co = 0x12F0,GR_ad_2
      fma.s1        FR_B13 = FR_B13,FR_x,FR_B12
      cmp.leu.unc   p10,p0 = GR_PseudoRoot,GR_2xDx
}
{ .mfi
      add           GR_ad_Ce = 0x1300,GR_ad_2
      fma.s1        FR_B11 = FR_B11,FR_x,FR_B10
      mov           GR_ExpMask = 0x1ffff
};;
{ .mfi
(p10) ldfe          FR_PR01 = [GR_ad_Co],0xF0
      fma.s1        FR_B9 = FR_B9,FR_x,FR_B8
      mov           GR_ExpBias = 0xFFFF
}
{ .mfb
(p10) ldfe          FR_PR11 = [GR_ad_Ce],0xF0
      fma.s1        FR_B7 = FR_B7,FR_x,FR_B6
(p10) br.cond.spnt  lgamma_pseudoroot
};;
{ .mfi
(p13) cmp.ge.unc    p6,p0 = 0xE,GR_Sig
(p7)  fma.s1        FR_NormX = FR_NormX,FR_Xp10,f0
      tbit.z.unc    p8,p0 = GR_Sig,0
}
{ .mfi
      mov           GR_SignOfGamma = 1
      fma.s1        FR_B5 = FR_B5,FR_x,FR_B4
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
};;
{ .mfi
      nop.m         0
      fma.s1        FR_B3 = FR_B3,FR_x,FR_B2
(p8)  sub           GR_SignOfGamma = r0,GR_SignOfGamma
}
{ .mfi
      nop.m         0
(p14) fms.s1        FR_w = f0,f0,f1
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
      fma.s1        FR_B1 = FR_B1,FR_x,FR_B0
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
      fma.s1        FR_B17 = FR_B17,FR_x2,FR_B15
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_B13 = FR_B13,FR_x2,FR_B11
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_B9 = FR_B9,FR_x2,FR_B7
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_NormX = FR_NormX,FR_Xp2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_B5 = FR_B5,FR_x2,FR_B3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_B17 = FR_B17,FR_x4,FR_B13
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_x8 = FR_x4,FR_x4,f0
      nop.i         0
};;
.pred.rel "mutex",p14,p15
{ .mfi
      nop.m         0
(p15) fms.s1        FR_w = FR_NormX,f1,f1
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fnma.s1       FR_w = FR_NormX,f1,FR_w
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_B9 = FR_B9,FR_x4,FR_B5
      nop.i         0
};;
{ .mfi
      nop.m         0
      frcpa.s1      FR_C,p0 = f1,FR_NormX
      nop.i         0
};;
{ .mfi
      getf.exp      GR_Exp = FR_NormX
      nop.f         0
      nop.i         0
};;
{ .mfi
      getf.d        GR_ArgAsIs = FR_NormX
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_w2 = FR_w,FR_w,f0
      nop.i         0
}
{ .mfi
      and           GR_Exp = GR_Exp,GR_ExpMask
      fma.s1        FR_Q8 = FR_Q8,FR_w,FR_Q7
      nop.i         0
};;
{ .mfi
      sub           GR_Exp = GR_Exp,GR_ExpBias
      fma.s1        FR_B17 = FR_B17,FR_x8,FR_B9
      extr.u        GR_Ind = GR_ArgAsIs,44,8
}
{ .mfi
      nop.m         0
      fma.s1        FR_Q6 = FR_Q6,FR_w,FR_Q5
      nop.i         0
};;
{ .mfi
      setf.sig      FR_int_N = GR_Exp
      fms.s1        FR_r = FR_C,FR_NormX,f1
      nop.i         0
}
{ .mfi
      shladd        GR_ad_2 = GR_Ind,4,GR_ad_2
      nop.f         0
      nop.i         0
};;
{ .mfi
      getf.exp      GR_SignExp_w = FR_w
      fma.s1        FR_Q4 = FR_Q4,FR_w,FR_Q3
      nop.i         0
}
{ .mfi
      ldfe          FR_T = [GR_ad_2]
      nop.f         0
      nop.i         0
};;
{ .mfi
      and           GR_Exp_w = GR_ExpMask, GR_SignExp_w
      fnma.s1       FR_Q1 = FR_05,FR_w2,FR_w
      mov           GR_fff9 = 0xfff9
}
{ .mfi
      nop.m         0
      fma.s1        FR_w3 = FR_w2,FR_w,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_w4 = FR_w2,FR_w2,f0
//    p13 <== large w __libm_lgamma
//    p14 <== small w __libm_lgamma
      cmp.ge        p13,p14 = GR_Exp_w,GR_fff9
}
{ .mfi
      nop.m         0
      fma.s1        FR_Qlo = FR_Q8,FR_w2,FR_Q6
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_r2 = FR_r,FR_r,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_B17 = FR_B17,FR_x2,FR_B1
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_P54 = FR_P5,FR_r,FR_P4
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_Q2 = FR_Q2,FR_w3,FR_Q1
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fma.s1        FR_w6 = FR_w3,FR_w3,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fcvt.xf       FR_N = FR_int_N
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_r3 = FR_r2,FR_r,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fnma.s1       FR_P10 = FR_r2,FR_05,FR_r
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_P54 = FR_P54,FR_r2,FR_P32
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_Qhi = FR_Q4,FR_w4,FR_Q2
      nop.i         0
}
{ .mfi
      nop.m         0
(p14) fnma.s1       FR_Qlo = FR_Qlo,FR_w6,FR_B17
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_TpNxLn2 = FR_N,FR_Ln2,FR_T
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_P54 = FR_P54,FR_r3,FR_P10
      nop.i         0
};;
.pred.rel "mutex",p13,p14
{ .mfi
      nop.m         0
(p14) fms.d.s0      f8 = FR_Qlo,f1,FR_Qhi
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_LnX = FR_TpNxLn2,f1,FR_P54
      nop.i         0
};;
{ .mfb
      nop.m         0
(p13) fms.d.s0      f8 = FR_B17,f1,FR_LnX
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) near negative roots
//---------------------------------------------------------------------
.align 32
lgamma_negroots:
{ .mfi
      shladd        GR_Offs = GR_RootInd,3,r0 //GR_RootInd*8
      fma.s1        FR_r2 = FR_r,FR_r,f0
      add           GR_ad_Co = 0x15C0,GR_ad_1//0x1590,GR_ad_1
}
{ .mfi
      add           GR_ad_Ce = 0x1610,GR_ad_1//0x15E0,GR_ad_1
      nop.f         0
      cmp.lt        p6,p0 = GR_ArgXfrAsIs,GR_Arg05
};;
{ .mfi
      add           GR_ad_Roots = 0x10A0,GR_ad_1
      nop.f         0
(p6)  add           GR_ad_Co = 0x820,GR_ad_Co
}
{ .mfi
(p6)  add           GR_ad_Ce = 0x820,GR_ad_Ce
      nop.f         0
      shladd        GR_Offs = GR_RootInd,1,GR_Offs //GR_RootInd*10
};;
{ .mmi
      shladd        GR_ad_Co = GR_Offs,4,GR_ad_Co
      shladd        GR_ad_Ce = GR_Offs,4,GR_ad_Ce
      cmp.eq        p8,p7 = r0,r0
};;
{ .mmi
      ldfpd         FR_A15,FR_A14 = [GR_ad_Co],16
      ldfpd         FR_A13,FR_A12 = [GR_ad_Ce],16
      mov           GR_SignOfGamma = 1
};;
{ .mmi
      ldfpd         FR_A11,FR_A10 = [GR_ad_Co],16
      ldfpd         FR_A9,FR_A8   = [GR_ad_Ce],16
(p6)  cmp.eq        p7,p8 = r0,GR_RootInd
};;
{ .mmi
      ldfpd         FR_A7,FR_A6 = [GR_ad_Co],16
      ldfpd         FR_A5,FR_A4 = [GR_ad_Ce],16
      tbit.z        p11,p0 = GR_Sig,0
};;
{ .mmi
      ldfe          FR_A3 = [GR_ad_Co],16
      ldfe          FR_A2 = [GR_ad_Ce],16
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
};;
{ .mmi
      ldfe          FR_A1 = [GR_ad_Co],16
      ldfe          FR_A0 = [GR_ad_Ce],16
(p11) sub           GR_SignOfGamma = r0,GR_SignOfGamma
};;
{ .mfi
      ldfe          FR_A00 = [GR_ad_Roots]
      fma.s1        FR_r4 = FR_r2,FR_r2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r,FR_A14
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A13 = FR_A13,FR_r,FR_A12
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
      fma.s1        FR_A11 = FR_A11,FR_r,FR_A10
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
      fma.s1        FR_A9 = FR_A9,FR_r,FR_A8
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
      fma.s1        FR_r8 = FR_r4,FR_r4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A1,FR_r,FR_A0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r2,FR_A13
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A11 = FR_A11,FR_r2,FR_A9
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r2,FR_A5
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A3,FR_r2,FR_A1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A15 = FR_A15,FR_r4,FR_A11
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A7,FR_r4,FR_A3
      nop.i         0
};;
.pred.rel "mutex",p7,p8
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_A1 = FR_A15,FR_r8,FR_A7
      nop.i         0
}
{ .mfi
      nop.m         0
(p8)  fma.d.s0      f8 = FR_A15,FR_r8,FR_A7
      nop.i         0
};;
{ .mfb
      nop.m         0
(p7)  fma.d.s0      f8 = FR_A1,FR_r,FR_A00
      br.ret.sptk   b0
};;
// branch for handling pseudo root on (-2;-1)
//---------------------------------------------------------------------
.align 32
lgamma_pseudoroot:
{ .mmi
      ldfe          FR_PR21 = [GR_ad_Co],32
      ldfe          FR_PR31 = [GR_ad_Ce],32
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
};;
{ .mmi
      ldfe          FR_PR00 = [GR_ad_Co],32
      ldfe          FR_PR10 = [GR_ad_Ce],0xF0
      mov           GR_SignOfGamma = 1
};;
{ .mmi
      ldfe          FR_PR20 = [GR_ad_Co],0xF0
      ldfe          FR_PR30 = [GR_ad_Ce]
      tbit.z        p8,p0 = GR_Sig,0
};;
{ .mfi
      ldfe          FR_PRN = [GR_ad_Co]
      fma.s1        FR_PR01 = f8,f1,FR_PR01
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_PR11 = f8,f1,FR_PR11
(p8)  sub           GR_SignOfGamma = r0,GR_SignOfGamma
};;
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
      fma.s1        FR_PR21 = f8,f1,FR_PR21
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
      fma.s1        FR_PR31 = f8,f1,FR_PR31
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_PR01 = f8,FR_PR01,FR_PR00
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_PR11 = f8,FR_PR11,FR_PR10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_PR21 = f8,FR_PR21,FR_PR20
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_PR31 = f8,FR_PR31,FR_PR30
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_PR01 = FR_PR11,FR_PR01,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_PR21 = FR_PR31,FR_PR21,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_PR01 = FR_PR21,FR_PR01,f0
      nop.i         0
};;
{ .mfb
      nop.m         0
      fma.d.s0      f8 = FR_PR01,FR_PRN,f0
      br.ret.sptk   b0
};;
// branch for handling +/-0, NaT, QNaN, +/-INF and denormalised numbers
//---------------------------------------------------------------------
.align 32
lgamma_spec:
{ .mfi
      getf.exp      GR_SignExp = FR_NormX
      fclass.m      p6,p0 = f8,0x21 // is arg +INF?
      mov           GR_SignOfGamma = 1
};;
{ .mfi
      getf.sig      GR_ArgAsIs = FR_NormX
      fclass.m      p7,p0 = f8,0xB // is x deno?
      // set p11 if signgum is 32-bit int
      // set p12 if signgum is 64-bit int
      cmp.eq        p12,p11 = 8,r34
};;
.pred.rel "mutex",p11,p12
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p11) st4           [r33] = GR_SignOfGamma
      fclass.m      p8,p0 = f8,0x1C0 // is arg NaT or NaN?
      dep.z         GR_Ind = GR_SignExp,8,4
}
{ .mib
      // store sign of gamma(x) as 64-bit int
(p12) st8           [r33] = GR_SignOfGamma
      cmp.lt        p10,p0 = GR_SignExp,GR_ExpBias
(p6)  br.ret.spnt   b0 // exit for +INF
};;
{ .mfi
      and           GR_Exp = GR_SignExp,GR_ExpMask
      fclass.m      p9,p0 = f8,0x22 // is arg -INF?
      nop.i         0
};;
{ .mfi
      add           GR_ad_Co = GR_Ind,GR_ad_Data
(p7)  fma.s0        FR_tmp = f8,f8,f8
      extr.u        GR_ArgAsIs = GR_ArgAsIs,11,52
}
{ .mfb
      nop.m         0
(p8)  fms.d.s0      f8 = f8,f1,f8
(p8)  br.ret.spnt   b0 // exit for NaT and NaN
};;
{ .mib
      nop.m         0
      shr.u         GR_Arg = GR_ArgAsIs,48
(p7)  br.cond.sptk  lgamma_common
};;
{ .mfb
      nop.m         0
(p9)  fmerge.s      f8 = f1,f8
(p9)  br.ret.spnt   b0 // exit -INF
};;
// branch for handling negative integers and +/-0
//---------------------------------------------------------------------
.align 32
lgamma_singularity:
{ .mfi
      mov           GR_ad_SignGam = r33
      fclass.m      p6,p0 = f8, 0x6 // is x -0?
      mov           GR_SignOfGamma = 1
}
{ .mfi
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
      fma.s1        FR_X = f0,f0,f8
      nop.i         0
};;
{ .mfi
      nop.m         0
      frcpa.s0      f8,p0 = f1,f0
      mov           GR_TAG = 106 // negative
}
{ .mib
      nop.m         0
(p6)  sub           GR_SignOfGamma = r0,GR_SignOfGamma
      br.cond.sptk  lgamma_libm_err
};;
// overflow (x > OVERFLOV_BOUNDARY)
//---------------------------------------------------------------------
.align 32
lgamma_overflow:
{ .mfi
      mov           GR_SignOfGamma = 1
      nop.f         0
      mov           r8 = 0x1FFFE
};;
{ .mfi
      setf.exp      f9 = r8
      fmerge.s      FR_X = f8,f8
      mov           GR_TAG = 105 // overflow
};;
{ .mfi
      mov           GR_ad_SignGam = r33
      nop.f         0
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
}
{ .mfi
      nop.m         0
      fma.d.s0      f8 = f9,f9,f0 // Set I,O and +INF result
      nop.i         0
};;
//
//---------------------------------------------------------------------
.align 32
lgamma_libm_err:
{ .mmi
      alloc         r32 = ar.pfs,1,4,4,0
      mov           GR_Parameter_TAG = GR_TAG
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mmi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [GR_ad_SignGam] = GR_SignOfGamma
      // store sign of gamma(x) as 64-bit int
(p10) st8           [GR_ad_SignGam] = GR_SignOfGamma
      nop.i         0
};;
GLOBAL_LIBM_END(__libm_lgamma)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov   GR_SAVE_PFS=ar.pfs                // Save ar.pfs
}
{ .mfi
.fframe 64
        add   sp=-64,sp                         // Create new stack
        nop.f 0
        mov   GR_SAVE_GP=gp                     // Save gp
};;
{ .mmi
        stfd [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfd [GR_Parameter_X] = FR_X                  // STORE Parameter 1
                                                      // on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3
                                                      // on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#         // Call error handling
                                                      // function
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
