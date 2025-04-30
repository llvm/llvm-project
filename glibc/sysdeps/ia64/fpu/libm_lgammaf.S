.file "libm_lgammaf.s"


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
// 01/25/02  Corrected parameter store, load, and tag for __libm_error_support
// 02/01/02  Added support of SIGN(GAMMA(x)) calculation
// 05/20/02  Cleaned up namespace and sf0 syntax
// 09/16/02  Improved accuracy on intervals reduced to [1;1.25]
// 10/21/02  Now it returns SIGN(GAMMA(x))=-1 for negative zero
// 02/10/03  Reordered header: .section, .global, .proc, .align
// 07/22/03  Reformatted some data tables
// 03/31/05  Reformatted delimiters between data tables
//
//*********************************************************************
//
//*********************************************************************
//
// Function: __libm_lgammaf(float x, int* signgam, int szsigngam)
// computes the principle value of the logarithm of the GAMMA function
// of x. Signum of GAMMA(x) is stored to memory starting at the address
// specified by the signgam.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f6-f15
//                              f32-f97
//
//    General Purpose Registers:
//      r8-r11
//      r14-r30
//      r32-r36
//      r37-r40 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    lgamma(+inf) = +inf
//    lgamma(-inf) = +inf
//    lgamma(+/-0) = +inf
//    lgamma(x<0, x - integer) = +inf
//    lgamma(SNaN) = QNaN
//    lgamma(QNaN) = QNaN
//
//*********************************************************************
//
// Overview
//
// The method consists of three cases.
//
// If      2^13 <= x < OVERFLOW_BOUNDARY  use case lgammaf_pstirling;
// else if 1 < x < 2^13                   use case lgammaf_regular;
// else if -9 < x < 1                     use case lgammaf_negrecursion;
// else if -2^13 <  x < -9                use case lgammaf_negpoly;
// else if x < -2^13                      use case lgammaf_negstirling;
// else if x is close to negative
//         roots of ln(GAMMA(x))          use case lgammaf_negroots;
//
//
// Case 2^13 <= x < OVERFLOW_BOUNDARY
// ----------------------------------
//   Here we use algorithm based on the Stirling formula:
//     ln(GAMMA(x)) = ln(sqrt(2*Pi)) + (x-0.5)*ln(x) - x
//
// Case 1 < x < 2^13
// -----------------
//   To calculate ln(GAMMA(x)) for such arguments we use polynomial
//   approximation on following intervals: [1.0; 1.25), [1.25; 1.5),
//   [1.5, 1.75), [1.75; 2), [2; 4), [2^i; 2^(i+1)), i=1..8
//
//   Following variants of approximation and argument reduction are used:
//    1. [1.0; 1.25)
//       ln(GAMMA(x)) ~ (x-1.0)*P7(x)
//
//    2. [1.25; 1.5)
//       ln(GAMMA(x)) ~ ln(GAMMA(x0))+(x-x0)*P8(x-x0),
//       where x0 - point of local minimum on [1;2] rounded to nearest double
//       precision number.
//
//    3. [1.5; 1.75)
//       ln(GAMMA(x)) ~ P8(x)
//
//    4. [1.75; 2.0)
//       ln(GAMMA(x)) ~ (x-2)*P7(x)
//
//    5. [2; 4)
//       ln(GAMMA(x)) ~ (x-2)*P10(x)
//
//    6. [2^i; 2^(i+1)), i=2..8
//       ln(GAMMA(x)) ~ P10((x-2^i)/2^i)
//
// Case -9 < x < 1
// ---------------
//   Here we use the recursive formula:
//   ln(GAMMA(x)) = ln(GAMMA(x+1)) - ln(x)
//
//   Using this formula we reduce argument to base interval [1.0; 2.0]
//
// Case -2^13 < x < -9
// --------------------
//   Here we use the formula:
//   ln(GAMMA(x)) = ln(Pi/(|x|*GAMMA(|x|)*sin(Pi*|x|))) =
//   = -ln(|x|) - ln((GAMMA(|x|)) - ln(sin(Pi*r)/(Pi*r)) - ln(|r|)
//   where r = x - rounded_to_nearest(x), i.e |r| <= 0.5 and
//   ln(sin(Pi*r)/(Pi*r)) is approximated by 8-degree polynomial of r^2
//
// Case x < -2^13
// --------------
//   Here we use algorithm based on the Stirling formula:
//   ln(GAMMA(x)) = -ln(sqrt(2*Pi)) + (|x|-0.5)ln(x) - |x| -
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
//
// Claculation of logarithm
// ------------------------
//   Consider  x = 2^N * xf so
//   ln(x) = ln(frcpa(x)*x/frcpa(x))
//         = ln(1/frcpa(x)) + ln(frcpa(x)*x)
//
//   frcpa(x) = 2^(-N) * frcpa(xf)
//
//   ln(1/frcpa(x)) = -ln(2^(-N)) - ln(frcpa(xf))
//                  = N*ln(2) - ln(frcpa(xf))
//                  = N*ln(2) + ln(1/frcpa(xf))
//
//   ln(x) = ln(1/frcpa(x)) + ln(frcpa(x)*x) =
//         = N*ln(2) + ln(1/frcpa(xf)) + ln(frcpa(x)*x)
//         = N*ln(2) + T + ln(frcpa(x)*x)
//
//   Let r = 1 - frcpa(x)*x, note that r is quite small by
//   absolute value so
//
//   ln(x) = N*ln(2) + T + ln(1+r) ~ N*ln(2) + T + Series(r),
//   where T - is precomputed tabular value,
//   Series(r) = (P3*r + P2)*r^2 + (P1*r + 1)
//
//*********************************************************************

GR_TAG                 = r8
GR_ad_Data             = r8
GR_ad_Co               = r9
GR_ad_SignGam          = r10
GR_ad_Ce               = r10
GR_SignExp             = r11

GR_ad_C650             = r14
GR_ad_RootCo           = r14
GR_ad_C0               = r15
GR_Dx                  = r15
GR_Ind                 = r16
GR_Offs                = r17
GR_IntNum              = r17
GR_ExpBias             = r18
GR_ExpMask             = r19
GR_Ind4T               = r20
GR_RootInd             = r20
GR_Sig                 = r21
GR_Exp                 = r22
GR_PureExp             = r23
GR_ad_C43              = r24
GR_StirlBound          = r25
GR_ad_T                = r25
GR_IndX8               = r25
GR_Neg2                = r25
GR_2xDx                = r25
GR_SingBound           = r26
GR_IndX2               = r26
GR_Neg4                = r26
GR_ad_RootCe           = r26
GR_Arg                 = r27
GR_ExpOf2              = r28
GR_fff7                = r28
GR_Root                = r28
GR_ReqBound            = r28
GR_N                   = r29
GR_ad_Root             = r30
GR_ad_OvfBound         = r30
GR_SignOfGamma         = r31

GR_SAVE_B0             = r33
GR_SAVE_PFS            = r34
GR_SAVE_GP             = r35
GR_SAVE_SP             = r36

GR_Parameter_X         = r37
GR_Parameter_Y         = r38
GR_Parameter_RESULT    = r39
GR_Parameter_TAG       = r40

//*********************************************************************

FR_X                   = f10
FR_Y                   = f1 // lgammaf is single argument function
FR_RESULT              = f8

FR_x                   = f6
FR_x2                  = f7

FR_x3                  = f9
FR_x4                  = f10
FR_xm2                 = f11
FR_w                   = f11
FR_w2                  = f12
FR_Q32                 = f13
FR_Q10                 = f14
FR_InvX                = f15

FR_NormX               = f32

FR_A0                  = f33
FR_A1                  = f34
FR_A2                  = f35
FR_A3                  = f36
FR_A4                  = f37
FR_A5                  = f38
FR_A6                  = f39
FR_A7                  = f40
FR_A8                  = f41
FR_A9                  = f42
FR_A10                 = f43

FR_int_N               = f44
FR_P3                  = f45
FR_P2                  = f46
FR_P1                  = f47
FR_LocalMin            = f48
FR_Ln2                 = f49
FR_05                  = f50
FR_LnSqrt2Pi           = f51
FR_3                   = f52
FR_r                   = f53
FR_r2                  = f54
FR_T                   = f55
FR_N                   = f56
FR_xm05                = f57
FR_int_Ln              = f58
FR_P32                 = f59
FR_P10                 = f60

FR_Xf                  = f61
FR_InvXf               = f62
FR_rf                  = f63
FR_rf2                 = f64
FR_Tf                  = f65
FR_Nf                  = f66
FR_xm05f               = f67
FR_P32f                = f68
FR_P10f                = f69
FR_Lnf                 = f70
FR_Xf2                 = f71
FR_Xf4                 = f72
FR_Xf8                 = f73
FR_Ln                  = f74
FR_xx                  = f75
FR_Root                = f75
FR_Req                 = f76
FR_1pXf                = f77

FR_S16                 = f78
FR_R3                  = f78
FR_S14                 = f79
FR_R2                  = f79
FR_S12                 = f80
FR_R1                  = f80
FR_S10                 = f81
FR_R0                  = f81
FR_S8                  = f82
FR_rx                  = f82
FR_S6                  = f83
FR_rx2                 = f84
FR_S4                  = f84
FR_S2                  = f85

FR_Xp1                 = f86
FR_Xp2                 = f87
FR_Xp3                 = f88
FR_Xp4                 = f89
FR_Xp5                 = f90
FR_Xp6                 = f91
FR_Xp7                 = f92
FR_Xp8                 = f93
FR_OverflowBound       = f93

FR_2                   = f94
FR_tmp                 = f95
FR_int_Ntrunc          = f96
FR_Ntrunc              = f97

//*********************************************************************

RODATA
.align 32
LOCAL_OBJECT_START(lgammaf_data)
log_table_1:
data8 0xbfd0001008f39d59 // P3
data8 0x3fd5556073e0c45a // P2
data8 0x3fe62e42fefa39ef // ln(2)
data8 0x3fe0000000000000 // 0.5
//
data8 0x3F60040155D5889E //ln(1/frcpa(1+   0/256)
data8 0x3F78121214586B54 //ln(1/frcpa(1+   1/256)
data8 0x3F841929F96832F0 //ln(1/frcpa(1+   2/256)
data8 0x3F8C317384C75F06 //ln(1/frcpa(1+   3/256)
data8 0x3F91A6B91AC73386 //ln(1/frcpa(1+   4/256)
data8 0x3F95BA9A5D9AC039 //ln(1/frcpa(1+   5/256)
data8 0x3F99D2A8074325F4 //ln(1/frcpa(1+   6/256)
data8 0x3F9D6B2725979802 //ln(1/frcpa(1+   7/256)
data8 0x3FA0C58FA19DFAAA //ln(1/frcpa(1+   8/256)
data8 0x3FA2954C78CBCE1B //ln(1/frcpa(1+   9/256)
data8 0x3FA4A94D2DA96C56 //ln(1/frcpa(1+  10/256)
data8 0x3FA67C94F2D4BB58 //ln(1/frcpa(1+  11/256)
data8 0x3FA85188B630F068 //ln(1/frcpa(1+  12/256)
data8 0x3FAA6B8ABE73AF4C //ln(1/frcpa(1+  13/256)
data8 0x3FAC441E06F72A9E //ln(1/frcpa(1+  14/256)
data8 0x3FAE1E6713606D07 //ln(1/frcpa(1+  15/256)
data8 0x3FAFFA6911AB9301 //ln(1/frcpa(1+  16/256)
data8 0x3FB0EC139C5DA601 //ln(1/frcpa(1+  17/256)
data8 0x3FB1DBD2643D190B //ln(1/frcpa(1+  18/256)
data8 0x3FB2CC7284FE5F1C //ln(1/frcpa(1+  19/256)
data8 0x3FB3BDF5A7D1EE64 //ln(1/frcpa(1+  20/256)
data8 0x3FB4B05D7AA012E0 //ln(1/frcpa(1+  21/256)
data8 0x3FB580DB7CEB5702 //ln(1/frcpa(1+  22/256)
data8 0x3FB674F089365A7A //ln(1/frcpa(1+  23/256)
data8 0x3FB769EF2C6B568D //ln(1/frcpa(1+  24/256)
data8 0x3FB85FD927506A48 //ln(1/frcpa(1+  25/256)
data8 0x3FB9335E5D594989 //ln(1/frcpa(1+  26/256)
data8 0x3FBA2B0220C8E5F5 //ln(1/frcpa(1+  27/256)
data8 0x3FBB0004AC1A86AC //ln(1/frcpa(1+  28/256)
data8 0x3FBBF968769FCA11 //ln(1/frcpa(1+  29/256)
data8 0x3FBCCFEDBFEE13A8 //ln(1/frcpa(1+  30/256)
data8 0x3FBDA727638446A2 //ln(1/frcpa(1+  31/256)
data8 0x3FBEA3257FE10F7A //ln(1/frcpa(1+  32/256)
data8 0x3FBF7BE9FEDBFDE6 //ln(1/frcpa(1+  33/256)
data8 0x3FC02AB352FF25F4 //ln(1/frcpa(1+  34/256)
data8 0x3FC097CE579D204D //ln(1/frcpa(1+  35/256)
data8 0x3FC1178E8227E47C //ln(1/frcpa(1+  36/256)
data8 0x3FC185747DBECF34 //ln(1/frcpa(1+  37/256)
data8 0x3FC1F3B925F25D41 //ln(1/frcpa(1+  38/256)
data8 0x3FC2625D1E6DDF57 //ln(1/frcpa(1+  39/256)
data8 0x3FC2D1610C86813A //ln(1/frcpa(1+  40/256)
data8 0x3FC340C59741142E //ln(1/frcpa(1+  41/256)
data8 0x3FC3B08B6757F2A9 //ln(1/frcpa(1+  42/256)
data8 0x3FC40DFB08378003 //ln(1/frcpa(1+  43/256)
data8 0x3FC47E74E8CA5F7C //ln(1/frcpa(1+  44/256)
data8 0x3FC4EF51F6466DE4 //ln(1/frcpa(1+  45/256)
data8 0x3FC56092E02BA516 //ln(1/frcpa(1+  46/256)
data8 0x3FC5D23857CD74D5 //ln(1/frcpa(1+  47/256)
data8 0x3FC6313A37335D76 //ln(1/frcpa(1+  48/256)
data8 0x3FC6A399DABBD383 //ln(1/frcpa(1+  49/256)
data8 0x3FC70337DD3CE41B //ln(1/frcpa(1+  50/256)
data8 0x3FC77654128F6127 //ln(1/frcpa(1+  51/256)
data8 0x3FC7E9D82A0B022D //ln(1/frcpa(1+  52/256)
data8 0x3FC84A6B759F512F //ln(1/frcpa(1+  53/256)
data8 0x3FC8AB47D5F5A310 //ln(1/frcpa(1+  54/256)
data8 0x3FC91FE49096581B //ln(1/frcpa(1+  55/256)
data8 0x3FC981634011AA75 //ln(1/frcpa(1+  56/256)
data8 0x3FC9F6C407089664 //ln(1/frcpa(1+  57/256)
data8 0x3FCA58E729348F43 //ln(1/frcpa(1+  58/256)
data8 0x3FCABB55C31693AD //ln(1/frcpa(1+  59/256)
data8 0x3FCB1E104919EFD0 //ln(1/frcpa(1+  60/256)
data8 0x3FCB94EE93E367CB //ln(1/frcpa(1+  61/256)
data8 0x3FCBF851C067555F //ln(1/frcpa(1+  62/256)
data8 0x3FCC5C0254BF23A6 //ln(1/frcpa(1+  63/256)
data8 0x3FCCC000C9DB3C52 //ln(1/frcpa(1+  64/256)
data8 0x3FCD244D99C85674 //ln(1/frcpa(1+  65/256)
data8 0x3FCD88E93FB2F450 //ln(1/frcpa(1+  66/256)
data8 0x3FCDEDD437EAEF01 //ln(1/frcpa(1+  67/256)
data8 0x3FCE530EFFE71012 //ln(1/frcpa(1+  68/256)
data8 0x3FCEB89A1648B971 //ln(1/frcpa(1+  69/256)
data8 0x3FCF1E75FADF9BDE //ln(1/frcpa(1+  70/256)
data8 0x3FCF84A32EAD7C35 //ln(1/frcpa(1+  71/256)
data8 0x3FCFEB2233EA07CD //ln(1/frcpa(1+  72/256)
data8 0x3FD028F9C7035C1C //ln(1/frcpa(1+  73/256)
data8 0x3FD05C8BE0D9635A //ln(1/frcpa(1+  74/256)
data8 0x3FD085EB8F8AE797 //ln(1/frcpa(1+  75/256)
data8 0x3FD0B9C8E32D1911 //ln(1/frcpa(1+  76/256)
data8 0x3FD0EDD060B78081 //ln(1/frcpa(1+  77/256)
data8 0x3FD122024CF0063F //ln(1/frcpa(1+  78/256)
data8 0x3FD14BE2927AECD4 //ln(1/frcpa(1+  79/256)
data8 0x3FD180618EF18ADF //ln(1/frcpa(1+  80/256)
data8 0x3FD1B50BBE2FC63B //ln(1/frcpa(1+  81/256)
data8 0x3FD1DF4CC7CF242D //ln(1/frcpa(1+  82/256)
data8 0x3FD214456D0EB8D4 //ln(1/frcpa(1+  83/256)
data8 0x3FD23EC5991EBA49 //ln(1/frcpa(1+  84/256)
data8 0x3FD2740D9F870AFB //ln(1/frcpa(1+  85/256)
data8 0x3FD29ECDABCDFA04 //ln(1/frcpa(1+  86/256)
data8 0x3FD2D46602ADCCEE //ln(1/frcpa(1+  87/256)
data8 0x3FD2FF66B04EA9D4 //ln(1/frcpa(1+  88/256)
data8 0x3FD335504B355A37 //ln(1/frcpa(1+  89/256)
data8 0x3FD360925EC44F5D //ln(1/frcpa(1+  90/256)
data8 0x3FD38BF1C3337E75 //ln(1/frcpa(1+  91/256)
data8 0x3FD3C25277333184 //ln(1/frcpa(1+  92/256)
data8 0x3FD3EDF463C1683E //ln(1/frcpa(1+  93/256)
data8 0x3FD419B423D5E8C7 //ln(1/frcpa(1+  94/256)
data8 0x3FD44591E0539F49 //ln(1/frcpa(1+  95/256)
data8 0x3FD47C9175B6F0AD //ln(1/frcpa(1+  96/256)
data8 0x3FD4A8B341552B09 //ln(1/frcpa(1+  97/256)
data8 0x3FD4D4F3908901A0 //ln(1/frcpa(1+  98/256)
data8 0x3FD501528DA1F968 //ln(1/frcpa(1+  99/256)
data8 0x3FD52DD06347D4F6 //ln(1/frcpa(1+ 100/256)
data8 0x3FD55A6D3C7B8A8A //ln(1/frcpa(1+ 101/256)
data8 0x3FD5925D2B112A59 //ln(1/frcpa(1+ 102/256)
data8 0x3FD5BF406B543DB2 //ln(1/frcpa(1+ 103/256)
data8 0x3FD5EC433D5C35AE //ln(1/frcpa(1+ 104/256)
data8 0x3FD61965CDB02C1F //ln(1/frcpa(1+ 105/256)
data8 0x3FD646A84935B2A2 //ln(1/frcpa(1+ 106/256)
data8 0x3FD6740ADD31DE94 //ln(1/frcpa(1+ 107/256)
data8 0x3FD6A18DB74A58C5 //ln(1/frcpa(1+ 108/256)
data8 0x3FD6CF31058670EC //ln(1/frcpa(1+ 109/256)
data8 0x3FD6F180E852F0BA //ln(1/frcpa(1+ 110/256)
data8 0x3FD71F5D71B894F0 //ln(1/frcpa(1+ 111/256)
data8 0x3FD74D5AEFD66D5C //ln(1/frcpa(1+ 112/256)
data8 0x3FD77B79922BD37E //ln(1/frcpa(1+ 113/256)
data8 0x3FD7A9B9889F19E2 //ln(1/frcpa(1+ 114/256)
data8 0x3FD7D81B037EB6A6 //ln(1/frcpa(1+ 115/256)
data8 0x3FD8069E33827231 //ln(1/frcpa(1+ 116/256)
data8 0x3FD82996D3EF8BCB //ln(1/frcpa(1+ 117/256)
data8 0x3FD85855776DCBFB //ln(1/frcpa(1+ 118/256)
data8 0x3FD8873658327CCF //ln(1/frcpa(1+ 119/256)
data8 0x3FD8AA75973AB8CF //ln(1/frcpa(1+ 120/256)
data8 0x3FD8D992DC8824E5 //ln(1/frcpa(1+ 121/256)
data8 0x3FD908D2EA7D9512 //ln(1/frcpa(1+ 122/256)
data8 0x3FD92C59E79C0E56 //ln(1/frcpa(1+ 123/256)
data8 0x3FD95BD750EE3ED3 //ln(1/frcpa(1+ 124/256)
data8 0x3FD98B7811A3EE5B //ln(1/frcpa(1+ 125/256)
data8 0x3FD9AF47F33D406C //ln(1/frcpa(1+ 126/256)
data8 0x3FD9DF270C1914A8 //ln(1/frcpa(1+ 127/256)
data8 0x3FDA0325ED14FDA4 //ln(1/frcpa(1+ 128/256)
data8 0x3FDA33440224FA79 //ln(1/frcpa(1+ 129/256)
data8 0x3FDA57725E80C383 //ln(1/frcpa(1+ 130/256)
data8 0x3FDA87D0165DD199 //ln(1/frcpa(1+ 131/256)
data8 0x3FDAAC2E6C03F896 //ln(1/frcpa(1+ 132/256)
data8 0x3FDADCCC6FDF6A81 //ln(1/frcpa(1+ 133/256)
data8 0x3FDB015B3EB1E790 //ln(1/frcpa(1+ 134/256)
data8 0x3FDB323A3A635948 //ln(1/frcpa(1+ 135/256)
data8 0x3FDB56FA04462909 //ln(1/frcpa(1+ 136/256)
data8 0x3FDB881AA659BC93 //ln(1/frcpa(1+ 137/256)
data8 0x3FDBAD0BEF3DB165 //ln(1/frcpa(1+ 138/256)
data8 0x3FDBD21297781C2F //ln(1/frcpa(1+ 139/256)
data8 0x3FDC039236F08819 //ln(1/frcpa(1+ 140/256)
data8 0x3FDC28CB1E4D32FD //ln(1/frcpa(1+ 141/256)
data8 0x3FDC4E19B84723C2 //ln(1/frcpa(1+ 142/256)
data8 0x3FDC7FF9C74554C9 //ln(1/frcpa(1+ 143/256)
data8 0x3FDCA57B64E9DB05 //ln(1/frcpa(1+ 144/256)
data8 0x3FDCCB130A5CEBB0 //ln(1/frcpa(1+ 145/256)
data8 0x3FDCF0C0D18F326F //ln(1/frcpa(1+ 146/256)
data8 0x3FDD232075B5A201 //ln(1/frcpa(1+ 147/256)
data8 0x3FDD490246DEFA6B //ln(1/frcpa(1+ 148/256)
data8 0x3FDD6EFA918D25CD //ln(1/frcpa(1+ 149/256)
data8 0x3FDD9509707AE52F //ln(1/frcpa(1+ 150/256)
data8 0x3FDDBB2EFE92C554 //ln(1/frcpa(1+ 151/256)
data8 0x3FDDEE2F3445E4AF //ln(1/frcpa(1+ 152/256)
data8 0x3FDE148A1A2726CE //ln(1/frcpa(1+ 153/256)
data8 0x3FDE3AFC0A49FF40 //ln(1/frcpa(1+ 154/256)
data8 0x3FDE6185206D516E //ln(1/frcpa(1+ 155/256)
data8 0x3FDE882578823D52 //ln(1/frcpa(1+ 156/256)
data8 0x3FDEAEDD2EAC990C //ln(1/frcpa(1+ 157/256)
data8 0x3FDED5AC5F436BE3 //ln(1/frcpa(1+ 158/256)
data8 0x3FDEFC9326D16AB9 //ln(1/frcpa(1+ 159/256)
data8 0x3FDF2391A2157600 //ln(1/frcpa(1+ 160/256)
data8 0x3FDF4AA7EE03192D //ln(1/frcpa(1+ 161/256)
data8 0x3FDF71D627C30BB0 //ln(1/frcpa(1+ 162/256)
data8 0x3FDF991C6CB3B379 //ln(1/frcpa(1+ 163/256)
data8 0x3FDFC07ADA69A910 //ln(1/frcpa(1+ 164/256)
data8 0x3FDFE7F18EB03D3E //ln(1/frcpa(1+ 165/256)
data8 0x3FE007C053C5002E //ln(1/frcpa(1+ 166/256)
data8 0x3FE01B942198A5A1 //ln(1/frcpa(1+ 167/256)
data8 0x3FE02F74400C64EB //ln(1/frcpa(1+ 168/256)
data8 0x3FE04360BE7603AD //ln(1/frcpa(1+ 169/256)
data8 0x3FE05759AC47FE34 //ln(1/frcpa(1+ 170/256)
data8 0x3FE06B5F1911CF52 //ln(1/frcpa(1+ 171/256)
data8 0x3FE078BF0533C568 //ln(1/frcpa(1+ 172/256)
data8 0x3FE08CD9687E7B0E //ln(1/frcpa(1+ 173/256)
data8 0x3FE0A10074CF9019 //ln(1/frcpa(1+ 174/256)
data8 0x3FE0B5343A234477 //ln(1/frcpa(1+ 175/256)
data8 0x3FE0C974C89431CE //ln(1/frcpa(1+ 176/256)
data8 0x3FE0DDC2305B9886 //ln(1/frcpa(1+ 177/256)
data8 0x3FE0EB524BAFC918 //ln(1/frcpa(1+ 178/256)
data8 0x3FE0FFB54213A476 //ln(1/frcpa(1+ 179/256)
data8 0x3FE114253DA97D9F //ln(1/frcpa(1+ 180/256)
data8 0x3FE128A24F1D9AFF //ln(1/frcpa(1+ 181/256)
data8 0x3FE1365252BF0865 //ln(1/frcpa(1+ 182/256)
data8 0x3FE14AE558B4A92D //ln(1/frcpa(1+ 183/256)
data8 0x3FE15F85A19C765B //ln(1/frcpa(1+ 184/256)
data8 0x3FE16D4D38C119FA //ln(1/frcpa(1+ 185/256)
data8 0x3FE18203C20DD133 //ln(1/frcpa(1+ 186/256)
data8 0x3FE196C7BC4B1F3B //ln(1/frcpa(1+ 187/256)
data8 0x3FE1A4A738B7A33C //ln(1/frcpa(1+ 188/256)
data8 0x3FE1B981C0C9653D //ln(1/frcpa(1+ 189/256)
data8 0x3FE1CE69E8BB106B //ln(1/frcpa(1+ 190/256)
data8 0x3FE1DC619DE06944 //ln(1/frcpa(1+ 191/256)
data8 0x3FE1F160A2AD0DA4 //ln(1/frcpa(1+ 192/256)
data8 0x3FE2066D7740737E //ln(1/frcpa(1+ 193/256)
data8 0x3FE2147DBA47A394 //ln(1/frcpa(1+ 194/256)
data8 0x3FE229A1BC5EBAC3 //ln(1/frcpa(1+ 195/256)
data8 0x3FE237C1841A502E //ln(1/frcpa(1+ 196/256)
data8 0x3FE24CFCE6F80D9A //ln(1/frcpa(1+ 197/256)
data8 0x3FE25B2C55CD5762 //ln(1/frcpa(1+ 198/256)
data8 0x3FE2707F4D5F7C41 //ln(1/frcpa(1+ 199/256)
data8 0x3FE285E0842CA384 //ln(1/frcpa(1+ 200/256)
data8 0x3FE294294708B773 //ln(1/frcpa(1+ 201/256)
data8 0x3FE2A9A2670AFF0C //ln(1/frcpa(1+ 202/256)
data8 0x3FE2B7FB2C8D1CC1 //ln(1/frcpa(1+ 203/256)
data8 0x3FE2C65A6395F5F5 //ln(1/frcpa(1+ 204/256)
data8 0x3FE2DBF557B0DF43 //ln(1/frcpa(1+ 205/256)
data8 0x3FE2EA64C3F97655 //ln(1/frcpa(1+ 206/256)
data8 0x3FE3001823684D73 //ln(1/frcpa(1+ 207/256)
data8 0x3FE30E97E9A8B5CD //ln(1/frcpa(1+ 208/256)
data8 0x3FE32463EBDD34EA //ln(1/frcpa(1+ 209/256)
data8 0x3FE332F4314AD796 //ln(1/frcpa(1+ 210/256)
data8 0x3FE348D90E7464D0 //ln(1/frcpa(1+ 211/256)
data8 0x3FE35779F8C43D6E //ln(1/frcpa(1+ 212/256)
data8 0x3FE36621961A6A99 //ln(1/frcpa(1+ 213/256)
data8 0x3FE37C299F3C366A //ln(1/frcpa(1+ 214/256)
data8 0x3FE38AE2171976E7 //ln(1/frcpa(1+ 215/256)
data8 0x3FE399A157A603E7 //ln(1/frcpa(1+ 216/256)
data8 0x3FE3AFCCFE77B9D1 //ln(1/frcpa(1+ 217/256)
data8 0x3FE3BE9D503533B5 //ln(1/frcpa(1+ 218/256)
data8 0x3FE3CD7480B4A8A3 //ln(1/frcpa(1+ 219/256)
data8 0x3FE3E3C43918F76C //ln(1/frcpa(1+ 220/256)
data8 0x3FE3F2ACB27ED6C7 //ln(1/frcpa(1+ 221/256)
data8 0x3FE4019C2125CA93 //ln(1/frcpa(1+ 222/256)
data8 0x3FE4181061389722 //ln(1/frcpa(1+ 223/256)
data8 0x3FE42711518DF545 //ln(1/frcpa(1+ 224/256)
data8 0x3FE436194E12B6BF //ln(1/frcpa(1+ 225/256)
data8 0x3FE445285D68EA69 //ln(1/frcpa(1+ 226/256)
data8 0x3FE45BCC464C893A //ln(1/frcpa(1+ 227/256)
data8 0x3FE46AED21F117FC //ln(1/frcpa(1+ 228/256)
data8 0x3FE47A1527E8A2D3 //ln(1/frcpa(1+ 229/256)
data8 0x3FE489445EFFFCCC //ln(1/frcpa(1+ 230/256)
data8 0x3FE4A018BCB69835 //ln(1/frcpa(1+ 231/256)
data8 0x3FE4AF5A0C9D65D7 //ln(1/frcpa(1+ 232/256)
data8 0x3FE4BEA2A5BDBE87 //ln(1/frcpa(1+ 233/256)
data8 0x3FE4CDF28F10AC46 //ln(1/frcpa(1+ 234/256)
data8 0x3FE4DD49CF994058 //ln(1/frcpa(1+ 235/256)
data8 0x3FE4ECA86E64A684 //ln(1/frcpa(1+ 236/256)
data8 0x3FE503C43CD8EB68 //ln(1/frcpa(1+ 237/256)
data8 0x3FE513356667FC57 //ln(1/frcpa(1+ 238/256)
data8 0x3FE522AE0738A3D8 //ln(1/frcpa(1+ 239/256)
data8 0x3FE5322E26867857 //ln(1/frcpa(1+ 240/256)
data8 0x3FE541B5CB979809 //ln(1/frcpa(1+ 241/256)
data8 0x3FE55144FDBCBD62 //ln(1/frcpa(1+ 242/256)
data8 0x3FE560DBC45153C7 //ln(1/frcpa(1+ 243/256)
data8 0x3FE5707A26BB8C66 //ln(1/frcpa(1+ 244/256)
data8 0x3FE587F60ED5B900 //ln(1/frcpa(1+ 245/256)
data8 0x3FE597A7977C8F31 //ln(1/frcpa(1+ 246/256)
data8 0x3FE5A760D634BB8B //ln(1/frcpa(1+ 247/256)
data8 0x3FE5B721D295F10F //ln(1/frcpa(1+ 248/256)
data8 0x3FE5C6EA94431EF9 //ln(1/frcpa(1+ 249/256)
data8 0x3FE5D6BB22EA86F6 //ln(1/frcpa(1+ 250/256)
data8 0x3FE5E6938645D390 //ln(1/frcpa(1+ 251/256)
data8 0x3FE5F673C61A2ED2 //ln(1/frcpa(1+ 252/256)
data8 0x3FE6065BEA385926 //ln(1/frcpa(1+ 253/256)
data8 0x3FE6164BFA7CC06B //ln(1/frcpa(1+ 254/256)
data8 0x3FE62643FECF9743 //ln(1/frcpa(1+ 255/256)
//
// [2;4)
data8 0xBEB2CC7A38B9355F,0x3F035F2D1833BF4C // A10,A9
data8 0xBFF51BAA7FD27785,0x3FFC9D5D5B6CDEFF // A2,A1
data8 0xBF421676F9CB46C7,0x3F7437F2FA1436C6 // A8,A7
data8 0xBFD7A7041DE592FE,0x3FE9F107FEE8BD29 // A4,A3
// [4;8)
data8 0x3F6BBBD68451C0CD,0xBF966EC3272A16F7 // A10,A9
data8 0x40022A24A39AD769,0x4014190EDF49C8C5 // A2,A1
data8 0x3FB130FD016EE241,0xBFC151B46E635248 // A8,A7
data8 0x3FDE8F611965B5FE,0xBFEB5110EB265E3D // A4,A3
// [8;16)
data8 0x3F736EF93508626A,0xBF9FE5DBADF58AF1 // A10,A9
data8 0x40110A9FC5192058,0x40302008A6F96B29 // A2,A1
data8 0x3FB8E74E0CE1E4B5,0xBFC9B5DA78873656 // A8,A7
data8 0x3FE99D0DF10022DC,0xBFF829C0388F9484 // A4,A3
// [16;32)
data8 0x3F7FFF9D6D7E9269,0xBFAA780A249AEDB1 // A10,A9
data8 0x402082A807AEA080,0x4045ED9868408013 // A2,A1
data8 0x3FC4E1E54C2F99B7,0xBFD5DE2D6FFF1490 // A8,A7
data8 0x3FF75FC89584AE87,0xC006B4BADD886CAE // A4,A3
// [32;64)
data8 0x3F8CE54375841A5F,0xBFB801ABCFFA1BE2 // A10,A9
data8 0x403040A8B1815BDA,0x405B99A917D24B7A // A2,A1
data8 0x3FD30CAB81BFFA03,0xBFE41AEF61ECF48B // A8,A7
data8 0x400650CC136BEC43,0xC016022046E8292B // A4,A3
// [64;128)
data8 0x3F9B69BD22CAA8B8,0xBFC6D48875B7A213 // A10,A9
data8 0x40402028CCAA2F6D,0x40709AACEB3CBE0F // A2,A1
data8 0x3FE22C6A5924761E,0xBFF342F5F224523D // A8,A7
data8 0x4015CD405CCA331F,0xC025AAD10482C769 // A4,A3
// [128;256)
data8 0x3FAAAD9CD0E40D06,0xBFD63FC8505D80CB // A10,A9
data8 0x40501008D56C2648,0x408364794B0F4376 // A2,A1
data8 0x3FF1BE0126E00284,0xC002D8E3F6F7F7CA // A8,A7
data8 0x40258C757E95D860,0xC0357FA8FD398011 // A4,A3
// [256;512)
data8 0x3FBA4DAC59D49FEB,0xBFE5F476D1C43A77 // A10,A9
data8 0x40600800D890C7C6,0x40962C42AAEC8EF0 // A2,A1
data8 0x40018680ECF19B89,0xC012A3EB96FB7BA4 // A8,A7
data8 0x40356C4CDD3B60F9,0xC0456A34BF18F440 // A4,A3
// [512;1024)
data8 0x3FCA1B54F6225A5A,0xBFF5CD67BA10E048 // A10,A9
data8 0x407003FED94C58C2,0x40A8F30B4ACBCD22 // A2,A1
data8 0x40116A135EB66D8C,0xC022891B1CED527E // A8,A7
data8 0x40455C4617FDD8BC,0xC0555F82729E59C4 // A4,A3
// [1024;2048)
data8 0x3FD9FFF9095C6EC9,0xC005B88CB25D76C9 // A10,A9
data8 0x408001FE58FA734D,0x40BBB953BAABB0F3 // A2,A1
data8 0x40215B2F9FEB5D87,0xC0327B539DEA5058 // A8,A7
data8 0x40555444B3E8D64D,0xC0655A2B26F9FC8A // A4,A3
// [2048;4096)
data8 0x3FE9F065A1C3D6B1,0xC015ACF6FAE8D78D // A10,A9
data8 0x409000FE383DD2B7,0x40CE7F5C1E8BCB8B // A2,A1
data8 0x40315324E5DB2EBE,0xC04274194EF70D18 // A8,A7
data8 0x4065504353FF2207,0xC075577FE1BFE7B6 // A4,A3
// [4096;8192)
data8 0x3FF9E6FBC6B1C70D,0xC025A62DAF76F85D // A10,A9
data8 0x40A0007E2F61EBE8,0x40E0A2A23FB5F6C3 // A2,A1
data8 0x40414E9BC0A0141A,0xC0527030F2B69D43 // A8,A7
data8 0x40754E417717B45B,0xC085562A447258E5 // A4,A3
//
data8 0xbfdffffffffaea15 // P1
data8 0x3FDD8B618D5AF8FE // point of local minimum on [1;2]
data8 0x3FED67F1C864BEB5 // ln(sqrt(2*Pi))
data8 0x4008000000000000 // 3.0
//
data8 0xBF9E1C289FB224AB,0x3FBF7422445C9460 // A6,A5
data8 0xBFF01E76D66F8D8A // A0
data8 0xBFE2788CFC6F91DA // A1 [1.0;1.25)
data8 0x3FCB8CC69000EB5C,0xBFD41997A0C2C641 // A6,A5
data8 0x3FFCAB0BFA0EA462 // A0
data8 0xBFBF19B9BCC38A42 // A0 [1.25;1.5)
data8 0x3FD51EE4DE0A364C,0xBFE00D7F98A16E4B // A6,A5
data8 0x40210CE1F327E9E4 // A0
data8 0x4001DB08F9DFA0CC // A0 [1.5;1.75)
data8 0x3FE24F606742D252,0xBFEC81D7D12574EC // A6,A5
data8 0x403BE636A63A9C27 // A0
data8 0x4000A0CB38D6CF0A // A0 [1.75;2.0)
data8 0x3FF1029A9DD542B4,0xBFFAD37C209D3B25 // A6,A5
data8 0x405385E6FD9BE7EA // A0
data8 0x478895F1C0000000 // Overflow boundary
data8 0x400062D97D26B523,0xC00A03E1529FF023 // A6,A5
data8 0x4069204C51E566CE // A0
data8 0x0000000000000000 // pad
data8 0x40101476B38FD501,0xC0199DE7B387C0FC // A6,A5
data8 0x407EB8DAEC83D759 // A0
data8 0x0000000000000000 // pad
data8 0x401FDB008D65125A,0xC0296B506E665581 // A6,A5
data8 0x409226D93107EF66 // A0
data8 0x0000000000000000 // pad
data8 0x402FB3EAAF3E7B2D,0xC039521142AD8E0D // A6,A5
data8 0x40A4EFA4F072792E // A0
data8 0x0000000000000000 // pad
data8 0x403FA024C66B2563,0xC0494569F250E691 // A6,A5
data8 0x40B7B747C9235BB8 // A0
data8 0x0000000000000000 // pad
data8 0x404F9607D6DA512C,0xC0593F0B2EDDB4BC // A6,A5
data8 0x40CA7E29C5F16DE2 // A0
data8 0x0000000000000000 // pad
data8 0x405F90C5F613D98D,0xC0693BD130E50AAF // A6,A5
data8 0x40DD4495238B190C // A0
data8 0x0000000000000000 // pad
//
// polynomial approximation of ln(sin(Pi*x)/(Pi*x)), |x| <= 0.5
data8 0xBFD58731A486E820,0xBFA4452CC28E15A9 // S16,S14
data8 0xBFD013F6E1B86C4F,0xBFD5B3F19F7A341F // S8,S6
data8 0xBFC86A0D5252E778,0xBFC93E08C9EE284B // S12,S10
data8 0xBFE15132555C9EDD,0xBFFA51A662480E35 // S4,S2
//
// [1.0;1.25)
data8 0xBFA697D6775F48EA,0x3FB9894B682A98E7 // A9,A8
data8 0xBFCA8969253CFF55,0x3FD15124EFB35D9D // A5,A4
data8 0xBFC1B00158AB719D,0x3FC5997D04E7F1C1 // A7,A6
data8 0xBFD9A4D50BAFF989,0x3FEA51A661F5176A // A3,A2
// [1.25;1.5)
data8 0x3F838E0D35A6171A,0xBF831BBBD61313B7 // A8,A7
data8 0x3FB08B40196425D0,0xBFC2E427A53EB830 // A4,A3
data8 0x3F9285DDDC20D6C3,0xBFA0C90C9C223044 // A6,A5
data8 0x3FDEF72BC8F5287C,0x3D890B3DAEBC1DFC // A2,A1
// [1.5;1.75)
data8 0x3F65D5A7EB31047F,0xBFA44EAC9BFA7FDE // A8,A7
data8 0x40051FEFE7A663D8,0xC012A5CFE00A2522 // A4,A3
data8 0x3FD0E1583AB00E08,0xBFF084AF95883BA5 // A6,A5
data8 0x40185982877AE0A2,0xC015F83DB73B57B7 // A2,A1
// [1.75;2.0)
data8 0x3F4A9222032EB39A,0xBF8CBC9587EEA5A3 // A8,A7
data8 0x3FF795400783BE49,0xC00851BC418B8A25 // A4,A3
data8 0x3FBBC992783E8C5B,0xBFDFA67E65E89B29 // A6,A5
data8 0x4012B408F02FAF88,0xC013284CE7CB0C39 // A2,A1
//
// roots
data8 0xC003A7FC9600F86C // -2.4570247382208005860
data8 0xC009260DBC9E59AF // -3.1435808883499798405
data8 0xC005FB410A1BD901 // -2.7476826467274126919
data8 0xC00FA471547C2FE5 // -3.9552942848585979085
//
// polynomial approximation of ln(GAMMA(x)) near roots
// near -2.4570247382208005860
data8 0x3FF694A6058D9592,0x40136EEBB003A92B // R3,R2
data8 0x3FF83FE966AF5360,0x3C90323B6D1FE86D // R1,R0
// near -3.1435808883499798405
data8 0x405C11371268DA38,0x4039D4D2977D2C23 // R3,R2
data8 0x401F20A65F2FAC62,0x3CDE9605E3AE7A62 // R1,R0
// near -2.7476826467274126919
data8 0xC034185AC31314FF,0x4023267F3C28DFE3 // R3,R2
data8 0xBFFEA12DA904B194,0x3CA8FB8530BA7689 // R1,R0
// near -2.7476826467274126919
data8 0xC0AD25359E70C888,0x406F76DEAEA1B8C6 // R3,R2
data8 0xC034B99D966C5644,0xBCBDDC0336980B58 // R1,R0
LOCAL_OBJECT_END(lgammaf_data)

//*********************************************************************

.section .text
GLOBAL_LIBM_ENTRY(__libm_lgammaf)
{ .mfi
      getf.exp      GR_SignExp = f8
      frcpa.s1      FR_InvX,p0 = f1,f8
      mov           GR_ExpOf2 = 0x10000
}
{ .mfi
      addl          GR_ad_Data = @ltoff(lgammaf_data),gp
      fcvt.fx.s1    FR_int_N = f8
      mov           GR_ExpMask = 0x1ffff
};;
{ .mfi
      getf.sig      GR_Sig = f8
      fclass.m      p13,p0 = f8,0x1EF // is x NaTVal, NaN,
                                      // +/-0, +/-INF or +/-deno?
      mov           GR_ExpBias = 0xffff
}
{ .mfi
      ld8           GR_ad_Data = [GR_ad_Data]
      fma.s1        FR_Xp1 = f8,f1,f1
      mov           GR_StirlBound = 0x1000C
};;
{ .mfi
      setf.exp      FR_2 = GR_ExpOf2
      fmerge.se     FR_x = f1,f8
      dep.z         GR_Ind = GR_SignExp,3,4
}
{ .mfi
      cmp.eq        p8,p0 = GR_SignExp,GR_ExpBias
      fcvt.fx.trunc.s1 FR_int_Ntrunc = f8
      and           GR_Exp = GR_ExpMask,GR_SignExp
};;
{ .mfi
      add           GR_ad_C650 = 0xB20,GR_ad_Data
      fcmp.lt.s1    p14,p15 = f8,f0
      extr.u        GR_Ind4T = GR_Sig,55,8
}
{ .mfb
      sub           GR_PureExp = GR_Exp,GR_ExpBias
      fnorm.s1      FR_NormX = f8
      // jump if x is NaTVal, NaN, +/-0, +/-INF or +/-deno
(p13) br.cond.spnt  lgammaf_spec
};;
lgammaf_core:
{ .mfi
      ldfpd         FR_P1,FR_LocalMin = [GR_ad_C650],16
      fms.s1        FR_xm2 = f8,f1,f1
      add           GR_ad_Co = 0x820,GR_ad_Data
}
{ .mib
      ldfpd         FR_P3,FR_P2 = [GR_ad_Data],16
      cmp.ltu       p9,p0 = GR_SignExp,GR_ExpBias
      // jump if x is from the interval [1; 2)
(p8)  br.cond.spnt  lgammaf_1_2
};;
{ .mfi
      setf.sig      FR_int_Ln = GR_PureExp
      fms.s1        FR_r = FR_InvX,f8,f1
      shladd        GR_ad_Co = GR_Ind,3,GR_ad_Co
}
{ .mib
      ldfpd         FR_LnSqrt2Pi,FR_3 = [GR_ad_C650],16
      cmp.lt        p13,p12 = GR_Exp,GR_StirlBound
      // jump if x is from the interval (0; 1)
(p9)  br.cond.spnt  lgammaf_0_1
};;
{ .mfi
      ldfpd         FR_Ln2,FR_05 = [GR_ad_Data],16
      fma.s1        FR_Xp2 = f1,f1,FR_Xp1 // (x+2)
      shladd        GR_ad_C650 = GR_Ind,2,GR_ad_C650
}
{ .mfi
      add           GR_ad_Ce = 0x20,GR_ad_Co
      nop.f         0
      add           GR_ad_C43 = 0x30,GR_ad_Co
};;
{ .mfi
      // load coefficients of polynomial approximation
      // of ln(GAMMA(x)), 2 <= x < 2^13
(p13) ldfpd         FR_A10,FR_A9 = [GR_ad_Co],16
      fcvt.xf       FR_N = FR_int_N
      cmp.eq.unc    p6,p7 = GR_ExpOf2,GR_SignExp
}
{ .mib
(p13) ldfpd         FR_A8,FR_A7 = [GR_ad_Ce]
(p14) cmp.le.unc    p9,p0 = GR_StirlBound,GR_Exp
      // jump if x is less or equal to -2^13
(p9)  br.cond.spnt  lgammaf_negstirling
};;
.pred.rel "mutex",p6,p7
{ .mfi
(p13) ldfpd         FR_A6,FR_A5 = [GR_ad_C650],16
(p6)  fma.s1        FR_x = f0,f0,FR_NormX
      shladd        GR_ad_T = GR_Ind4T,3,GR_ad_Data
}
{ .mfi
(p13) ldfpd         FR_A4,FR_A3 = [GR_ad_C43]
(p7)  fms.s1        FR_x = FR_x,f1,f1
(p14) mov           GR_ReqBound = 0x20005
};;
{ .mfi
(p13) ldfpd         FR_A2,FR_A1 = [GR_ad_Co],16
      fms.s1        FR_xm2 = FR_xm2,f1,f1
(p14) extr.u        GR_Arg = GR_Sig,60,4
}
{ .mfi
      mov           GR_SignOfGamma = 1 // set sign of gamma(x) to 1
      fcvt.xf       FR_Ntrunc = FR_int_Ntrunc
      nop.i         0
};;
{ .mfi
      ldfd          FR_T = [GR_ad_T]
      fma.s1        FR_r2 = FR_r,FR_r,f0
      shl           GR_ReqBound = GR_ReqBound,3
}
{ .mfi
      add           GR_ad_Co = 0xCA0,GR_ad_Data
      fnma.s1       FR_Req = FR_Xp1,FR_NormX,f0 // -x*(x+1)
(p14) shladd        GR_Arg = GR_Exp,4,GR_Arg
};;
{ .mfi
(p13) ldfd          FR_A0 = [GR_ad_C650]
      fma.s1        FR_Xp3 = FR_2,f1,FR_Xp1 // (x+3)
(p14) cmp.le.unc    p9,p0 = GR_Arg,GR_ReqBound
}
{ .mfi
(p14) add           GR_ad_Ce = 0x20,GR_ad_Co
      fma.s1        FR_Xp4 = FR_2,FR_2,FR_NormX // (x+4)
(p15) add           GR_ad_OvfBound = 0xBB8,GR_ad_Data
};;
{ .mfi
      // load coefficients of polynomial approximation
      // of ln(sin(Pi*xf)/(Pi*xf)), |xf| <= 0.5
(p14) ldfpd         FR_S16,FR_S14 = [GR_ad_Co],16
(p14) fms.s1        FR_Xf = FR_NormX,f1,FR_N  // xf = x - [x]
(p14) sub           GR_SignOfGamma = r0,GR_SignOfGamma // set sign of
                                                       // gamma(x) to -1
}
{ .mfb
(p14) ldfpd         FR_S12,FR_S10 = [GR_ad_Ce],16
      fma.s1        FR_Xp5 = FR_2,FR_2,FR_Xp1 // (x+5)
      // jump if x is from the interval (-9; 0)
(p9)  br.cond.spnt  lgammaf_negrecursion
};;
{ .mfi
(p14) ldfpd         FR_S8,FR_S6 = [GR_ad_Co],16
      fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
}
{ .mfb
(p14) ldfpd         FR_S4,FR_S2 = [GR_ad_Ce],16
      fma.s1        FR_x2 = FR_x,FR_x,f0
      // jump if x is from the interval (-2^13; -9)
(p14) br.cond.spnt  lgammaf_negpoly
};;
{ .mfi
      ldfd          FR_OverflowBound = [GR_ad_OvfBound]
(p12) fcvt.xf       FR_N = FR_int_Ln
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
}
{ .mfi
      nop.m         0
(p12) fma.s1        FR_P10 = FR_P1,FR_r,f1
      nop.i         0
};;
.pred.rel "mutex",p6,p7
.pred.rel "mutex",p9,p10
{ .mfi
      // store sign of gamma(x) as 32-bit int
(p9)  st4           [r33] = GR_SignOfGamma
(p6)  fma.s1        FR_xx = FR_x,FR_xm2,f0
      nop.i         0
}
{ .mfi
      // store sign of gamma(x) as 64-bit int
(p10) st8           [r33] = GR_SignOfGamma
(p7)  fma.s1        FR_xx = f0,f0,FR_x
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A9 = FR_A10,FR_x,FR_A9
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A7 = FR_A8,FR_x,FR_A7
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A5 = FR_A6,FR_x,FR_A5
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A3 = FR_A4,FR_x,FR_A3
      nop.i         0
};;
{ .mfi
      nop.m         0
(p15) fcmp.eq.unc.s1 p8,p0 = FR_NormX,FR_2 // is input argument 2.0?
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A1 = FR_A2,FR_x,FR_A1
      nop.i         0
};;
{ .mfi
      nop.m         0
(p12) fma.s1        FR_T = FR_N,FR_Ln2,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
(p12) fma.s1        FR_P32 = FR_P32,FR_r2,FR_P10
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_x3 = FR_x2,FR_xx,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A7 = FR_A9,FR_x2,FR_A7
      nop.i         0
}
{ .mfb
      nop.m         0
(p8)  fma.s.s0      f8 = f0,f0,f0
(p8)  br.ret.spnt   b0 // fast exit for 2.0
};;
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_A0 = FR_A0,FR_xm2,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A3 = FR_A5,FR_x2,FR_A3
      nop.i         0
};;
{ .mfi
      nop.m         0
(p15) fcmp.le.unc.s1 p8,p0 = FR_OverflowBound,FR_NormX // overflow test
      nop.i         0
}
{ .mfi
      nop.m         0
(p12) fms.s1        FR_xm05 = FR_NormX,f1,FR_05
      nop.i         0
};;
{ .mfi
      nop.m         0
(p12) fma.s1        FR_Ln = FR_P32,FR_r,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
(p12) fms.s1        FR_LnSqrt2Pi = FR_LnSqrt2Pi,f1,FR_NormX
      nop.i         0
};;
{ .mfi
      nop.m         0
(p13) fma.s1        FR_A0 = FR_A1,FR_xx,FR_A0
      nop.i         0
}
{ .mfb
      nop.m         0
(p13) fma.s1        FR_A3 = FR_A7,FR_x4,FR_A3
      // jump if result overflows
(p8)  br.cond.spnt  lgammaf_overflow
};;
.pred.rel "mutex",p12,p13
{ .mfi
      nop.m         0
(p12) fma.s.s0      f8 = FR_Ln,FR_xm05,FR_LnSqrt2Pi
      nop.i         0
}
{ .mfb
      nop.m         0
(p13) fma.s.s0      f8 = FR_A3,FR_x3,FR_A0
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for 0 < x < 1
//---------------------------------------------------------------------
.align 32
lgammaf_0_1:
{ .mfi
      getf.sig      GR_Ind = FR_Xp1
      fma.s1        FR_r2 = FR_r,FR_r,f0
      mov           GR_fff7 = 0xFFF7
}
{ .mfi
      ldfpd         FR_Ln2,FR_05 = [GR_ad_Data],16
      fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      // input argument can't be equal to 1.0
      cmp.eq        p0,p14 = r0,r0
};;
{ .mfi
      getf.exp      GR_Exp = FR_w
      fcvt.xf       FR_N = FR_int_Ln
      add           GR_ad_Co = 0xCE0,GR_ad_Data
}
{ .mfi
      shladd        GR_ad_T = GR_Ind4T,3,GR_ad_Data
      fma.s1        FR_P10 = FR_P1,FR_r,f1
      add           GR_ad_Ce = 0xD00,GR_ad_Data
};;
{ .mfi
      ldfd          FR_T = [GR_ad_T]
      fma.s1        FR_w2 = FR_w,FR_w,f0
      extr.u        GR_Ind = GR_Ind,61,2
}
{ .mfi
      nop.m         0
      fma.s1        FR_Q32 = FR_P3,FR_w,FR_P2
////      add           GR_ad_C0 = 0xB30,GR_ad_Data
      add           GR_ad_C0 = 0xB38,GR_ad_Data
};;
{ .mfi
      and           GR_Exp = GR_Exp,GR_ExpMask
      nop.f         0
      shladd        GR_IndX8 = GR_Ind,3,r0
}
{ .mfi
      shladd        GR_IndX2 = GR_Ind,1,r0
      fma.s1        FR_Q10 = FR_P1,FR_w,f1
      cmp.eq        p6,p15 = 0,GR_Ind
};;
{ .mfi
      shladd        GR_ad_Co = GR_IndX8,3,GR_ad_Co
(p6)  fma.s1        FR_x = f0,f0,FR_NormX
      shladd        GR_ad_C0 = GR_IndX2,4,GR_ad_C0
}
{ .mfi
      shladd        GR_ad_Ce = GR_IndX8,3,GR_ad_Ce
      nop.f         0
(p15) cmp.eq.unc    p7,p8 = 1,GR_Ind
};;
.pred.rel "mutex",p7,p8
{ .mfi
      ldfpd         FR_A8,FR_A7 = [GR_ad_Co],16
(p7)  fms.s1        FR_x = FR_NormX,f1,FR_LocalMin
      cmp.ge        p10,p11 = GR_Exp,GR_fff7
}
{ .mfb
      ldfpd         FR_A6,FR_A5 = [GR_ad_Ce],16
(p8)  fma.s1        FR_x = f1,f1,FR_NormX
      br.cond.sptk  lgamma_0_2_core
};;
// branch for calculating of ln(GAMMA(x)) for 1 <= x < 2
//---------------------------------------------------------------------
.align 32
lgammaf_1_2:
{ .mfi
      add           GR_ad_Co = 0xCF0,GR_ad_Data
      fcmp.eq.s1    p14,p0 = f1,FR_NormX // is input argument 1.0?
      extr.u        GR_Ind = GR_Sig,61,2
}
{ .mfi
      add           GR_ad_Ce = 0xD10,GR_ad_Data
      nop.f         0
////      add           GR_ad_C0 = 0xB40,GR_ad_Data
      add           GR_ad_C0 = 0xB48,GR_ad_Data
};;
{ .mfi
      shladd        GR_IndX8 = GR_Ind,3,r0
      nop.f         0
      shladd        GR_IndX2 = GR_Ind,1,r0
}
{ .mfi
      cmp.eq        p6,p15 = 0,GR_Ind // p6 <- x from [1;1.25)
      nop.f         0
      cmp.ne        p9,p0 = r0,r0
};;
{ .mfi
      shladd        GR_ad_Co = GR_IndX8,3,GR_ad_Co
(p6)  fms.s1        FR_x = FR_NormX,f1,f1 // reduced x for [1;1.25)
      shladd        GR_ad_C0 = GR_IndX2,4,GR_ad_C0
}
{ .mfi
      shladd        GR_ad_Ce = GR_IndX8,3,GR_ad_Ce
(p14) fma.s.s0      f8 = f0,f0,f0
(p15) cmp.eq.unc    p7,p8 = 1,GR_Ind // p7 <- x from [1.25;1.5)
};;
.pred.rel "mutex",p7,p8
{ .mfi
      ldfpd         FR_A8,FR_A7 = [GR_ad_Co],16
(p7)  fms.s1        FR_x = FR_xm2,f1,FR_LocalMin
      nop.i         0
}
{ .mfi
      ldfpd         FR_A6,FR_A5 = [GR_ad_Ce],16
(p8)  fma.s1        FR_x = f0,f0,FR_NormX
(p9)  cmp.eq.unc    p10,p11 = r0,r0
};;
lgamma_0_2_core:
{ .mmi
      ldfpd         FR_A4,FR_A3 = [GR_ad_Co],16
      ldfpd         FR_A2,FR_A1 = [GR_ad_Ce],16
      mov           GR_SignOfGamma = 1 // set sign of gamma(x) to 1
};;
{ .mfi
//      add           GR_ad_C0 = 8,GR_ad_C0
      ldfd          FR_A0 = [GR_ad_C0]
      nop.f         0
      // set p13 if signgum is 32-bit int
      // set p15 if signgum is 64-bit int
      cmp.eq        p15,p13 = 8,r34
};;
.pred.rel "mutex",p13,p15
{ .mmf
      // store sign of gamma(x)
(p13) st4           [r33] = GR_SignOfGamma // as 32-bit int
(p15) st8           [r33] = GR_SignOfGamma // as 64-bit int
(p11) fma.s1        FR_Q32 = FR_Q32,FR_w2,FR_Q10
};;
{ .mfb
      nop.m         0
(p10) fma.s1        FR_P32 = FR_P32,FR_r2,FR_P10
(p14) br.ret.spnt   b0 // fast exit for 1.0
};;
{ .mfi
      nop.m         0
(p10) fma.s1        FR_T = FR_N,FR_Ln2,FR_T
      cmp.eq        p6,p7 = 0,GR_Ind // p6 <- x from [1;1.25)
}
{ .mfi
      nop.m         0
      fma.s1        FR_x2 = FR_x,FR_x,f0
      cmp.eq        p8,p0 = r0,r0 // set p8 to 1 that means we on [1;2]
};;
{ .mfi
      nop.m         0
(p11) fma.s1        FR_Ln = FR_Q32,FR_w,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      nop.f         0
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_xx = f0,f0,FR_x
      nop.i         0
}
{ .mfi
      nop.m         0
(p7)  fma.s1        FR_xx = f0,f0,f1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A8,FR_x,FR_A7
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A5 = FR_A6,FR_x,FR_A5
(p9)  cmp.ne        p8,p0 = r0,r0 // set p8 to 0 that means we on [0;1]
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A4,FR_x,FR_A3
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A2,FR_x,FR_A1
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_Ln = FR_P32,FR_r,FR_T
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A5 = FR_A7,FR_x2,FR_A5
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A3,FR_x2,FR_A1
      nop.i         0
};;
.pred.rel "mutex",p9,p8
{ .mfi
      nop.m         0
(p9)  fms.d.s1      FR_A0 = FR_A0,FR_xx,FR_Ln
      nop.i         0
}
{ .mfi
      nop.m         0
(p8)  fms.s1        FR_A0 = FR_A0,FR_xx,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.d.s1      FR_A1 = FR_A5,FR_x4,FR_A1
      nop.i         0
}
{ .mfi
      nop.m         0
      nop.f         0
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fma.s.s0      f8 = FR_A1,FR_x2,FR_A0
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.s.s0      f8 = FR_A1,FR_x,FR_A0
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for -9 < x < 1
//---------------------------------------------------------------------
.align 32
lgammaf_negrecursion:
{ .mfi
      getf.sig      GR_N = FR_int_Ntrunc
      fms.s1        FR_1pXf = FR_Xp2,f1,FR_Ntrunc // 1 + (x+1) - [x]
      mov           GR_Neg2 = 2
}
{ .mfi
      add           GR_ad_Co = 0xCE0,GR_ad_Data
      fms.s1        FR_Xf = FR_Xp1,f1,FR_Ntrunc // (x+1) - [x]
      mov           GR_Neg4 = 4
};;
{ .mfi
      add           GR_ad_Ce = 0xD00,GR_ad_Data
      fma.s1        FR_Xp6 = FR_2,FR_2,FR_Xp2 // (x+6)
      add           GR_ad_C0 = 0xB30,GR_ad_Data
}
{ .mfi
      sub           GR_Neg2 = r0,GR_Neg2
      fma.s1        FR_Xp7 = FR_2,FR_3,FR_Xp1 // (x+7)
      sub           GR_Neg4 = r0,GR_Neg4
};;
{ .mfi
      cmp.ne        p8,p0 = r0,GR_N
      fcmp.eq.s1    p13,p0 = FR_NormX,FR_Ntrunc
      and           GR_IntNum = 0xF,GR_N
}
{ .mfi
      cmp.lt        p6,p0 = GR_N,GR_Neg2
      fma.s1        FR_Xp8 = FR_2,FR_3,FR_Xp2 // (x+8)
      cmp.lt        p7,p0 = GR_N,GR_Neg4
};;
{ .mfi
      getf.d        GR_Arg = FR_NormX
(p6)  fma.s1        FR_Xp2 = FR_Xp2,FR_Xp3,f0
(p8)  tbit.z.unc    p14,p15 = GR_IntNum,0
}
{ .mfi
      sub           GR_RootInd = 0xE,GR_IntNum
(p7)  fma.s1        FR_Xp4 = FR_Xp4,FR_Xp5,f0
      add           GR_ad_Root = 0xDE0,GR_ad_Data
};;
{ .mfi
      shladd        GR_ad_Root = GR_RootInd,3,GR_ad_Root
      fms.s1        FR_x = FR_Xp1,f1,FR_Ntrunc // (x+1) - [x]
      nop.i         0
}
{ .mfb
      nop.m         0
      nop.f         0
(p13) br.cond.spnt  lgammaf_singularity
};;
.pred.rel "mutex",p14,p15
{ .mfi
      cmp.gt        p6,p0 = 0xA,GR_IntNum
(p14) fma.s1        FR_Req = FR_Req,FR_Xf,f0
      cmp.gt        p7,p0 = 0xD,GR_IntNum
}
{ .mfi
(p15) mov           GR_SignOfGamma = 1 // set sign of gamma(x) to 1
(p15) fnma.s1       FR_Req = FR_Req,FR_Xf,f0
      cmp.leu       p0,p13 = 2,GR_RootInd
};;
{ .mfi
      nop.m         0
(p6)  fma.s1        FR_Xp6 = FR_Xp6,FR_Xp7,f0
(p13) add           GR_ad_RootCo = 0xE00,GR_ad_Data
};;
{ .mfi
      nop.m         0
      fcmp.eq.s1    p12,p11 = FR_1pXf,FR_2
      nop.i         0
};;
{ .mfi
      getf.sig      GR_Sig = FR_1pXf
      fcmp.le.s1    p9,p0 = FR_05,FR_Xf
      nop.i         0
}
{ .mfi
(p13) shladd        GR_RootInd = GR_RootInd,4,r0
(p7)  fma.s1        FR_Xp2 = FR_Xp2,FR_Xp4,f0
(p8)  cmp.gt.unc    p10,p0 = 0x9,GR_IntNum
};;
.pred.rel "mutex",p11,p12
{ .mfi
      nop.m         0
(p10) fma.s1        FR_Req = FR_Req,FR_Xp8,f0
(p11) extr.u        GR_Ind = GR_Sig,61,2
}
{ .mfi
(p13) add           GR_RootInd = GR_RootInd,GR_RootInd
      nop.f         0
(p12) mov           GR_Ind = 3
};;
{ .mfi
      shladd        GR_IndX2 = GR_Ind,1,r0
      nop.f         0
      cmp.gt        p14,p0 = 2,GR_Ind
}
{ .mfi
      shladd        GR_IndX8 = GR_Ind,3,r0
      nop.f         0
      cmp.eq        p6,p0 = 1,GR_Ind
};;
.pred.rel "mutex",p6,p9
{ .mfi
      shladd        GR_ad_Co = GR_IndX8,3,GR_ad_Co
(p6)  fms.s1        FR_x = FR_Xf,f1,FR_LocalMin
      cmp.gt        p10,p0 = 0xB,GR_IntNum
}
{ .mfi
      shladd        GR_ad_Ce = GR_IndX8,3,GR_ad_Ce
(p9)  fma.s1        FR_x = f0,f0,FR_1pXf
      shladd        GR_ad_C0 = GR_IndX2,4,GR_ad_C0
};;
{ .mfi
      // load coefficients of polynomial approximation
      // of ln(GAMMA(x)), 1 <= x < 2
      ldfpd         FR_A8,FR_A7 = [GR_ad_Co],16
(p10) fma.s1        FR_Xp2 = FR_Xp2,FR_Xp6,f0
      add           GR_ad_C0 = 8,GR_ad_C0
}
{ .mfi
      ldfpd         FR_A6,FR_A5 = [GR_ad_Ce],16
      nop.f         0
(p14) add           GR_ad_Root = 0x10,GR_ad_Root
};;
{ .mfi
      ldfpd         FR_A4,FR_A3 = [GR_ad_Co],16
      nop.f         0
      add           GR_ad_RootCe = 0xE10,GR_ad_Data
}
{ .mfi
      ldfpd         FR_A2,FR_A1 = [GR_ad_Ce],16
      nop.f         0
(p14) add           GR_RootInd = 0x40,GR_RootInd
};;
{ .mmi
      ldfd          FR_A0 = [GR_ad_C0]
(p13) add           GR_ad_RootCo = GR_ad_RootCo,GR_RootInd
(p13) add           GR_ad_RootCe = GR_ad_RootCe,GR_RootInd
};;
{ .mmi
(p13) ld8           GR_Root = [GR_ad_Root]
(p13) ldfd          FR_Root = [GR_ad_Root]
      mov           GR_ExpBias = 0xffff
};;
{ .mfi
      nop.m         0
      fma.s1        FR_x2 = FR_x,FR_x,f0
      nop.i         0
}
{ .mlx
(p8)  cmp.gt.unc    p10,p0 = 0xF,GR_IntNum
      movl          GR_Dx = 0x000000014F8B588E
};;
{ .mfi
      // load coefficients of polynomial approximation
      // of ln(GAMMA(x)), x is close to one of negative roots
(p13) ldfpd         FR_R3,FR_R2 = [GR_ad_RootCo]
      // arguments for logarithm
(p10) fma.s1        FR_Req = FR_Req,FR_Xp2,f0
      mov           GR_ExpMask = 0x1ffff
}
{ .mfi
(p13) ldfpd         FR_R1,FR_R0 = [GR_ad_RootCe]
      nop.f         0
      // set p9 if signgum is 32-bit int
      // set p8 if signgum is 64-bit int
      cmp.eq        p8,p9 = 8,r34
};;
.pred.rel "mutex",p9,p8
{ .mfi
(p9)  st4           [r33] = GR_SignOfGamma // as 32-bit int
      fma.s1        FR_A7 = FR_A8,FR_x,FR_A7
(p13) sub           GR_Root = GR_Arg,GR_Root
}
{ .mfi
(p8)  st8           [r33] = GR_SignOfGamma // as 64-bit int
      fma.s1        FR_A5 = FR_A6,FR_x,FR_A5
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_w = FR_Req,f1,f1
(p13) add           GR_Root = GR_Root,GR_Dx
}
{ .mfi
      nop.m         0
      nop.f         0
(p13) add           GR_2xDx = GR_Dx,GR_Dx
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A4,FR_x,FR_A3
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A2,FR_x,FR_A1
(p13) cmp.leu.unc   p10,p0 = GR_Root,GR_2xDx
};;
{ .mfi
      nop.m         0
      frcpa.s1      FR_InvX,p0 = f1,FR_Req
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fms.s1        FR_rx = FR_NormX,f1,FR_Root
      nop.i         0
};;
{ .mfi
      getf.exp      GR_SignExp = FR_Req
      fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
};;
{ .mfi
      getf.sig      GR_Sig = FR_Req
      fma.s1        FR_A5 = FR_A7,FR_x2,FR_A5
      nop.i         0
};;
{ .mfi
      sub           GR_PureExp = GR_SignExp,GR_ExpBias
      fma.s1        FR_w2 = FR_w,FR_w,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_Q32 = FR_P3,FR_w,FR_P2
      nop.i         0
};;
{ .mfi
      setf.sig      FR_int_Ln = GR_PureExp
      fma.s1        FR_A1 = FR_A3,FR_x2,FR_A1
      extr.u        GR_Ind4T = GR_Sig,55,8
}
{ .mfi
      nop.m         0
      fma.s1        FR_Q10 = FR_P1,FR_w,f1
      nop.i         0
};;
{ .mfi
      shladd        GR_ad_T = GR_Ind4T,3,GR_ad_Data
      fms.s1        FR_r = FR_InvX,FR_Req,f1
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fms.s1        FR_rx2 = FR_rx,FR_rx,f0
      nop.i         0
};;
{ .mfi
      ldfd          FR_T = [GR_ad_T]
(p10) fma.s1        FR_R2 = FR_R3,FR_rx,FR_R2
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fma.s1        FR_R0 = FR_R1,FR_rx,FR_R0
      nop.i         0
};;
{ .mfi
      getf.exp      GR_Exp = FR_w
      fma.s1        FR_A1 = FR_A5,FR_x4,FR_A1
      mov           GR_ExpMask = 0x1ffff
}
{ .mfi
      nop.m         0
      fma.s1        FR_Q32 = FR_Q32, FR_w2,FR_Q10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_r2 = FR_r,FR_r,f0
      mov           GR_fff7 = 0xFFF7
}
{ .mfi
      nop.m         0
      fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P10 = FR_P1,FR_r,f1
      and           GR_Exp = GR_ExpMask,GR_Exp
}
{ .mfb
      nop.m         0
(p10) fma.s.s0      f8 = FR_R2,FR_rx2,FR_R0
(p10) br.ret.spnt   b0 // exit for arguments close to negative roots
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_N = FR_int_Ln
      nop.i         0
}
{ .mfi
      cmp.ge        p14,p15 = GR_Exp,GR_fff7
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A0 = FR_A1,FR_x,FR_A0
      nop.i         0
}
{ .mfi
      nop.m         0
(p15) fma.s1        FR_Ln = FR_Q32,FR_w,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_P32 = FR_P32,FR_r2,FR_P10
      cmp.eq        p6,p7 = 0,GR_Ind
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_T = FR_N,FR_Ln2,FR_T
      nop.i         0
};;
{ .mfi
      nop.m         0
(p14) fma.s1        FR_Ln = FR_P32,FR_r,FR_T
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fms.s.s0      f8 = FR_A0,FR_x,FR_Ln
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fms.s.s0      f8 = FR_A0,f1,FR_Ln
      br.ret.sptk   b0
};;

// branch for calculating of ln(GAMMA(x)) for x < -2^13
//---------------------------------------------------------------------
.align 32
lgammaf_negstirling:
{ .mfi
      shladd        GR_ad_T = GR_Ind4T,3,GR_ad_Data
      fms.s1        FR_Xf = FR_NormX,f1,FR_N  // xf = x - [x]
      mov           GR_SingBound = 0x10016
}
{ .mfi
      add           GR_ad_Co = 0xCA0,GR_ad_Data
      fma.s1        FR_P32 = FR_P3,FR_r,FR_P2
      nop.i         0
};;
{ .mfi
      ldfd          FR_T = [GR_ad_T]
      fcvt.xf       FR_int_Ln = FR_int_Ln
      cmp.le        p6,p0 = GR_SingBound,GR_Exp
}
{ .mfb
      add           GR_ad_Ce = 0x20,GR_ad_Co
      fma.s1        FR_r2 = FR_r,FR_r,f0
(p6)  br.cond.spnt  lgammaf_singularity
};;
{ .mfi
      // load coefficients of polynomial approximation
      // of ln(sin(Pi*xf)/(Pi*xf)), |xf| <= 0.5
      ldfpd         FR_S16,FR_S14 = [GR_ad_Co],16
      fma.s1        FR_P10 = FR_P1,FR_r,f1
      nop.i         0
}
{ .mfi
      ldfpd         FR_S12,FR_S10 = [GR_ad_Ce],16
      fms.s1        FR_xm05 = FR_NormX,f1,FR_05
      nop.i         0
};;
{ .mmi
      ldfpd         FR_S8,FR_S6 = [GR_ad_Co],16
      ldfpd         FR_S4,FR_S2 = [GR_ad_Ce],16
      nop.i         0
};;
{ .mfi
      getf.sig      GR_N = FR_int_Ntrunc // signgam calculation
      fma.s1        FR_Xf2 = FR_Xf,FR_Xf,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      frcpa.s1      FR_InvXf,p0 = f1,FR_Xf
      nop.i         0
};;
{ .mfi
      getf.d        GR_Arg = FR_Xf
      fcmp.eq.s1    p6,p0 = FR_NormX,FR_N
      mov           GR_ExpBias = 0x3FF
};;
{ .mfi
      nop.m         0
      fma.s1        FR_T = FR_int_Ln,FR_Ln2,FR_T
      extr.u        GR_Exp = GR_Arg,52,11
}
{ .mfi
      nop.m         0
      fma.s1        FR_P32 = FR_P32,FR_r2,FR_P10
      nop.i         0
};;
{ .mfi
      sub           GR_PureExp = GR_Exp,GR_ExpBias
      fma.s1        FR_S14 = FR_S16,FR_Xf2,FR_S14
      extr.u        GR_Ind4T = GR_Arg,44,8
}
{ .mfb
      mov           GR_SignOfGamma = 1 // set signgam to -1
      fma.s1        FR_S10 = FR_S12,FR_Xf2,FR_S10
(p6)  br.cond.spnt  lgammaf_singularity
};;
{ .mfi
      setf.sig      FR_int_Ln = GR_PureExp
      fms.s1        FR_rf = FR_InvXf,FR_Xf,f1
      // set p14 if GR_N is even
      tbit.z        p14,p0 = GR_N,0
}
{ .mfi
      shladd        GR_ad_T = GR_Ind4T,3,GR_ad_Data
      fma.s1        FR_Xf4 = FR_Xf2,FR_Xf2,f0
      nop.i         0
};;
{ .mfi
(p14) sub           GR_SignOfGamma = r0,GR_SignOfGamma // set signgam to -1
      fma.s1        FR_S6 = FR_S8,FR_Xf2,FR_S6
      nop.i         0
}
{ .mfi
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
      fma.s1        FR_S2 = FR_S4,FR_Xf2,FR_S2
      nop.i         0
};;
{ .mfi
      ldfd          FR_Tf = [GR_ad_T]
      fma.s1        FR_Ln = FR_P32,FR_r,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_LnSqrt2Pi = FR_LnSqrt2Pi,f1,FR_NormX
      nop.i         0
};;
.pred.rel "mutex",p9,p10
{ .mfi
(p9)  st4           [r33] = GR_SignOfGamma  // as 32-bit int
      fma.s1        FR_rf2 = FR_rf,FR_rf,f0
      nop.i         0
}
{ .mfi
(p10) st8           [r33] = GR_SignOfGamma  // as 64-bit int
      fma.s1        FR_S10 = FR_S14,FR_Xf4,FR_S10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P32f = FR_P3,FR_rf,FR_P2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_Xf8 = FR_Xf4,FR_Xf4,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P10f = FR_P1,FR_rf,f1
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S2 = FR_S6,FR_Xf4,FR_S2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_Ln = FR_Ln,FR_xm05,FR_LnSqrt2Pi
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_Nf = FR_int_Ln
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S2 = FR_S10,FR_Xf8,FR_S2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Tf = FR_Nf,FR_Ln2,FR_Tf
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P32f = FR_P32f,FR_rf2,FR_P10f // ??????
      nop.i         0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_Ln = FR_S2,FR_Xf2,FR_Ln
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Lnf = FR_P32f,FR_rf,FR_Tf
      nop.i         0
};;
{ .mfb
      nop.m         0
      fms.s.s0      f8 = FR_Ln,f1,FR_Lnf
      br.ret.sptk   b0
};;
// branch for calculating of ln(GAMMA(x)) for -2^13 < x < -9
//---------------------------------------------------------------------
.align 32
lgammaf_negpoly:
{ .mfi
      getf.d        GR_Arg = FR_Xf
      frcpa.s1      FR_InvXf,p0 = f1,FR_Xf
      mov           GR_ExpBias = 0x3FF
}
{ .mfi
      nop.m         0
      fma.s1        FR_Xf2 = FR_Xf,FR_Xf,f0
      nop.i         0
};;
{ .mfi
      getf.sig      GR_N = FR_int_Ntrunc
      fcvt.xf       FR_N = FR_int_Ln
      mov           GR_SignOfGamma = 1
}
{ .mfi
      nop.m         0
      fma.s1        FR_A9 = FR_A10,FR_x,FR_A9
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P10 = FR_P1,FR_r,f1
      extr.u        GR_Exp = GR_Arg,52,11
}
{ .mfi
      nop.m         0
      fma.s1        FR_x4 = FR_x2,FR_x2,f0
      nop.i         0
};;
{ .mfi
      sub           GR_PureExp = GR_Exp,GR_ExpBias
      fma.s1        FR_A7 = FR_A8,FR_x,FR_A7
      tbit.z        p14,p0 = GR_N,0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A5 = FR_A6,FR_x,FR_A5
      nop.i         0
};;
{ .mfi
      setf.sig      FR_int_Ln = GR_PureExp
      fma.s1        FR_A3 = FR_A4,FR_x,FR_A3
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A1 = FR_A2,FR_x,FR_A1
(p14) sub           GR_SignOfGamma = r0,GR_SignOfGamma
};;
{ .mfi
      nop.m         0
      fms.s1        FR_rf = FR_InvXf,FR_Xf,f1
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_Xf4 = FR_Xf2,FR_Xf2,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S14 = FR_S16,FR_Xf2,FR_S14
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S10 = FR_S12,FR_Xf2,FR_S10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_T = FR_N,FR_Ln2,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P32 = FR_P32,FR_r2,FR_P10
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S6 = FR_S8,FR_Xf2,FR_S6
      extr.u        GR_Ind4T = GR_Arg,44,8
}
{ .mfi
      nop.m         0
      fma.s1        FR_S2 = FR_S4,FR_Xf2,FR_S2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A7 = FR_A9,FR_x2,FR_A7
      nop.i         0
}
{ .mfi
      shladd        GR_ad_T = GR_Ind4T,3,GR_ad_Data
      fma.s1        FR_A3 = FR_A5,FR_x2,FR_A3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Xf8 = FR_Xf4,FR_Xf4,f0
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_rf2 = FR_rf,FR_rf,f0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P32f = FR_P3,FR_rf,FR_P2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_P10f = FR_P1,FR_rf,f1
      nop.i         0
};;
{ .mfi
      ldfd          FR_Tf = [GR_ad_T]
      fma.s1        FR_Ln = FR_P32,FR_r,FR_T
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A0 = FR_A1,FR_x,FR_A0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_S10 = FR_S14,FR_Xf4,FR_S10
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_S2 = FR_S6,FR_Xf4,FR_S2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_Nf = FR_int_Ln
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A3 = FR_A7,FR_x4,FR_A3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcmp.eq.s1    p13,p0 = FR_NormX,FR_Ntrunc
      nop.i         0
}
{ .mfi
      nop.m         0
      fnma.s1       FR_x3 = FR_x2,FR_x,f0 // -x^3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_P32f = FR_P32f,FR_rf2,FR_P10f
      nop.i         0
};;
{ .mfb
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
      fma.s1        FR_S2 = FR_S10,FR_Xf8,FR_S2
(p13) br.cond.spnt  lgammaf_singularity
};;
.pred.rel "mutex",p9,p10
{ .mmf
(p9)  st4           [r33] = GR_SignOfGamma  // as 32-bit int
(p10) st8           [r33] = GR_SignOfGamma  // as 64-bit int
      fms.s1        FR_A0 = FR_A3,FR_x3,FR_A0 // -A3*x^3-A0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Tf = FR_Nf,FR_Ln2,FR_Tf
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Ln = FR_S2,FR_Xf2,FR_Ln // S2*Xf^2+Ln
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_Lnf = FR_P32f,FR_rf,FR_Tf
      nop.i         0
};;
{ .mfi
      nop.m         0
      fms.s1        FR_Ln = FR_A0,f1,FR_Ln
      nop.i         0
};;
{ .mfb
      nop.m         0
      fms.s.s0      f8 = FR_Ln,f1,FR_Lnf
      br.ret.sptk   b0
};;
// branch for handling +/-0, NaT, QNaN, +/-INF and denormalised numbers
//---------------------------------------------------------------------
.align 32
lgammaf_spec:
{ .mfi
      getf.exp      GR_SignExp = FR_NormX
      fclass.m      p6,p0 = f8,0x21 // is arg +INF?
      mov           GR_SignOfGamma = 1 // set signgam to 1
};;
{ .mfi
      getf.sig      GR_Sig = FR_NormX
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
      dep.z         GR_Ind = GR_SignExp,3,4
}
{ .mib
      // store sign of gamma(x) as 64-bit int
(p12) st8           [r33] = GR_SignOfGamma
      and           GR_Exp = GR_ExpMask,GR_SignExp
(p6)  br.ret.spnt   b0 // exit for +INF
};;
{ .mfi
      sub           GR_PureExp = GR_Exp,GR_ExpBias
      fclass.m      p9,p0 = f8,0x22 // is arg -INF?
      extr.u        GR_Ind4T = GR_Sig,55,8
}
{ .mfb
      nop.m         0
(p7)  fma.s0        FR_tmp = f1,f1,f8
(p7)  br.cond.sptk  lgammaf_core
};;
{ .mfb
      nop.m         0
(p8)  fms.s.s0      f8 = f8,f1,f8
(p8)  br.ret.spnt   b0 // exit for NaT and NaN
};;
{ .mfb
      nop.m         0
(p9)  fmerge.s      f8 = f1,f8
(p9)  br.ret.spnt   b0 // exit -INF
};;
// branch for handling negative integers and +/-0
//---------------------------------------------------------------------
.align 32
lgammaf_singularity:
{ .mfi
      mov           GR_SignOfGamma = 1 // set signgam to 1
      fclass.m      p6,p0 = f8,0x6 // is x -0?
      mov           GR_TAG = 109 // negative
}
{ .mfi
      mov           GR_ad_SignGam = r33
      fma.s1        FR_X = f0,f0,f8
      nop.i         0
};;
{ .mfi
      nop.m         0
      frcpa.s0      f8,p0 = f1,f0
      // set p9  if signgum is 32-bit int
      // set p10 if signgum is 64-bit int
      cmp.eq        p10,p9 = 8,r34
}
{ .mib
      nop.m         0
(p6)  sub           GR_SignOfGamma = r0,GR_SignOfGamma
      br.cond.sptk  lgammaf_libm_err
};;
// overflow (x > OVERFLOV_BOUNDARY)
//---------------------------------------------------------------------
.align 32
lgammaf_overflow:
{ .mfi
      nop.m         0
      nop.f         0
      mov           r8 = 0x1FFFE
};;
{ .mfi
      setf.exp      f9 = r8
      fmerge.s      FR_X = f8,f8
      mov           GR_TAG = 108 // overflow
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
      fma.s.s0      f8 = f9,f9,f0 // Set I,O and +INF result
      nop.i         0
};;
// gate to __libm_error_support#
//---------------------------------------------------------------------
.align 32
lgammaf_libm_err:
{ .mmi
      alloc        r32 = ar.pfs,1,4,4,0
      mov          GR_Parameter_TAG = GR_TAG
      nop.i        0
};;
.pred.rel "mutex",p9,p10
{ .mmi
      // store sign of gamma(x) as 32-bit int
(p9)  st4          [GR_ad_SignGam] = GR_SignOfGamma
      // store sign of gamma(x) as 64-bit int
(p10) st8          [GR_ad_SignGam] = GR_SignOfGamma
      nop.i        0
};;
GLOBAL_LIBM_END(__libm_lgammaf)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
      add   GR_Parameter_Y=-32,sp             // Parameter 2 value
      nop.f 0
.save ar.pfs,GR_SAVE_PFS
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
      stfs [GR_Parameter_X] = FR_X                  // STORE Parameter 1
                                                    // on stack
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
      nop.b 0
}
{ .mib
      stfs [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3
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
