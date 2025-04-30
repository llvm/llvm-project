.file "erfcf.s"


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
// 01/17/02  Initial version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 02/06/03  Reordered header: .section, .global, .proc, .align
// 03/31/05  Reformatted delimiters between data tables
//
// API
//==============================================================
// float erfcf(float)
//
// Overview of operation
//==============================================================
// 1. 0 <= x <= 10.06
//
//    erfcf(x)  = P15(x) * exp( -x^2 )
//
//    Comment:
//
//    Let x(0)=0, x(i) = 2^(i), i=1,...3, x(4)= 10.06
//
//    Let x(i)<= x < x(i+1).
//    We can find i as exponent of argument x (let i = 0 for 0<= x < 2  )
//
//    Let P15(x) - polynomial approximation of degree 15 for function
//    erfcf(x) * exp( x^2) and x(i) <= x <= x(i+1), i = 0,1,2,3
//    Polynomial coefficients we have in the table erfc_p_table.
//
//    So we can find result for erfcf(x) as above.
//    Algorithm description for exp function see below.
//
// 2. -4.4 <= x < 0
//
//    erfcf(x)  = 2.0 - erfcf(-x)
//
// 3. x > 10.06
//
//    erfcf(x)  ~=~ 0.0
//
// 4. x < -4.4
//
//    erfcf(x)  ~=~ 2.0

// Special values
//==============================================================
// erfcf(+0)    = 1.0
// erfcf(-0)    = 1.0

// erfcf(+qnan) = +qnan
// erfcf(-qnan) = -qnan
// erfcf(+snan) = +qnan
// erfcf(-snan) = -qnan

// erfcf(-inf)  = 2.0
// erfcf(+inf)  = +0

//==============================================================
// Take double exp(double) from libm_64.
//
// Overview of operation
//==============================================================
// Take the input x. w is "how many log2/128 in x?"
//  w = x * 128/log2
//  n = int(w)
//  x = n log2/128 + r + delta

//  n = 128M + index_1 + 2^4 index_2
//  x = M log2 + (log2/128) index_1 + (log2/8) index_2 + r + delta

//  exp(x) = 2^M  2^(index_1/128)  2^(index_2/8) exp(r) exp(delta)
//       Construct 2^M
//       Get 2^(index_1/128) from table_1;
//       Get 2^(index_2/8)   from table_2;
//       Calculate exp(r) by series
//          r = x - n (log2/128)_high
//          delta = - n (log2/128)_low
//       Calculate exp(delta) as 1 + delta
//
// Comment for erfcf:
//
// Let exp(r) = 1 + x + 0.5*x^2 + (1/6)*x^3
// Let delta  = 0.
//==============================================================
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f6,f7,f9 -> f11,  f32 -> f92

// General registers used:
// r14 -> r22,r32 -> r50

// Predicate registers used:
// p6 -> p15

// Assembly macros
//==============================================================
EXP_AD_TB1             = r14
exp_GR_sig_inv_ln2     = r15
exp_TB1_size           = r16
exp_GR_rshf_2to56      = r17
exp_GR_exp_2tom56      = r18

exp_GR_rshf            = r33
EXP_AD_TB2             = r34
EXP_AD_P               = r35
exp_GR_N               = r36
exp_GR_index_1         = r37
exp_GR_index_2_16      = r38
exp_GR_biased_M        = r39
EXP_AD_T1              = r40
EXP_AD_T2              = r41
exp_TB2_size           = r42

// GR for erfcf(x)
//==============================================================
GR_IndxPlusBias        = r19
GR_ExpMask             = r20
GR_BIAS                = r21
GR_ShftPi_bias         = r22

GR_P_POINT_1           = r43
GR_P_POINT_2           = r44
GR_P_POINT_3           = r45
GR_P_POINT_4           = r46

GR_ShftPi              = r47
GR_EpsNorm             = r48

GR_05                  = r49
GR_1_by_6              = r50

// GR for __libm_support call
//==============================================================

GR_SAVE_B0             = r43
GR_SAVE_PFS            = r44
GR_SAVE_GP             = r45
GR_SAVE_SP             = r46

GR_Parameter_X         = r47
GR_Parameter_Y         = r48
GR_Parameter_RESULT    = r49
GR_Parameter_TAG       = r50


// FR for exp(-x^2)
//==============================================================
FR_X                   = f10
FR_Y                   = f1
FR_RESULT              = f8

EXP_2TOM56             = f6
EXP_INV_LN2_2TO63      = f7
EXP_W_2TO56_RSH        = f9
exp_ln2_by_128_hi      = f11

EXP_RSHF_2TO56         = f32
exp_ln2_by_128_lo      = f33
EXP_RSHF               = f34
EXP_Nfloat             = f35
exp_r                  = f36
exp_rsq                = f37
EXP_2M                 = f38
exp_S1                 = f39
exp_T1                 = f40
exp_P                  = f41
exp_S                  = f42
EXP_NORM_f8            = f43
exp_S2                 = f44
exp_T2                 = f45

// FR for erfcf(x)
//==============================================================
FR_AbsArg              = f46
FR_Tmp                 = f47
FR_Tmp1                = f48
FR_Tmpf                = f49
FR_NormX               = f50

FR_A15                 = f51
FR_A14                 = f52

FR_A13                 = f53
FR_A12                 = f54

FR_A11                 = f55
FR_A10                 = f56

FR_A9                  = f57
FR_A8                  = f58

FR_A7                  = f59
FR_A6                  = f60

FR_A5                  = f61
FR_A4                  = f62

FR_A3                  = f63
FR_A2                  = f64

FR_A1                  = f65
FR_A0                  = f66

FR_P15_0_1             = f67
FR_P15_1_1             = f68
FR_P15_1_2             = f69
FR_P15_2_1             = f70
FR_P15_2_2             = f71
FR_P15_3_1             = f72
FR_P15_3_2             = f73
FR_P15_4_1             = f74
FR_P15_4_2             = f75
FR_P15_7_1             = f76
FR_P15_7_2             = f77
FR_P15_8_1             = f78
FR_P15_9_1             = f79
FR_P15_9_2             = f80
FR_P15_13_1            = f81
FR_P15_14_1            = f82
FR_P15_14_2            = f83

FR_2                   = f84
FR_05                  = f85
FR_1_by_6              = f86
FR_Pol                 = f87
FR_Exp                 = f88

FR_POS_ARG_ASYMP       = f89
FR_NEG_ARG_ASYMP       = f90

FR_UnfBound            = f91
FR_EpsNorm             = f92

// Data tables
//==============================================================
RODATA
.align 16

// ************* DO NOT CHANGE ORDER OF THESE TABLES ********************

// double-extended 1/ln(2)
// 3fff b8aa 3b29 5c17 f0bb be87fed0691d3e88
// 3fff b8aa 3b29 5c17 f0bc
// For speed the significand will be loaded directly with a movl and setf.sig
//   and the exponent will be bias+63 instead of bias+0.  Thus subsequent
//   computations need to scale appropriately.
// The constant 128/ln(2) is needed for the computation of w.  This is also
//   obtained by scaling the computations.
//
// Two shifting constants are loaded directly with movl and setf.d.
//   1. EXP_RSHF_2TO56 = 1.1000..00 * 2^(63-7)
//        This constant is added to x*1/ln2 to shift the integer part of
//        x*128/ln2 into the rightmost bits of the significand.
//        The result of this fma is EXP_W_2TO56_RSH.
//   2. EXP_RSHF       = 1.1000..00 * 2^(63)
//        This constant is subtracted from EXP_W_2TO56_RSH * 2^(-56) to give
//        the integer part of w, n, as a floating-point number.
//        The result of this fms is EXP_Nfloat.


LOCAL_OBJECT_START(exp_table_1)

data4 0x4120f5c3, 0x408ccccd      //POS_ARG_ASYMP = 10.06, NEG_ARG_ASYMP = 4.4
data4 0x41131Cdf, 0x00800000     //UnfBound ~=~ 9.1, EpsNorm ~=~ 1.1754944e-38
//
data8 0xb17217f7d1cf79ab , 0x00003ff7                            // ln2/128 hi
data8 0xc9e3b39803f2f6af , 0x00003fb7                            // ln2/128 lo
//
// Table 1 is 2^(index_1/128) where
// index_1 goes from 0 to 15
//
data8 0x8000000000000000 , 0x00003FFF
data8 0x80B1ED4FD999AB6C , 0x00003FFF
data8 0x8164D1F3BC030773 , 0x00003FFF
data8 0x8218AF4373FC25EC , 0x00003FFF
data8 0x82CD8698AC2BA1D7 , 0x00003FFF
data8 0x8383594EEFB6EE37 , 0x00003FFF
data8 0x843A28C3ACDE4046 , 0x00003FFF
data8 0x84F1F656379C1A29 , 0x00003FFF
data8 0x85AAC367CC487B15 , 0x00003FFF
data8 0x8664915B923FBA04 , 0x00003FFF
data8 0x871F61969E8D1010 , 0x00003FFF
data8 0x87DB357FF698D792 , 0x00003FFF
data8 0x88980E8092DA8527 , 0x00003FFF
data8 0x8955EE03618E5FDD , 0x00003FFF
data8 0x8A14D575496EFD9A , 0x00003FFF
data8 0x8AD4C6452C728924 , 0x00003FFF
LOCAL_OBJECT_END(exp_table_1)

// Table 2 is 2^(index_1/8) where
// index_2 goes from 0 to 7

LOCAL_OBJECT_START(exp_table_2)

data8 0x8000000000000000 , 0x00003FFF
data8 0x8B95C1E3EA8BD6E7 , 0x00003FFF
data8 0x9837F0518DB8A96F , 0x00003FFF
data8 0xA5FED6A9B15138EA , 0x00003FFF
data8 0xB504F333F9DE6484 , 0x00003FFF
data8 0xC5672A115506DADD , 0x00003FFF
data8 0xD744FCCAD69D6AF4 , 0x00003FFF
data8 0xEAC0C6E7DD24392F , 0x00003FFF
LOCAL_OBJECT_END(exp_table_2)

LOCAL_OBJECT_START(erfc_p_table)

// Pol_0
data8 0xBEA3260C63CB0446             //A15 = -5.70673541831883454676e-07
data8 0x3EE63D6178077654             //A14 = +1.06047480138940182343e-05
data8 0xBF18646BC5FC70A7             //A13 = -9.30491237309283694347e-05
data8 0x3F40F92F909117FE             //A12 = +5.17986512144075019133e-04
data8 0xBF611344289DE1E6             //A11 = -2.08438217390159994419e-03
data8 0x3F7AF9FE6AD16DC0             //A10 = +6.58606893292862351928e-03
data8 0xBF91D219E196CBA7             //A9 = -1.74030345858217321001e-02
data8 0x3FA4AFDDA355854C             //A8 = +4.04042493708041968315e-02
data8 0xBFB5D465BB7025AE             //A7 = -8.52721769916999425445e-02
data8 0x3FC54C15A95B717D             //A6 = +1.66384418195672549029e-01
data8 0xBFD340A75B4B1AB5             //A5 = -3.00821150926292166899e-01
data8 0x3FDFFFC0BFCD247F             //A4 = +4.99984919839853542841e-01
data8 0xBFE81270C361852B             //A3 = -7.52251035312075583309e-01
data8 0x3FEFFFFFC67295FC             //A2 = +9.99999892800303301771e-01
data8 0xBFF20DD74F8CD2BF             //A1 = -1.12837916445020868099e+00
data8 0x3FEFFFFFFFFE7C1D             //A0 = +9.99999999988975570714e-01
// Pol_1
data8 0xBDE8EC4BDD953B56             //A15 = -1.81338928934942767144e-10
data8 0x3E43607F269E2A1C             //A14 = +9.02309090272196442358e-09
data8 0xBE8C4D9E69C10E02             //A13 = -2.10875261143659275328e-07
data8 0x3EC9CF2F84566725             //A12 = +3.07671055805877356583e-06
data8 0xBF007980B1B46A4D             //A11 = -3.14228438702169818945e-05
data8 0x3F2F4C3AD6DEF24A             //A10 = +2.38783056770846320260e-04
data8 0xBF56F5129F8D30FA             //A9 = -1.40120333363130546426e-03
data8 0x3F7AA6C7ABFC38EE             //A8 = +6.50671002200751820429e-03
data8 0xBF98E7522CB84BEF             //A7 = -2.43199195666185511109e-02
data8 0x3FB2F68EB1C3D073             //A6 = +7.40746673580490638637e-02
data8 0xBFC7C16055AC6385             //A5 = -1.85588876564704611769e-01
data8 0x3FD8A707AEF5A440             //A4 = +3.85194702967570635211e-01
data8 0xBFE547BFE39AE2EA             //A3 = -6.65008492032112467310e-01
data8 0x3FEE7C91BDF13578             //A2 = +9.52706213932898128515e-01
data8 0xBFF1CB5B61F8C589             //A1 = -1.11214769621105541214e+00
data8 0x3FEFEA56BC81FD37             //A0 = +9.97355812243688815239e-01
// Pol_2
data8 0xBD302724A12F46E0             //A15 = -5.73866382814058809406e-14
data8 0x3D98889B75D3102E             //A14 = +5.57829983681360947356e-12
data8 0xBDF16EA15074A1E9             //A13 = -2.53671153922423457844e-10
data8 0x3E3EC6E688CFEE5F             //A12 = +7.16581828336436419561e-09
data8 0xBE82E5ED44C52609             //A11 = -1.40802202239825487803e-07
data8 0x3EC120BE5CE42353             //A10 = +2.04180535157522081699e-06
data8 0xBEF7B8B0311A1911             //A9 = -2.26225266204633600888e-05
data8 0x3F29A281F43FC238             //A8 = +1.95577968156184077632e-04
data8 0xBF55E19858B3B7A4             //A7 = -1.33552434527526534043e-03
data8 0x3F7DAC8C3D12E5FD             //A6 = +7.24463253680473816303e-03
data8 0xBF9FF9C04613FB47             //A5 = -3.12261622211693854028e-02
data8 0x3FBB3D5DBF9D9366             //A4 = +1.06405123978743883370e-01
data8 0xBFD224DE9F62C258             //A3 = -2.83500342989133623476e-01
data8 0x3FE28A95CB8C6D3E             //A2 = +5.79417131000276437708e-01
data8 0xBFEC21205D358672             //A1 = -8.79043752717008257224e-01
data8 0x3FEDAE44D5EDFE5B             //A0 = +9.27523057776805771830e-01
// Pol_3
data8 0xBCA3BCA734AC82F1             //A15 = -1.36952437983096410260e-16
data8 0x3D16740DC3990612             //A14 = +1.99425676175410093285e-14
data8 0xBD77F4353812C46A             //A13 = -1.36162367755616790260e-12
data8 0x3DCFD0BE13C73DB4             //A12 = +5.78718761040355136007e-11
data8 0xBE1D728DF71189B4             //A11 = -1.71406885583934105120e-09
data8 0x3E64252C8CB710B5             //A10 = +3.75233795940731111303e-08
data8 0xBEA514B93180F33D             //A9 = -6.28261292774310809962e-07
data8 0x3EE1381118CC7151             //A8 = +8.21066421390821904504e-06
data8 0xBF1634404FB0FA72             //A7 = -8.47019436358372148764e-05
data8 0x3F46B2CBBCF0EB32             //A6 = +6.92700845213200923490e-04
data8 0xBF725C2B445E6D81             //A5 = -4.48243046949004063741e-03
data8 0x3F974E7CFA4D89D9             //A4 = +2.27603462002522228717e-02
data8 0xBFB6D7BAC2E342D1             //A3 = -8.92292714882032736443e-02
data8 0x3FD0D156AD9CE2A6             //A2 = +2.62777013343603696631e-01
data8 0xBFE1C228572AADB0             //A1 = -5.54950876471982857725e-01
data8 0x3FE8A739F48B9A3B             //A0 = +7.70413377406675619766e-01
LOCAL_OBJECT_END(erfc_p_table)


.section .text
GLOBAL_LIBM_ENTRY(erfcf)

// Form index i for table erfc_p_table as exponent of x
// We use i + bias in real calculations
{ .mlx
      getf.exp       GR_IndxPlusBias = f8          // (sign + exp + bias) of x
      movl           exp_GR_sig_inv_ln2 = 0xb8aa3b295c17f0bc //signif.of 1/ln2
}
{ .mlx
      addl           EXP_AD_TB1    = @ltoff(exp_table_1), gp
      movl           exp_GR_rshf_2to56 = 0x4768000000000000 // 1.100 2^(63+56)
}
;;

// Form argument EXP_NORM_f8 for exp(-x^2)
{ .mfi
      ld8            EXP_AD_TB1    = [EXP_AD_TB1]
      fcmp.ge.s1     p6,p7 = f8, f0                     // p6: x >= 0 ,p7: x<0
      mov            GR_BIAS = 0x0FFFF
}
{ .mfi
      mov            exp_GR_exp_2tom56 = 0xffff-56
      fnma.s1        EXP_NORM_f8   = f8, f8, f0                       //  -x^2
      mov            GR_ExpMask  = 0x1ffff
}
;;

// Form two constants we need
//  1/ln2 * 2^63  to compute  w = x * 1/ln2 * 128
//  1.1000..000 * 2^(63+63-7) to right shift int(w) into the significand

// p9:  x = 0,+inf,-inf,nan,unnorm.
// p10: x!= 0,+inf,-inf,nan,unnorm.
{ .mfi
      setf.sig       EXP_INV_LN2_2TO63 = exp_GR_sig_inv_ln2 // Form 1/ln2*2^63
      fclass.m       p9,p10 = f8,0xef
      shl            GR_ShftPi_bias = GR_BIAS, 7
}
{ .mfi
      setf.d         EXP_RSHF_2TO56 = exp_GR_rshf_2to56 //Const 1.10*2^(63+56)
      nop.f          0
      and            GR_IndxPlusBias = GR_IndxPlusBias, GR_ExpMask // i + bias
}
;;

{ .mfi
      alloc          r32 = ar.pfs, 0, 15, 4, 0
(p6)  fma.s1         FR_AbsArg = f1, f0, f8                  // |x| if x >= 0
      cmp.lt         p15,p0 = GR_IndxPlusBias, GR_BIAS//p15: i < 0 (for |x|<1)
}
{ .mlx
      setf.exp       EXP_2TOM56 = exp_GR_exp_2tom56 //2^-56 for scaling Nfloat
      movl           exp_GR_rshf = 0x43e8000000000000 //1.10 2^63,right shift.
}
;;

{ .mfi
      ldfps          FR_POS_ARG_ASYMP, FR_NEG_ARG_ASYMP = [EXP_AD_TB1],8
      nop.f          0
(p15) mov            GR_IndxPlusBias = GR_BIAS            //Let i = 0 if i < 0
}
{ .mlx
      mov            GR_P_POINT_3 = 0x1A0
      movl           GR_05 = 0x3fe0000000000000
}
;;

// Form shift GR_ShftPi from the beginning of erfc_p_table
// to the polynomial with number i
{ .mfi
      ldfps          FR_UnfBound, FR_EpsNorm = [EXP_AD_TB1],8
      nop.f          0
      shl            GR_ShftPi = GR_IndxPlusBias, 7
}
{ .mfi
      setf.d         EXP_RSHF = exp_GR_rshf   // Form right shift 1.100 * 2^63
(p7)  fms.s1         FR_AbsArg = f1, f0, f8                   // |x|  if x < 0
      mov            exp_TB1_size  = 0x100
}
;;

// Form pointer GR_P_POINT_3 to the beginning of erfc_p_table
{ .mfi
      setf.d         FR_05 = GR_05
      nop.f          0
      sub            GR_ShftPi = GR_ShftPi,GR_ShftPi_bias
}
{ .mfb
      add            GR_P_POINT_3 = GR_P_POINT_3, EXP_AD_TB1
      nop.f          0
(p9)  br.cond.spnt   SPECIAL                  // For x = 0,+inf,-inf,nan,unnorm
}
;;

{ .mfi
      add            GR_P_POINT_1 = GR_P_POINT_3, GR_ShftPi
      nop.f          0
      add            GR_P_POINT_2 = GR_P_POINT_3, GR_ShftPi
}
{ .mfi
      ldfe           exp_ln2_by_128_hi  = [EXP_AD_TB1],16
      fma.s1         FR_NormX = f8,f1,f0
      add            GR_P_POINT_3 = GR_P_POINT_3, GR_ShftPi
}
;;

// Load coefficients for polynomial P15(x)
{ .mfi
      ldfpd          FR_A15, FR_A14 = [GR_P_POINT_1], 16
      nop.f          0
      add            GR_P_POINT_3 = 0x30, GR_P_POINT_3
}
{ .mfi
      ldfe           exp_ln2_by_128_lo  = [EXP_AD_TB1], 16
      nop.f          0
      add            GR_P_POINT_2 = 0x20, GR_P_POINT_2
}
;;

// Now EXP_AD_TB1 points to the beginning of table 1
{ .mlx
      ldfpd          FR_A13, FR_A12 = [GR_P_POINT_1]
      movl           GR_1_by_6 = 0x3FC5555555555555
}
{ .mfi
      add            GR_P_POINT_4 = 0x30, GR_P_POINT_2
      nop.f          0
      nop.i          0
}
;;

{ .mfi
      ldfpd          FR_A11, FR_A10 = [GR_P_POINT_2]
      fma.s1         FR_2 = f1, f1, f1
      mov            exp_TB2_size  = 0x80
}
{ .mfi
      ldfpd          FR_A9, FR_A8 = [GR_P_POINT_3],16
      nop.f          0
      add            GR_P_POINT_1 = 0x60 ,GR_P_POINT_1
}
;;

// W = X * Inv_log2_by_128
// By adding 1.10...0*2^63 we shift and get round_int(W) in significand.
// We actually add 1.10...0*2^56 to X * Inv_log2 to do the same thing.
{ .mfi
      ldfpd          FR_A7, FR_A6 = [GR_P_POINT_3]
      fma.s1     EXP_W_2TO56_RSH = EXP_NORM_f8,EXP_INV_LN2_2TO63,EXP_RSHF_2TO56
      add            EXP_AD_TB2 = exp_TB1_size, EXP_AD_TB1

}
{ .mfi
      ldfpd          FR_A5, FR_A4 = [GR_P_POINT_4], 16
      nop.f          0
      nop.i          0
}
;;

{ .mfi
      ldfpd          FR_A3, FR_A2 = [GR_P_POINT_4]
      fmerge.s       FR_X = f8,f8
      nop.i          0
}
{ .mfi
      ldfpd          FR_A1, FR_A0 = [GR_P_POINT_1]
      nop.f          0
      nop.i          0
}
;;

//p14: x < - NEG_ARG_ASYMP = -4.4 -> erfcf(x) ~=~ 2.0
{ .mfi
      setf.d         FR_1_by_6  = GR_1_by_6
(p7)  fcmp.gt.unc.s1 p14,p0 = FR_AbsArg, FR_NEG_ARG_ASYMP          //p7: x < 0
      nop.i          0
}
;;

//p15: x > POS_ARG_ASYMP = 10.06 -> erfcf(x) ~=~ 0.0
{ .mfi
      nop.m          0
(p6)  fcmp.gt.unc.s1 p15,p0 = FR_AbsArg, FR_POS_ARG_ASYMP          //p6: x > 0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fcmp.le.s1     p8,p0 = FR_NormX, FR_UnfBound        // p8: x <= UnfBound
      nop.i          0
}
{ .mfb
      nop.m          0
(p14) fnma.s.s0      FR_RESULT = FR_EpsNorm, FR_EpsNorm, FR_2//y = 2 if x <-4.4
(p14) br.ret.spnt    b0
}
;;

// Nfloat = round_int(W)
// The signficand of EXP_W_2TO56_RSH contains the rounded integer part of W,
// as a twos complement number in the lower bits (that is, it may be negative).
// That twos complement number (called N) is put into exp_GR_N.

// Since EXP_W_2TO56_RSH is scaled by 2^56, it must be multiplied by 2^-56
// before the shift constant 1.10000 * 2^63 is subtracted to yield EXP_Nfloat.
// Thus, EXP_Nfloat contains the floating point version of N

{ .mfi
      nop.m          0
      fms.s1         EXP_Nfloat = EXP_W_2TO56_RSH, EXP_2TOM56, EXP_RSHF
      nop.i          0
}
{ .mfb
(p15) mov            GR_Parameter_TAG = 209
(p15) fma.s.s0       FR_RESULT = FR_EpsNorm,FR_EpsNorm,f0 //Result.for x>10.06
(p15) br.cond.spnt   __libm_error_region
}
;;

// Now we can calculate polynomial P15(x)
{ .mfi
      nop.m          0
      fma.s1         FR_P15_1_1 = FR_AbsArg, FR_AbsArg, f0             // x ^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_0_1 = FR_A15, FR_AbsArg, FR_A14
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_1_2 = FR_A13, FR_AbsArg, FR_A12
      nop.i          0
}
;;

{ .mfi
      getf.sig       exp_GR_N        = EXP_W_2TO56_RSH
      fma.s1         FR_P15_2_1 = FR_A9, FR_AbsArg, FR_A8
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_2_2 = FR_A11, FR_AbsArg, FR_A10
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_P15_3_1 = FR_A5, FR_AbsArg, FR_A4
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_3_2 = FR_A7, FR_AbsArg, FR_A6
      nop.i          0
}
;;

// exp_GR_index_1 has index_1
// exp_GR_index_2_16 has index_2 * 16
// exp_GR_biased_M has M
// exp_GR_index_1_16 has index_1 * 16

// r2 has true M
{ .mfi
      and            exp_GR_index_1 = 0x0f, exp_GR_N
      fma.s1         FR_P15_4_1 = FR_A1, FR_AbsArg, FR_A0
      shr            r2 = exp_GR_N,  0x7

}
{ .mfi
      and            exp_GR_index_2_16 = 0x70, exp_GR_N
      fma.s1         FR_P15_4_2 = FR_A3, FR_AbsArg, FR_A2
      nop.i          0
}
;;

// EXP_AD_T1 has address of T1
// EXP_AD_T2 has address if T2

{ .mfi
      add            EXP_AD_T2 = EXP_AD_TB2, exp_GR_index_2_16
      nop.f          0
      shladd         EXP_AD_T1 = exp_GR_index_1, 4, EXP_AD_TB1
}
{ .mfi
      addl           exp_GR_biased_M = 0xffff, r2
      fnma.s1        exp_r   = EXP_Nfloat, exp_ln2_by_128_hi, EXP_NORM_f8
      nop.i          0
}
;;

// Create Scale = 2^M
// r = x - Nfloat * ln2_by_128_hi

{ .mfi
      setf.exp       EXP_2M = exp_GR_biased_M
      fma.s1         FR_P15_7_1 = FR_P15_0_1, FR_P15_1_1, FR_P15_1_2
      nop.i          0
}
{ .mfi
      ldfe           exp_T2  = [EXP_AD_T2]
      nop.f          0
      nop.i          0
}
;;

// Load T1 and T2

{ .mfi
      ldfe           exp_T1  = [EXP_AD_T1]
      fma.s1         FR_P15_7_2 = FR_P15_1_1, FR_P15_1_1, f0            // x^4
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
      fma.s1         FR_P15_9_1 = FR_P15_1_1, FR_P15_4_2, FR_P15_4_1
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
      fma.s1         exp_P = FR_1_by_6, exp_r, FR_05
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         exp_rsq = exp_r, exp_r, f0
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
      fma.s1         FR_P15_14_1 = FR_P15_7_2, FR_P15_9_2, FR_P15_9_1
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         FR_P15_14_2 = FR_P15_7_2, FR_P15_7_2, f0           // x^8
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         exp_P     = exp_P, exp_rsq, exp_r
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         exp_S1  = EXP_2M, exp_T2, f0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_Pol = FR_P15_14_2, FR_P15_13_1, FR_P15_14_1  // P15(x)
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         exp_S   = exp_S1, exp_T1, f0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         FR_Exp = exp_S, exp_P, exp_S                 // exp(-x^2)
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s.s0       FR_Tmpf = f8, f1, f0                          //  Flag  d
      nop.i          0
}
;;

//p6: result for     0 < x < = POS_ARG_ASYMP
//p7: result for   - NEG_ARG_ASYMP  <= x < 0
//p8: exit   for   - NEG_ARG_ASYMP <= x <= UnfBound, x!=0
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m          0
(p6)  fma.s.s0       f8 = FR_Exp, FR_Pol, f0
      nop.i          0
}
{ .mfb
      mov            GR_Parameter_TAG = 209
(p7)  fnma.s.s0      f8 = FR_Exp, FR_Pol, FR_2
(p8)  br.ret.sptk    b0
}
;;

//p10: branch for  UnfBound < x < = POS_ARG_ASYMP
{ .mfb
      nop.m          0
      nop.f          0
(p10) br.cond.spnt   __libm_error_region
}
;;

//Only via (p9)  br.cond.spnt   SPECIAL  for x = 0,+inf,-inf,nan,unnorm
SPECIAL:

{ .mfi
      nop.m          0
      fclass.m.unc   p10,p0 = f8,0x07                            // p10: x = 0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fclass.m.unc   p11,p0 = f8,0x21                         // p11: x = +inf
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fclass.m.unc   p12,p0 = f8,0x22                          // p12 x = -inf
      nop.i          0
}
{ .mfb
      nop.m          0
(p10) fma.s.s0       f8 = f1, f1, f0
(p10) br.ret.sptk    b0                                // Quick exit for x = 0
}
;;

{ .mfi
      nop.m          0
      fclass.m.unc   p13,p0 = f8,0xc3                          // p13: x = nan
      nop.i          0
}
{ .mfb
      nop.m          0
(p11) fma.s.s0       f8 = f0, f1, f0
(p11) br.ret.spnt    b0                             // Quick exit for x = +inf
}
;;
{ .mfi
      nop.m          0
      fclass.m.unc   p14,p0 = f8,0x0b                 // P14: x = unnormalized
      nop.i          0
}
{ .mfb
      nop.m          0
(p12) fma.s.s0       f8 = f1, f1, f1
(p12) br.ret.spnt    b0                             // Quick exit for x = -inf
}
;;

{ .mfb
      nop.m          0
(p13) fma.s.s0       f8 = f8, f1, f0
(p13) br.ret.sptk    b0                              // Quick exit for x = nan
}
;;

{ .mfb
      nop.m          0
(p14) fnma.s.s0      f8 = f8, f1, f1
(p14) br.ret.sptk    b0                     // Quick exit for x = unnormalized
}
;;

GLOBAL_LIBM_END(erfcf)
libm_alias_float_other (erfc, erfc)


// Call via (p10) br.cond.spnt   __libm_error_region
//          for  UnfBound < x < = POS_ARG_ASYMP
// and
//
// call via (p15) br.cond.spnt   __libm_error_region
//          for  x > POS_ARG_ASYMP

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
        stfs [GR_Parameter_Y] = FR_Y,16          // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp                      // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                                           // Save b0
};;
.body
{ .mib
        stfs [GR_Parameter_X] = FR_X             // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y    // Parameter 3 address
        nop.b 0
}
{ .mib
        stfs [GR_Parameter_Y] = FR_RESULT        // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT]        // Get return result off stack
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
