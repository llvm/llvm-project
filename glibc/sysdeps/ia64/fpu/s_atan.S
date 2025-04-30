.file "atan.s"


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
// History
//==============================================================
// 02/02/00  Initial version
// 04/13/00  Improved speed
// 04/19/00  Removed the qualifying predicate from the fmerge.s that
//           takes the absolute value.
// 06/16/00  Reassigned FP registers to eliminate stalls on loads
// 08/30/00  Saved 5 cycles in main path by rearranging large argument logic
//           and delaying use of result of fcmp in load by 1 group
// 05/20/02  Cleaned up namespace and sf0 syntax
// 08/20/02  Use atan2 algorithm with x=1 for better accuracy
// 02/06/03  Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// double atan(double Y)
//
// Overview of operation
//==============================================================
//
// The atan function returns values in the interval [-pi/2,+pi/2].
//
// The algorithm used is the atan2(Y,X) algorithm where we fix X=1.0.
//
// There are two basic paths: swap true and swap false.
// atan2(Y,X) ==> atan2(V/U) where U >= V. If Y > X, we must swap.
//
// p6  swap True    |Y| > |X|
// p7  swap False   |Y| <= |X|
//
//
// Simple trigonometric identities show
//   Region 1
//         |Y|<=1.0, V=Y, U=1.0     atan2(Y,X) = sgnY * (0 + atan(V/U))
//
//   Region 2
//         |Y|>1.0, V=1.0, U=Y      atan2(Y,X) = sgnY * (pi/2 - atan(V/U))
//
//
// We compute atan(V/U) from the identity
//      atan(z) + atan([(V/U)-z] / [1+(V/U)z])
//      where z is a limited precision approximation (16 bits) to V/U
//
// z is calculated with the assistance of the frcpa instruction.
//
// atan(z) is calculated by a polynomial z + z^3 * p(w),  w=z^2
// where p(w) = P0+P1*w+...+P22*w^22
//
// Let d = [(V/U)-z] / [1+(V/U)z]) = (V-U*z)/(U+V*z)
//
// Approximate atan(d) by d + P0*d^3
// Let F = 1/(U+V*z) * (1-a), where |a|< 2^-8.8.
// Compute q(a) = 1 + a + ... + a^5.
// Then F*q(a) approximates the reciprocal to more than 50 bits.

// Special values
//==============================================================
// atan(QNAN)  = QNAN
// atan(SNAN)  = quieted SNAN
// atan(+-inf) = +- pi/2
// atan(+-0)   = +-0

// Registers used
//==============================================================

// predicate registers used:
// p6 -> p15

// floating-point registers used:
// f8, input
// f32 -> f116

// general registers used
// r14 -> r16

// Assembly macros
//==============================================================

EXP_AD_P1                    = r14
EXP_AD_P2                    = r15
rsig_near_one                = r16

atan2_Y                      = f8
atan2_X                      = f1

atan2_u1_X                   = f32
atan2_u1_Y                   = f33
atan2_z2_X                   = f34

atan2_two                    = f36
atan2_B1sq_Y                 = f37
atan2_z1_X                   = f38
atan2_B1X                    = f40

atan2_B1Y                    = f41
atan2_wp_X                   = f42
atan2_B1sq_X                 = f43
atan2_z                      = f44
atan2_w                      = f45

atan2_P0                     = f46
atan2_P1                     = f47
atan2_P2                     = f48
atan2_P3                     = f49
atan2_P4                     = f50

atan2_P5                     = f51
atan2_P6                     = f52
atan2_P7                     = f53
atan2_P8                     = f54
atan2_P9                     = f55

atan2_P10                    = f56
atan2_P11                    = f57
atan2_P12                    = f58
atan2_P13                    = f59
atan2_P14                    = f60

atan2_P15                    = f61
atan2_P16                    = f62
atan2_P17                    = f63
atan2_P18                    = f64
atan2_P19                    = f65

atan2_P20                    = f66
atan2_P21                    = f67
atan2_P22                    = f68
atan2_pi_by_2                = f69
atan2_sgn_pi_by_2            = f69
atan2_V13                    = f70

atan2_W11                    = f71
atan2_E                      = f72
atan2_wp_Y                   = f73
atan2_V11                    = f74
atan2_V12                    = f75

atan2_V7                     = f76
atan2_V8                     = f77
atan2_W7                     = f78
atan2_W8                     = f79
atan2_W3                     = f80

atan2_W4                     = f81
atan2_V3                     = f82
atan2_V4                     = f83
atan2_F                      = f84
atan2_gV                     = f85

atan2_V10                    = f86
atan2_zcub                   = f87
atan2_V6                     = f88
atan2_V9                     = f89
atan2_W10                    = f90

atan2_W6                     = f91
atan2_W2                     = f92
atan2_V2                     = f93
atan2_alpha                  = f94
atan2_alpha_1                = f95

atan2_gVF                    = f96
atan2_V5                     = f97
atan2_W12                    = f98
atan2_W5                     = f99
atan2_alpha_sq               = f100

atan2_Cp                     = f101
atan2_V1                     = f102
atan2_ysq                    = f103
atan2_W1                     = f104
atan2_alpha_cub              = f105

atan2_C                      = f106
atan2_d                      = f108
atan2_A_hi                   = f109
atan2_dsq                    = f110

atan2_pd                     = f111
atan2_A_lo                   = f112
atan2_A                      = f113
atan2_Pp                     = f114
atan2_sgnY                   = f115

atan2_sig_near_one           = f116
atan2_near_one               = f116

/////////////////////////////////////////////////////////////


RODATA

.align 16

LOCAL_OBJECT_START(atan2_tb1)
data8 0xA21922DC45605EA1 ,  0x00003FFA // P11
data8 0xB199DD6D2675C40F ,  0x0000BFFA // P10
data8 0xC2F01E5DDD100DBE ,  0x00003FFA // P9
data8 0xD78F28FC2A592781 ,  0x0000BFFA // P8
data8 0xF0F03ADB3FC930D3 ,  0x00003FFA // P7
data8 0x88887EBB209E3543 ,  0x0000BFFB // P6
data8 0x9D89D7D55C3287A5 ,  0x00003FFB // P5
data8 0xBA2E8B9793955C77 ,  0x0000BFFB // P4
data8 0xE38E38E320A8A098 ,  0x00003FFB // P3
data8 0x9249249247E37913 ,  0x0000BFFC // P2
data8 0xCCCCCCCCCCC906CD ,  0x00003FFC // P1
data8 0xAAAAAAAAAAAAA8A9 ,  0x0000BFFD // P0
data8 0x0000000000000000 ,  0x00000000 // pad to avoid bank conflict
LOCAL_OBJECT_END(atan2_tb1)

LOCAL_OBJECT_START(atan2_tb2)
data8 0xCE585A259BD8374C ,  0x00003FF0 // P21
data8 0x9F90FB984D8E39D0 ,  0x0000BFF3 // P20
data8 0x9D3436AABE218776 ,  0x00003FF5 // P19
data8 0xDEC343E068A6D2A8 ,  0x0000BFF6 // P18
data8 0xF396268151CFB11C ,  0x00003FF7 // P17
data8 0xD818B4BB43D84BF2 ,  0x0000BFF8 // P16
data8 0xA2270D30A90AA220 ,  0x00003FF9 // P15
data8 0xD5F4F2182E7A8725 ,  0x0000BFF9 // P14
data8 0x80D601879218B53A ,  0x00003FFA // P13
data8 0x9297B23CCFFB291F ,  0x0000BFFA // P12
data8 0xFE7E52D2A89995B3 ,  0x0000BFEC // P22
data8 0xC90FDAA22168C235 ,  0x00003FFF // pi/2
LOCAL_OBJECT_END(atan2_tb2)




.section .text
GLOBAL_LIBM_ENTRY(atan)

{ .mfi
           nop.m 999
           frcpa.s1     atan2_u1_Y,p7 = f1,atan2_Y
           nop.i 999
}
{ .mfi
           addl         EXP_AD_P1   = @ltoff(atan2_tb1), gp
           fma.s1       atan2_two  = f1,f1,f1
           nop.i 999
;;
}

{ .mfi
           ld8  EXP_AD_P1 = [EXP_AD_P1]
           frcpa.s1     atan2_u1_X,p6 = f1,atan2_X
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_ysq  = atan2_Y,atan2_Y,f0
           nop.i 999
}
;;

{ .mfi
           add  EXP_AD_P2 = 0xd0,EXP_AD_P1
           fmerge.s     atan2_sgnY = atan2_Y,f1
           nop.i 999
}
;;


{ .mfi
           ldfe         atan2_P11  = [EXP_AD_P1],16
           fclass.m p10,p0 = atan2_Y, 0xc3     // Test for y=nan
           nop.i 999
}
{ .mfi
           ldfe         atan2_P21  = [EXP_AD_P2],16
           nop.f 999
           nop.i 999
;;
}


{ .mfi
           ldfe         atan2_P10  = [EXP_AD_P1],16
           fnma.s1      atan2_B1Y  = atan2_u1_Y, atan2_Y, atan2_two
           nop.i 999
}
{ .mfi
           ldfe         atan2_P20  = [EXP_AD_P2],16
           fma.s1       atan2_wp_Y   = atan2_u1_Y, atan2_u1_Y, f0
           nop.i 999
;;
}

{ .mfi
           ldfe         atan2_P9   = [EXP_AD_P1],16
           fma.s1       atan2_z1_X = atan2_u1_X, atan2_Y, f0
           nop.i 999
}
{ .mfi
           ldfe         atan2_P19  = [EXP_AD_P2],16
           fnma.s1      atan2_B1X  = atan2_u1_X, atan2_X, atan2_two
           nop.i 999
}
;;

{ .mfi
           ldfe         atan2_P8   = [EXP_AD_P1],16
           fma.s1       atan2_z2_X = atan2_u1_X, atan2_ysq, f0
           nop.i 999
}
{ .mfb
           ldfe         atan2_P18  = [EXP_AD_P2],16
(p10)      fma.d.s0 f8 = atan2_Y,atan2_X,f0   // If y=nan, result quietized y
(p10)      br.ret.spnt b0        // Exit if y=nan
}
;;

// p6 true if swap,    means |y| >  1.0    or ysq > 1.0
// p7 true if no swap, means 1.0 >= |y|    or 1.0 >= ysq
{ .mfi
           ldfe         atan2_P7   = [EXP_AD_P1],16
           fcmp.ge.s1 p7,p6    = f1, atan2_ysq
           nop.i 999
}
{ .mmf
           ldfe         atan2_P17  = [EXP_AD_P2],16
           nop.m 999
           nop.f 999
}
;;

{ .mfi
           ldfe         atan2_P6   = [EXP_AD_P1],16
           fma.s1       atan2_E         = atan2_u1_Y, atan2_B1Y, atan2_Y
           nop.i 999
}
{ .mfi
           ldfe         atan2_P16  = [EXP_AD_P2],16
           fma.s1       atan2_B1sq_Y = atan2_B1Y, atan2_B1Y, f0
           nop.i 999
;;
}

{ .mfi
           ldfe         atan2_P5   = [EXP_AD_P1],16
(p7)       fma.s1       atan2_wp_X   = atan2_z1_X, atan2_z1_X, f0
           nop.i 999
}
{ .mfi
           ldfe         atan2_P15       = [EXP_AD_P2],16
(p7)       fma.s1       atan2_B1sq_X = atan2_B1X, atan2_B1X, f0
           nop.i 999
;;
}

{ .mfi
           ldfe         atan2_P4   = [EXP_AD_P1],16
(p6)       fma.s1       atan2_z         = atan2_u1_Y, atan2_B1Y, f0
           nop.i 999
}
{ .mfi
           ldfe         atan2_P14  = [EXP_AD_P2],16
(p7)       fma.s1       atan2_E         = atan2_z2_X, atan2_B1X, atan2_X
           nop.i 999
;;
}


{ .mfi
           ldfe         atan2_P3        = [EXP_AD_P1],16
           fcmp.eq.s0  p14,p15=atan2_X,atan2_Y  // Dummy for denorm and invalid
           nop.i 999
}
{ .mmf
           ldfe         atan2_P13  = [EXP_AD_P2],16
           nop.m 999
(p7)       fma.s1       atan2_z         = atan2_z1_X, atan2_B1X, f0
;;
}

{ .mfi
           ldfe         atan2_P2        = [EXP_AD_P1],16
(p6)       fma.s1       atan2_w         = atan2_wp_Y, atan2_B1sq_Y,f0
           nop.i 999
}
{ .mlx
           ldfe         atan2_P12  = [EXP_AD_P2],16
           movl         rsig_near_one = 0x8000000000000001 // signif near 1.0
;;
}

{ .mfi
           ldfe         atan2_P1        = [EXP_AD_P1],16
           fclass.m p9,p0 = atan2_Y, 0x23  // test if y inf
           nop.i 999
}
{ .mfi
           ldfe         atan2_P22       = [EXP_AD_P2],16
(p7)       fma.s1       atan2_w         = atan2_wp_X, atan2_B1sq_X,f0
           nop.i 999
;;
}

{ .mfi
           ldfe         atan2_P0        = [EXP_AD_P1],16
           frcpa.s1     atan2_F,p0     = f1, atan2_E
           nop.i 999
}
{ .mfi
           ldfe         atan2_pi_by_2   = [EXP_AD_P2],16
(p6)       fnma.s1       atan2_gV        = atan2_Y, atan2_z, atan2_X
           nop.i 999
;;
}

{ .mfi
           setf.sig      atan2_sig_near_one = rsig_near_one
(p7)       fnma.s1       atan2_gV        = atan2_X, atan2_z, atan2_Y
           nop.i 999
}
{ .mfb
           nop.m 999
(p9)       fma.d.s0  f8 = atan2_sgnY, atan2_pi_by_2, f0  // +-pi/2 if y inf
(p9)       br.ret.spnt b0      // exit if y inf, result is +-pi/2
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V13       = atan2_w, atan2_P11, atan2_P10
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W11       = atan2_w, atan2_P21, atan2_P20
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V11       = atan2_w, atan2_P9, atan2_P8
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_V12       = atan2_w, atan2_w, f0
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V8        = atan2_w, atan2_P7 , atan2_P6
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W8        = atan2_w, atan2_P19, atan2_P18
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fnma.s1      atan2_alpha     = atan2_E, atan2_F, f1
           nop.i 999
}
{ .mfi
           nop.m 999
           fnma.s1      atan2_alpha_1   = atan2_E, atan2_F, atan2_two
           nop.i 999
;;
}


{ .mfi
           nop.m 999
           fma.s1       atan2_V7        = atan2_w, atan2_P5 , atan2_P4
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W7        = atan2_w, atan2_P17, atan2_P16
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V4        = atan2_w, atan2_P3 , atan2_P2
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W4        = atan2_w, atan2_P15, atan2_P14
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V3        = atan2_w, atan2_P1 , atan2_P0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W3        = atan2_w, atan2_P13, atan2_P12
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V10       = atan2_V12, atan2_V13, atan2_V11
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_gVF       = atan2_gV, atan2_F, f0
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_alpha_sq  = atan2_alpha, atan2_alpha, f0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_Cp        = atan2_alpha, atan2_alpha_1, f1
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V9        = atan2_V12, atan2_V12, f0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W10       = atan2_V12, atan2_P22 , atan2_W11
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V6        = atan2_V12, atan2_V8 , atan2_V7
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W6        = atan2_V12, atan2_W8 , atan2_W7
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V2        = atan2_V12, atan2_V4 , atan2_V3
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W2        = atan2_V12, atan2_W4  , atan2_W3
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_alpha_cub = atan2_alpha, atan2_alpha_sq, f0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_C         = atan2_gVF, atan2_Cp, f0
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_W12       = atan2_V9, atan2_V9, f0
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_V5        = atan2_V9, atan2_V10, atan2_V6
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W5        = atan2_V9, atan2_W10, atan2_W6
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fclass.m p8,p0 = atan2_Y, 0x07  // Test for y=0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_d         = atan2_alpha_cub, atan2_C, atan2_C
           nop.i 999
}
;;

{ .mfi
           nop.m 999
           fma.s1       atan2_W12       = atan2_V9, atan2_W12, f0
           nop.i 999
}
;;

{ .mfi
           nop.m 999
           fma.s1       atan2_V1        = atan2_V9, atan2_V5, atan2_V2
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_W1        = atan2_V9, atan2_W5, atan2_W2
           nop.i 999
;;
}

{ .mfi
           nop.m 999
(p8)       fmerge.s     f8              = atan2_sgnY, f0  // +-0 if y=0
           nop.i 999
}
{ .mfb
           nop.m 999
           fma.s1       atan2_zcub      = atan2_z, atan2_w, f0
(p8)       br.ret.spnt b0      // Exit if y=0
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_pd        = atan2_P0, atan2_d, f0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_dsq       = atan2_d, atan2_d, f0
           nop.i 999
;;
}


{ .mfi
           nop.m 999
           fmerge.se    atan2_near_one = f1, atan2_sig_near_one // Const ~1.0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_Pp        = atan2_W12, atan2_W1, atan2_V1
           nop.i 999
;;
}

{ .mfi
           nop.m 999
           fma.s1       atan2_sgn_pi_by_2 = atan2_pi_by_2, atan2_sgnY, f0
           nop.i 999
}
{ .mfi
           nop.m 999
           fma.s1       atan2_A_lo      = atan2_pd, atan2_dsq, atan2_d
           nop.i 999
;;
}


{ .mfi
           nop.m 999
           fma.s1       atan2_A_hi      = atan2_zcub, atan2_Pp, atan2_z
           nop.i 999
;;
}


{ .mfi
           nop.m 999
(p6)       fma.s1       atan2_A         = atan2_A_hi, f1, atan2_A_lo
           nop.i 999
}
// For |Y| <= |X| and X > 0, result is A_hi + A_lo
{ .mfi
           nop.m 999
(p7)       fma.d.s0       f8         = atan2_A_hi, f1, atan2_A_lo
           nop.i 999
;;
}

// For |Y| > |X|, result is  +- pi/2 - (A_hi + A_lo)
// We perturb A by multiplying by 1.0+1ulp as we produce the result
// in order to get symmetrically rounded results in directed rounding modes.
// If we don't do this, there are a few cases where the trailing 11 bits of
// the significand of the result, before converting to double, are zero.  These
// cases do not round symmetrically in round to +infinity or round to -infinity.
{ .mfb
           nop.m 999
(p6)       fnma.d.s0      f8        = atan2_A, atan2_near_one, atan2_sgn_pi_by_2
           br.ret.sptk  b0
;;
}

GLOBAL_LIBM_END(atan)
libm_alias_double_other (atan, atan)
