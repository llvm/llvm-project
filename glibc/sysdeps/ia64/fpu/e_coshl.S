.file "coshl.s"


// Copyright (c) 2000 - 2002, Intel Corporation
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
// 02/02/00 Initial version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 01/23/01 Set inexact flag for large args.
// 05/07/01 Reworked to improve speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 12/06/02 Improved performance
//
// API
//==============================================================
// long double = coshl(long double)
// input  floating point f8
// output floating point f8
//
// Registers used
//==============================================================
// general registers:
// r14 -> r40
// predicate registers used:
// p6 -> p11
// floating-point registers used:
// f9 -> f15; f32 -> f90;
// f8 has input, then output
//
// Overview of operation
//==============================================================
// There are seven paths
// 1. 0 < |x| < 0.25          COSH_BY_POLY
// 2. 0.25 <=|x| < 32         COSH_BY_TBL
// 3. 32 <= |x| < 11357.21655 COSH_BY_EXP (merged path with COSH_BY_TBL)
// 4. |x| >= 11357.21655      COSH_HUGE
// 5. x=0                     Done with early exit
// 6. x=inf,nan               Done with early exit
// 7. x=denormal              COSH_DENORM
//
// For double extended we get overflow for x >= 400c b174 ddc0 31ae c0ea
//                                           >= 11357.21655
//
//
// 1. COSH_BY_POLY   0 < |x| < 0.25
// ===============
// Evaluate cosh(x) by a 12th order polynomial
// Care is take for the order of multiplication; and P2 is not exactly 1/4!,
// P3 is not exactly 1/6!, etc.
// cosh(x) = 1 + (P1*x^2 + P2*x^4 + P3*x^6 + P4*x^8 + P5*x^10 + P6*x^12)
//
// 2. COSH_BY_TBL   0.25 <= |x| < 32.0
// =============
// cosh(x) = cosh(B+R)
//         = cosh(B)cosh(R) + sinh(B)sinh(R)
//
// ax = |x| = M*log2/64 + R
// B = M*log2/64
// M = 64*N + j
//   We will calculate M and get N as (M-j)/64
//   The division is a shift.
// exp(B)  = exp(N*log2 + j*log2/64)
//         = 2^N * 2^(j*log2/64)
// cosh(B) = 1/2(e^B + e^-B)
//         = 1/2(2^N * 2^(j*log2/64) + 2^-N * 2^(-j*log2/64))
// cosh(B) = (2^(N-1) * 2^(j*log2/64) + 2^(-N-1) * 2^(-j*log2/64))
// sinh(B) = (2^(N-1) * 2^(j*log2/64) - 2^(-N-1) * 2^(-j*log2/64))
// 2^(j*log2/64) is stored as Tjhi + Tjlo , j= -32,....,32
// Tjhi is double-extended (80-bit) and Tjlo is single(32-bit)
//
// R = ax - M*log2/64
// R = ax - M*log2_by_64_hi - M*log2_by_64_lo
// exp(R) = 1 + R +R^2(1/2! + R(1/3! + R(1/4! + ... + R(1/n!)...)
//        = 1 + p_odd + p_even
//        where the p_even uses the A coefficients and the p_even uses
//        the B coefficients
//
// So sinh(R) = 1 + p_odd + p_even -(1 -p_odd -p_even)/2 = p_odd
//    cosh(R) = 1 + p_even
//    cosh(B) = C_hi + C_lo
//    sinh(B) = S_hi
// cosh(x) = cosh(B)cosh(R) + sinh(B)sinh(R)
//
// 3. COSH_BY_EXP   32.0 <= |x| < 11357.21655  ( 400c b174 ddc0 31ae c0ea )
// ==============
// Can approximate result by exp(x)/2 in this region.
// Y_hi = Tjhi
// Y_lo = Tjhi * (p_odd + p_even) + Tjlo
// cosh(x) = Y_hi + Y_lo
//
// 4. COSH_HUGE     |x| >= 11357.21655  ( 400c b174 ddc0 31ae c0ea )
// ============
// Set error tag and call error support
//
//
// Assembly macros
//==============================================================
r_ad5                 = r14
r_rshf_2to57          = r15
r_exp_denorm          = r15
r_ad_mJ_lo            = r15
r_ad_J_lo             = r16
r_2Nm1                = r17
r_2mNm1               = r18
r_exp_x               = r18
r_ad_J_hi             = r19
r_ad2o                = r19
r_ad_mJ_hi            = r20
r_mj                  = r21
r_ad2e                = r22
r_ad3                 = r23
r_ad1                 = r24
r_Mmj                 = r24
r_rshf                = r25
r_M                   = r25
r_N                   = r25
r_jshf                = r26
r_exp_2tom57          = r26
r_j                   = r26
r_exp_mask            = r27
r_signexp_x           = r28
r_signexp_0_5         = r28
r_exp_0_25            = r29
r_sig_inv_ln2         = r30
r_exp_32              = r30
r_exp_huge            = r30
r_ad4                 = r31

GR_SAVE_PFS           = r34
GR_SAVE_B0            = r35
GR_SAVE_GP            = r36

GR_Parameter_X        = r37
GR_Parameter_Y        = r38
GR_Parameter_RESULT   = r39
GR_Parameter_TAG      = r40


f_ABS_X               = f9
f_X2                  = f10
f_X4                  = f11
f_tmp                 = f14
f_RSHF                = f15

f_Inv_log2by64        = f32
f_log2by64_lo         = f33
f_log2by64_hi         = f34
f_A1                  = f35

f_A2                  = f36
f_A3                  = f37
f_Rcub                = f38
f_M_temp              = f39
f_R_temp              = f40

f_Rsq                 = f41
f_R                   = f42
f_M                   = f43
f_B1                  = f44
f_B2                  = f45

f_B3                  = f46
f_peven_temp1         = f47
f_peven_temp2         = f48
f_peven               = f49
f_podd_temp1          = f50

f_podd_temp2          = f51
f_podd                = f52
f_poly65              = f53
f_poly6543            = f53
f_poly6to1            = f53
f_poly43              = f54
f_poly21              = f55

f_X3                  = f56
f_INV_LN2_2TO63       = f57
f_RSHF_2TO57          = f58
f_2TOM57              = f59
f_smlst_oflow_input   = f60

f_pre_result          = f61
f_huge                = f62
f_spos                = f63
f_sneg                = f64
f_Tjhi                = f65

f_Tjlo                = f66
f_Tmjhi               = f67
f_Tmjlo               = f68
f_S_hi                = f69
f_SC_hi_temp          = f70

f_C_lo_temp1          = f71
f_C_lo_temp2          = f72
f_C_lo_temp3          = f73
f_C_lo_temp4          = f73
f_C_lo                = f74
f_C_hi                = f75

f_Y_hi                = f77
f_Y_lo_temp           = f78
f_Y_lo                = f79
f_NORM_X              = f80

f_P1                  = f81
f_P2                  = f82
f_P3                  = f83
f_P4                  = f84
f_P5                  = f85

f_P6                  = f86
f_Tjhi_spos           = f87
f_Tjlo_spos           = f88
f_huge                = f89
f_signed_hi_lo        = f90


// Data tables
//==============================================================

// DO NOT CHANGE ORDER OF THESE TABLES
RODATA

.align 16
LOCAL_OBJECT_START(cosh_arg_reduction)
//   data8 0xB8AA3B295C17F0BC, 0x00004005  // 64/log2 -- signif loaded with setf
   data8 0xB17217F7D1000000, 0x00003FF8  // log2/64 high part
   data8 0xCF79ABC9E3B39804, 0x00003FD0  // log2/64 low part
   data8 0xb174ddc031aec0ea, 0x0000400c  // Smallest x to overflow (11357.21655)
LOCAL_OBJECT_END(cosh_arg_reduction)

LOCAL_OBJECT_START(cosh_p_table)
   data8 0x8FA02AC65BCBD5BC, 0x00003FE2  // P6
   data8 0xD00D00D1021D7370, 0x00003FEF  // P4
   data8 0xAAAAAAAAAAAAAB80, 0x00003FFA  // P2
   data8 0x93F27740C0C2F1CC, 0x00003FE9  // P5
   data8 0xB60B60B60B4FE884, 0x00003FF5  // P3
   data8 0x8000000000000000, 0x00003FFE  // P1
LOCAL_OBJECT_END(cosh_p_table)

LOCAL_OBJECT_START(cosh_ab_table)
   data8 0xAAAAAAAAAAAAAAAC, 0x00003FFC  // A1
   data8 0x88888888884ECDD5, 0x00003FF8  // A2
   data8 0xD00D0C6DCC26A86B, 0x00003FF2  // A3
   data8 0x8000000000000002, 0x00003FFE  // B1
   data8 0xAAAAAAAAAA402C77, 0x00003FFA  // B2
   data8 0xB60B6CC96BDB144D, 0x00003FF5  // B3
LOCAL_OBJECT_END(cosh_ab_table)

LOCAL_OBJECT_START(cosh_j_hi_table)
   data8 0xB504F333F9DE6484, 0x00003FFE
   data8 0xB6FD91E328D17791, 0x00003FFE
   data8 0xB8FBAF4762FB9EE9, 0x00003FFE
   data8 0xBAFF5AB2133E45FB, 0x00003FFE
   data8 0xBD08A39F580C36BF, 0x00003FFE
   data8 0xBF1799B67A731083, 0x00003FFE
   data8 0xC12C4CCA66709456, 0x00003FFE
   data8 0xC346CCDA24976407, 0x00003FFE
   data8 0xC5672A115506DADD, 0x00003FFE
   data8 0xC78D74C8ABB9B15D, 0x00003FFE
   data8 0xC9B9BD866E2F27A3, 0x00003FFE
   data8 0xCBEC14FEF2727C5D, 0x00003FFE
   data8 0xCE248C151F8480E4, 0x00003FFE
   data8 0xD06333DAEF2B2595, 0x00003FFE
   data8 0xD2A81D91F12AE45A, 0x00003FFE
   data8 0xD4F35AABCFEDFA1F, 0x00003FFE
   data8 0xD744FCCAD69D6AF4, 0x00003FFE
   data8 0xD99D15C278AFD7B6, 0x00003FFE
   data8 0xDBFBB797DAF23755, 0x00003FFE
   data8 0xDE60F4825E0E9124, 0x00003FFE
   data8 0xE0CCDEEC2A94E111, 0x00003FFE
   data8 0xE33F8972BE8A5A51, 0x00003FFE
   data8 0xE5B906E77C8348A8, 0x00003FFE
   data8 0xE8396A503C4BDC68, 0x00003FFE
   data8 0xEAC0C6E7DD24392F, 0x00003FFE
   data8 0xED4F301ED9942B84, 0x00003FFE
   data8 0xEFE4B99BDCDAF5CB, 0x00003FFE
   data8 0xF281773C59FFB13A, 0x00003FFE
   data8 0xF5257D152486CC2C, 0x00003FFE
   data8 0xF7D0DF730AD13BB9, 0x00003FFE
   data8 0xFA83B2DB722A033A, 0x00003FFE
   data8 0xFD3E0C0CF486C175, 0x00003FFE
   data8 0x8000000000000000, 0x00003FFF // Center of table
   data8 0x8164D1F3BC030773, 0x00003FFF
   data8 0x82CD8698AC2BA1D7, 0x00003FFF
   data8 0x843A28C3ACDE4046, 0x00003FFF
   data8 0x85AAC367CC487B15, 0x00003FFF
   data8 0x871F61969E8D1010, 0x00003FFF
   data8 0x88980E8092DA8527, 0x00003FFF
   data8 0x8A14D575496EFD9A, 0x00003FFF
   data8 0x8B95C1E3EA8BD6E7, 0x00003FFF
   data8 0x8D1ADF5B7E5BA9E6, 0x00003FFF
   data8 0x8EA4398B45CD53C0, 0x00003FFF
   data8 0x9031DC431466B1DC, 0x00003FFF
   data8 0x91C3D373AB11C336, 0x00003FFF
   data8 0x935A2B2F13E6E92C, 0x00003FFF
   data8 0x94F4EFA8FEF70961, 0x00003FFF
   data8 0x96942D3720185A00, 0x00003FFF
   data8 0x9837F0518DB8A96F, 0x00003FFF
   data8 0x99E0459320B7FA65, 0x00003FFF
   data8 0x9B8D39B9D54E5539, 0x00003FFF
   data8 0x9D3ED9A72CFFB751, 0x00003FFF
   data8 0x9EF5326091A111AE, 0x00003FFF
   data8 0xA0B0510FB9714FC2, 0x00003FFF
   data8 0xA27043030C496819, 0x00003FFF
   data8 0xA43515AE09E6809E, 0x00003FFF
   data8 0xA5FED6A9B15138EA, 0x00003FFF
   data8 0xA7CD93B4E965356A, 0x00003FFF
   data8 0xA9A15AB4EA7C0EF8, 0x00003FFF
   data8 0xAB7A39B5A93ED337, 0x00003FFF
   data8 0xAD583EEA42A14AC6, 0x00003FFF
   data8 0xAF3B78AD690A4375, 0x00003FFF
   data8 0xB123F581D2AC2590, 0x00003FFF
   data8 0xB311C412A9112489, 0x00003FFF
   data8 0xB504F333F9DE6484, 0x00003FFF
LOCAL_OBJECT_END(cosh_j_hi_table)

LOCAL_OBJECT_START(cosh_j_lo_table)
   data4 0x1EB2FB13
   data4 0x1CE2CBE2
   data4 0x1DDC3CBC
   data4 0x1EE9AA34
   data4 0x9EAEFDC1
   data4 0x9DBF517B
   data4 0x1EF88AFB
   data4 0x1E03B216
   data4 0x1E78AB43
   data4 0x9E7B1747
   data4 0x9EFE3C0E
   data4 0x9D36F837
   data4 0x9DEE53E4
   data4 0x9E24AE8E
   data4 0x1D912473
   data4 0x1EB243BE
   data4 0x1E669A2F
   data4 0x9BBC610A
   data4 0x1E761035
   data4 0x9E0BE175
   data4 0x1CCB12A1
   data4 0x1D1BFE90
   data4 0x1DF2F47A
   data4 0x1EF22F22
   data4 0x9E3F4A29
   data4 0x1EC01A5B
   data4 0x1E8CAC3A
   data4 0x9DBB3FAB
   data4 0x1EF73A19
   data4 0x9BB795B5
   data4 0x1EF84B76
   data4 0x9EF5818B
   data4 0x00000000 // Center of table
   data4 0x1F77CACA
   data4 0x1EF8A91D
   data4 0x1E57C976
   data4 0x9EE8DA92
   data4 0x1EE85C9F
   data4 0x1F3BF1AF
   data4 0x1D80CA1E
   data4 0x9D0373AF
   data4 0x9F167097
   data4 0x1EB70051
   data4 0x1F6EB029
   data4 0x1DFD6D8E
   data4 0x9EB319B0
   data4 0x1EBA2BEB
   data4 0x1F11D537
   data4 0x1F0D5A46
   data4 0x9E5E7BCA
   data4 0x9F3AAFD1
   data4 0x9E86DACC
   data4 0x9F3EDDC2
   data4 0x1E496E3D
   data4 0x9F490BF6
   data4 0x1DD1DB48
   data4 0x1E65EBFB
   data4 0x9F427496
   data4 0x1F283C4A
   data4 0x1F4B0047
   data4 0x1F130152
   data4 0x9E8367C0
   data4 0x9F705F90
   data4 0x1EFB3C53
   data4 0x1F32FB13
LOCAL_OBJECT_END(cosh_j_lo_table)


.section .text
GLOBAL_IEEE754_ENTRY(coshl)

{ .mlx
      getf.exp        r_signexp_x = f8   // Get signexp of x, must redo if unorm
      movl            r_sig_inv_ln2 = 0xb8aa3b295c17f0bc // significand of 1/ln2
}
{ .mlx
      addl            r_ad1 = @ltoff(cosh_arg_reduction), gp
      movl            r_rshf_2to57 = 0x4778000000000000 // 1.10000 2^(63+57)
}
;;

{ .mfi
      ld8             r_ad1 = [r_ad1]
      fmerge.s        f_ABS_X    = f0,f8
      mov             r_exp_0_25 = 0x0fffd    // Form exponent for 0.25
}
{ .mfi
      nop.m           0
      fnorm.s1        f_NORM_X = f8
      mov             r_exp_2tom57 = 0xffff-57
}
;;

{ .mfi
      setf.d          f_RSHF_2TO57 = r_rshf_2to57 // Form const 1.100 * 2^120
      fclass.m        p10,p0 = f8, 0x0b           // Test for denorm
      mov             r_exp_mask = 0x1ffff
}
{ .mlx
      setf.sig        f_INV_LN2_2TO63 = r_sig_inv_ln2 // Form 1/ln2 * 2^63
      movl            r_rshf = 0x43e8000000000000 // 1.1000 2^63 for right shift
}
;;

{ .mfi
      nop.m           0
      fclass.m        p7,p0 = f8, 0x07  // Test if x=0
      nop.i           0
}
{ .mfi
      setf.exp        f_2TOM57 = r_exp_2tom57 // Form 2^-57 for scaling
      nop.f           0
      add             r_ad3 = 0x90, r_ad1  // Point to ab_table
}
;;

{ .mfi
      setf.d          f_RSHF = r_rshf     // Form right shift const 1.100 * 2^63
      fclass.m        p6,p0 = f8, 0xe3     // Test if x nan, inf
      add             r_ad4 = 0x2f0, r_ad1 // Point to j_hi_table midpoint
}
{ .mib
      add             r_ad2e = 0x20, r_ad1 // Point to p_table
      nop.i           0
(p10) br.cond.spnt    COSH_DENORM          // Branch if x denorm
}
;;

// Common path -- return here from COSH_DENORM if x is unnorm
COSH_COMMON:
{ .mfi
      ldfe            f_smlst_oflow_input = [r_ad2e],16
(p7)  fma.s0          f8 = f1, f1, f0      // Result = 1.0 if x=0
      add             r_ad5 = 0x580, r_ad1 // Point to j_lo_table midpoint
}
{ .mib
      ldfe            f_log2by64_hi  = [r_ad1],16
      and             r_exp_x = r_exp_mask, r_signexp_x
(p7)  br.ret.spnt     b0                  // Exit if x=0
}
;;

// Get the A coefficients for COSH_BY_TBL
{ .mfi
      ldfe            f_A1 = [r_ad3],16
      fcmp.lt.s1      p8,p9 = f8,f0           // Test for x<0
      cmp.lt          p7,p0 = r_exp_x, r_exp_0_25  // Test x < 0.25
}
{ .mfb
      add             r_ad2o = 0x30, r_ad2e  // Point to p_table odd coeffs
(p6)  fma.s0          f8 = f8,f8,f0          // Result for x nan, inf
(p6)  br.ret.spnt     b0                     // Exit for x nan, inf
}
;;

// Calculate X2 = ax*ax for COSH_BY_POLY
{ .mfi
      ldfe            f_log2by64_lo  = [r_ad1],16
      nop.f           0
      nop.i           0
}
{ .mfb
      ldfe            f_A2 = [r_ad3],16
      fma.s1          f_X2 = f_NORM_X, f_NORM_X, f0
(p7)  br.cond.spnt    COSH_BY_POLY
}
;;

// Here if |x| >= 0.25
COSH_BY_TBL:
// ******************************************************
// STEP 1 (TBL and EXP) - Argument reduction
// ******************************************************
// Get the following constants.
// Inv_log2by64
// log2by64_hi
// log2by64_lo


// We want 2^(N-1) and 2^(-N-1). So bias N-1 and -N-1 and
// put them in an exponent.
// f_spos = 2^(N-1) and f_sneg = 2^(-N-1)
// 0xffff + (N-1)  = 0xffff +N -1
// 0xffff - (N +1) = 0xffff -N -1


// Calculate M and keep it as integer and floating point.
// M = round-to-integer(x*Inv_log2by64)
// f_M = M = truncate(ax/(log2/64))
// Put the integer representation of M in r_M
//    and the floating point representation of M in f_M

// Get the remaining A,B coefficients
{ .mmi
      ldfe            f_A3 = [r_ad3],16
      nop.m           0
      nop.i           0
}
;;

// Use constant (1.100*2^(63-6)) to get rounded M into rightmost significand
// |x| * 64 * 1/ln2 * 2^(63-6) + 1.1000 * 2^(63+(63-6))
{ .mfi
      nop.m           0
      fma.s1          f_M_temp = f_ABS_X, f_INV_LN2_2TO63, f_RSHF_2TO57
      mov             r_signexp_0_5 = 0x0fffe // signexp of +0.5
}
;;

// Test for |x| >= overflow limit
{ .mfi
      ldfe            f_B1 = [r_ad3],16
      fcmp.ge.s1      p6,p0 = f_ABS_X, f_smlst_oflow_input
      nop.i           0
}
;;

{ .mfi
      ldfe            f_B2 = [r_ad3],16
      nop.f           0
      mov             r_exp_32 = 0x10004
}
;;

// Subtract RSHF constant to get rounded M as a floating point value
// M_temp * 2^(63-6) - 2^63
{ .mfb
      ldfe            f_B3 = [r_ad3],16
      fms.s1          f_M = f_M_temp, f_2TOM57, f_RSHF
(p6)  br.cond.spnt    COSH_HUGE  // Branch if result will overflow
}
;;

{ .mfi
      getf.sig        r_M = f_M_temp
      nop.f           0
      cmp.ge          p7,p6 = r_exp_x, r_exp_32 // Test if x >= 32
}
;;

// Calculate j. j is the signed extension of the six lsb of M. It
// has a range of -32 thru 31.

// Calculate R
// ax - M*log2by64_hi
// R = (ax - M*log2by64_hi) - M*log2by64_lo

{ .mfi
      nop.m           0
      fnma.s1         f_R_temp = f_M, f_log2by64_hi, f_ABS_X
      and             r_j = 0x3f, r_M
}
;;

{ .mii
      nop.m           0
      shl             r_jshf = r_j, 0x2 // Shift j so can sign extend it
;;
      sxt1            r_jshf = r_jshf
}
;;

{ .mii
      nop.m           0
      shr             r_j = r_jshf, 0x2    // Now j has range -32 to 31
      nop.i           0
}
;;

{ .mmi
      shladd          r_ad_J_hi = r_j, 4, r_ad4 // pointer to Tjhi
      sub             r_Mmj = r_M, r_j          // M-j
      sub             r_mj = r0, r_j            // Form -j
}
;;

// The TBL and EXP branches are merged and predicated
// If TBL, p6 true, 0.25 <= |x| < 32
// If EXP, p7 true, 32 <= |x| < overflow_limit
//
// N = (M-j)/64
{ .mfi
      ldfe            f_Tjhi = [r_ad_J_hi]
      fnma.s1         f_R = f_M, f_log2by64_lo, f_R_temp
      shr             r_N = r_Mmj, 0x6            // N = (M-j)/64
}
{ .mfi
      shladd          r_ad_mJ_hi = r_mj, 4, r_ad4 // pointer to Tmjhi
      nop.f           0
      shladd          r_ad_mJ_lo = r_mj, 2, r_ad5 // pointer to Tmjlo
}
;;

{ .mfi
      sub             r_2mNm1 = r_signexp_0_5, r_N // signexp 2^(-N-1)
      nop.f           0
      shladd          r_ad_J_lo = r_j, 2, r_ad5   // pointer to Tjlo
}
{ .mfi
      ldfe            f_Tmjhi = [r_ad_mJ_hi]
      nop.f           0
      add             r_2Nm1 = r_signexp_0_5, r_N // signexp 2^(N-1)
}
;;

{ .mmf
      ldfs            f_Tmjlo = [r_ad_mJ_lo]
      setf.exp        f_sneg = r_2mNm1            // Form 2^(-N-1)
      nop.f           0
}
;;

{ .mmf
      ldfs            f_Tjlo  = [r_ad_J_lo]
      setf.exp        f_spos = r_2Nm1             // Form 2^(N-1)
      nop.f           0
}
;;

// ******************************************************
// STEP 2 (TBL and EXP)
// ******************************************************
// Calculate Rsquared and Rcubed in preparation for p_even and p_odd

{ .mmf
      nop.m           0
      nop.m           0
      fma.s1          f_Rsq  = f_R, f_R, f0
}
;;


// Calculate p_even
// B_2 + Rsq *B_3
// B_1 + Rsq * (B_2 + Rsq *B_3)
// p_even = Rsq * (B_1 + Rsq * (B_2 + Rsq *B_3))
{ .mfi
      nop.m           0
      fma.s1          f_peven_temp1 = f_Rsq, f_B3, f_B2
      nop.i           0
}
// Calculate p_odd
// A_2 + Rsq *A_3
// A_1 + Rsq * (A_2 + Rsq *A_3)
// podd = R + Rcub * (A_1 + Rsq * (A_2 + Rsq *A_3))
{ .mfi
      nop.m           0
      fma.s1          f_podd_temp1 = f_Rsq, f_A3, f_A2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_Rcub = f_Rsq, f_R, f0
      nop.i           0
}
;;

//
// If TBL,
// Calculate S_hi and S_lo, and C_hi
// SC_hi_temp = sneg * Tmjhi
// S_hi = spos * Tjhi - SC_hi_temp
// S_hi = spos * Tjhi - (sneg * Tmjhi)
// C_hi = spos * Tjhi + SC_hi_temp
// C_hi = spos * Tjhi + (sneg * Tmjhi)

{ .mfi
      nop.m           0
(p6)  fma.s1          f_SC_hi_temp = f_sneg, f_Tmjhi, f0
      nop.i           0
}
;;

// If TBL,
// C_lo_temp3 = sneg * Tmjlo
// C_lo_temp4 = spos * Tjlo + C_lo_temp3
// C_lo_temp4 = spos * Tjlo + (sneg * Tmjlo)
{ .mfi
      nop.m           0
(p6)  fma.s1          f_C_lo_temp3 =  f_sneg, f_Tmjlo, f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_peven_temp2 = f_Rsq, f_peven_temp1, f_B1
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          f_podd_temp2 = f_Rsq, f_podd_temp1, f_A1
      nop.i           0
}
;;

// If EXP,
// Compute 2^(N-1) * Tjhi and 2^(N-1) * Tjlo
{ .mfi
      nop.m           0
(p7)  fma.s1          f_Tjhi_spos = f_Tjhi, f_spos, f0
      nop.i           0
}
{ .mfi
      nop.m           0
(p7)  fma.s1          f_Tjlo_spos = f_Tjlo, f_spos, f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p6)  fma.s1          f_C_hi = f_spos, f_Tjhi, f_SC_hi_temp
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p6)  fms.s1          f_S_hi = f_spos, f_Tjhi, f_SC_hi_temp
      nop.i           0
}
{ .mfi
      nop.m           0
(p6)  fma.s1          f_C_lo_temp4 = f_spos, f_Tjlo, f_C_lo_temp3
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_peven = f_Rsq, f_peven_temp2, f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          f_podd = f_podd_temp2, f_Rcub, f_R
      nop.i           0
}
;;

// If TBL,
// C_lo_temp1 =  spos * Tjhi - C_hi
// C_lo_temp2 =  sneg * Tmjlo + C_lo_temp1
// C_lo_temp2 =  sneg * Tmjlo + (spos * Tjhi - C_hi)

{ .mfi
      nop.m           0
(p6)  fms.s1          f_C_lo_temp1 =  f_spos, f_Tjhi,  f_C_hi
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p6)  fma.s1          f_C_lo_temp2 = f_sneg, f_Tmjhi, f_C_lo_temp1
      nop.i           0
}
;;

// If EXP,
// Y_hi = 2^(N-1) * Tjhi
// Y_lo = 2^(N-1) * Tjhi * (p_odd + p_even) + 2^(N-1) * Tjlo
{ .mfi
      nop.m           0
(p7)  fma.s1          f_Y_lo_temp =  f_peven, f1, f_podd
      nop.i           0
}
;;

// If TBL,
// C_lo = C_lo_temp4 + C_lo_temp2
{ .mfi
      nop.m           0
(p6)  fma.s1          f_C_lo = f_C_lo_temp4, f1, f_C_lo_temp2
      nop.i           0
}
;;

// If TBL,
// Y_hi = C_hi
// Y_lo = S_hi*p_odd + (C_hi*p_even + C_lo)
{ .mfi
      nop.m           0
(p6)  fma.s1          f_Y_lo_temp = f_C_hi, f_peven, f_C_lo
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p7)  fma.s1          f_Y_lo = f_Tjhi_spos, f_Y_lo_temp, f_Tjlo_spos
      nop.i           0
}
;;

// Dummy multiply to generate inexact
{ .mfi
      nop.m           0
      fmpy.s0         f_tmp = f_B2, f_B2
      nop.i           0
}
{ .mfi
      nop.m           0
(p6)  fma.s1          f_Y_lo = f_S_hi, f_podd, f_Y_lo_temp
      nop.i           0
}
;;

// f8 = answer = Y_hi + Y_lo
{ .mfi
      nop.m           0
(p7)  fma.s0          f8 = f_Y_lo,  f1, f_Tjhi_spos
      nop.i           0
}
;;

// f8 = answer = Y_hi + Y_lo
{ .mfb
      nop.m           0
(p6)  fma.s0          f8 = f_Y_lo, f1, f_C_hi
      br.ret.sptk     b0      // Exit for COSH_BY_TBL and COSH_BY_EXP
}
;;


// Here if 0 < |x| < 0.25
COSH_BY_POLY:
{ .mmf
      ldfe            f_P6 = [r_ad2e],16
      ldfe            f_P5 = [r_ad2o],16
      nop.f           0
}
;;

{ .mmi
      ldfe            f_P4 = [r_ad2e],16
      ldfe            f_P3 = [r_ad2o],16
      nop.i           0
}
;;

{ .mmi
      ldfe            f_P2 = [r_ad2e],16
      ldfe            f_P1 = [r_ad2o],16
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_X3 = f_NORM_X, f_X2, f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          f_X4 = f_X2, f_X2, f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_poly65 = f_X2, f_P6, f_P5
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          f_poly43 = f_X2, f_P4, f_P3
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_poly21 = f_X2, f_P2, f_P1
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_poly6543 = f_X4, f_poly65, f_poly43
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          f_poly6to1 = f_X4, f_poly6543, f_poly21
      nop.i           0
}
;;

// Dummy multiply to generate inexact
{ .mfi
      nop.m           0
      fmpy.s0         f_tmp = f_P6, f_P6
      nop.i           0
}
{ .mfb
      nop.m           0
      fma.s0          f8 = f_poly6to1, f_X2, f1
      br.ret.sptk     b0                // Exit COSH_BY_POLY
}
;;


// Here if x denorm or unorm
COSH_DENORM:
// Determine if x really a denorm and not a unorm
{ .mmf
      getf.exp        r_signexp_x = f_NORM_X
      mov             r_exp_denorm = 0x0c001   // Real denorms have exp < this
      fmerge.s        f_ABS_X = f0, f_NORM_X
}
;;

{ .mfi
      nop.m           0
      fcmp.eq.s0      p10,p0 = f8, f0  // Set denorm flag
      nop.i           0
}
;;

// Set p8 if really a denorm
{ .mmi
      and             r_exp_x = r_exp_mask, r_signexp_x
;;
      cmp.lt          p8,p9 = r_exp_x, r_exp_denorm
      nop.i           0
}
;;

// Identify denormal operands.
{ .mfb
      nop.m           0
(p8)  fma.s0          f8 =  f8,f8,f1 // If x denorm, result=1+x^2
(p9)  br.cond.sptk    COSH_COMMON    // Return to main path if x unorm
}
;;

{ .mfb
      nop.m           0
      nop.f           0
      br.ret.sptk     b0            // Exit if x denorm
}
;;


// Here if |x| >= overflow limit
COSH_HUGE:
// for COSH_HUGE, put 24000 in exponent; take sign from input
{ .mmi
      mov             r_exp_huge = 0x15dbf
;;
      setf.exp        f_huge  = r_exp_huge
      nop.i           0
}
;;

{ .mfi
      alloc           r32 = ar.pfs,0,5,4,0
      fma.s1          f_signed_hi_lo = f_huge, f1, f1
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s0          f_pre_result = f_signed_hi_lo, f_huge, f0
      mov             GR_Parameter_TAG = 63
}
;;

GLOBAL_IEEE754_END(coshl)
libm_alias_ldouble_other (__cosh, cosh)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

{ .mfi
        add   GR_Parameter_Y=-32,sp              // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                  // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                            // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                        // Save gp
};;

{ .mmi
        stfe [GR_Parameter_Y] = f0,16            // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp               // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                        // Save b0
};;

.body
{ .mib
        stfe [GR_Parameter_X] = f8               // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y   // Parameter 3 address
        nop.b 0
}
{ .mib
        stfe [GR_Parameter_Y] = f_pre_result     // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#    // Call error handling function
};;

{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;

{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]         // Get return result off stack
.restore sp
        add   sp = 64,sp                         // Restore stack pointer
        mov   b0 = GR_SAVE_B0                    // Restore return address
};;

{ .mib
        mov   gp = GR_SAVE_GP                    // Restore gp
        mov   ar.pfs = GR_SAVE_PFS               // Restore ar.pfs
        br.ret.sptk     b0                       // Return
};;

LOCAL_LIBM_END(__libm_error_region)


.type   __libm_error_support#,@function
.global __libm_error_support#
