.file "tancot.s"


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
// 02/02/00 Initial version
// 04/04/00 Unwind support added
// 12/27/00 Improved speed
// 02/21/01 Updated to call tanl
// 05/30/02 Added cot
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// double tan(double x);
// double cot(double x);
//
// Overview of operation
//==============================================================
// If the input value in radians is |x| >= 1.xxxxx 2^10 call the
// older slower version.
//
// The new algorithm is used when |x| <= 1.xxxxx 2^9.
//
// Represent the input X as Nfloat * pi/2 + r
//    where r can be negative and |r| <= pi/4
//
//     tan_W  = x * 2/pi
//     Nfloat = round_int(tan_W)
//
//     tan_r  = x - Nfloat * (pi/2)_hi
//  a) tan_r  = tan_r - Nfloat * (pi/2)_lo (for tan)
//  b) tan_r  = Nfloat * (pi/2)_lo - tan_r (for cot)
//
// We have two paths: p8, when Nfloat is even and p9. when Nfloat is odd.
//  a) for tan:  p8: tan(X) =  tan(r)
//               p9: tan(X) = -cot(r)
//  b) for cot:  p9: cot(X) =  cot(r)
//               p8: cot(X) = -tan(r)
//
// Each is evaluated as a series. The p9 path requires 1/r.
//
// The coefficients used in the series are stored in a table as
// are the pi constants.
//
// Registers used
//==============================================================
//
// predicate registers used:
// p6-12
//
// floating-point registers used:
// f10-15, f32-106
// f8, input
//
// general registers used
// r14-26, r32-39
//
// Assembly macros
//==============================================================
TAN_INV_PI_BY_2_2TO64        = f10
TAN_RSHF_2TO64               = f11
TAN_2TOM64                   = f12
TAN_RSHF                     = f13
TAN_W_2TO64_RSH              = f14
TAN_NFLOAT                   = f15

tan_Inv_Pi_by_2              = f32
tan_Pi_by_2_hi               = f33
tan_Pi_by_2_lo               = f34


tan_P0                       = f35
tan_P1                       = f36
tan_P2                       = f37
tan_P3                       = f38
tan_P4                       = f39
tan_P5                       = f40
tan_P6                       = f41
tan_P7                       = f42
tan_P8                       = f43
tan_P9                       = f44
tan_P10                      = f45
tan_P11                      = f46
tan_P12                      = f47
tan_P13                      = f48
tan_P14                      = f49
tan_P15                      = f50

tan_Q0                       = f51
tan_Q1                       = f52
tan_Q2                       = f53
tan_Q3                       = f54
tan_Q4                       = f55
tan_Q5                       = f56
tan_Q6                       = f57
tan_Q7                       = f58
tan_Q8                       = f59
tan_Q9                       = f60
tan_Q10                      = f61

tan_r                        = f62
tan_rsq                      = f63
tan_rcube                    = f64

tan_v18                      = f65
tan_v16                      = f66
tan_v17                      = f67
tan_v12                      = f68
tan_v13                      = f69
tan_v7                       = f70
tan_v8                       = f71
tan_v4                       = f72
tan_v5                       = f73
tan_v15                      = f74
tan_v11                      = f75
tan_v14                      = f76
tan_v3                       = f77
tan_v6                       = f78
tan_v10                      = f79
tan_v2                       = f80
tan_v9                       = f81
tan_v1                       = f82
tan_int_Nfloat               = f83
tan_Nfloat                   = f84

tan_NORM_f8                  = f85
tan_W                        = f86

tan_y0                       = f87
tan_d                        = f88
tan_y1                       = f89
tan_dsq                      = f90
tan_y2                       = f91
tan_d4                       = f92
tan_inv_r                    = f93

tan_z1                       = f94
tan_z2                       = f95
tan_z3                       = f96
tan_z4                       = f97
tan_z5                       = f98
tan_z6                       = f99
tan_z7                       = f100
tan_z8                       = f101
tan_z9                       = f102
tan_z10                      = f103
tan_z11                      = f104
tan_z12                      = f105

arg_copy                     = f106

/////////////////////////////////////////////////////////////

tan_GR_sig_inv_pi_by_2       = r14
tan_GR_rshf_2to64            = r15
tan_GR_exp_2tom64            = r16
tan_GR_n                     = r17
tan_GR_rshf                  = r18
tan_AD                       = r19
tan_GR_10009                 = r20
tan_GR_17_ones               = r21
tan_GR_N_odd_even            = r22
tan_GR_N                     = r23
tan_signexp                  = r24
tan_exp                      = r25
tan_ADQ                      = r26

GR_SAVE_B0                   = r33
GR_SAVE_PFS                  = r34
GR_SAVE_GP                   = r35
GR_Parameter_X               = r36
GR_Parameter_Y               = r37
GR_Parameter_RESULT          = r38
GR_Parameter_Tag             = r39


RODATA

.align 16

LOCAL_OBJECT_START(double_tan_constants)
   data8 0xC90FDAA22168C234, 0x00003FFF // pi/2 hi
   data8 0xBEEA54580DDEA0E1 // P14
   data8 0x3ED3021ACE749A59 // P15
   data8 0xBEF312BD91DC8DA1 // P12
   data8 0x3EFAE9AFC14C5119 // P13
   data8 0x3F2F342BF411E769 // P8
   data8 0x3F1A60FC9F3B0227 // P9
   data8 0x3EFF246E78E5E45B // P10
   data8 0x3F01D9D2E782875C // P11
   data8 0x3F8226E34C4499B6 // P4
   data8 0x3F6D6D3F12C236AC // P5
   data8 0x3F57DA1146DCFD8B // P6
   data8 0x3F43576410FE3D75 // P7
   data8 0x3FD5555555555555 // P0
   data8 0x3FC11111111111C2 // P1
   data8 0x3FABA1BA1BA0E850 // P2
   data8 0x3F9664F4886725A7 // P3
LOCAL_OBJECT_END(double_tan_constants)

LOCAL_OBJECT_START(double_Q_tan_constants)
   data8 0xC4C6628B80DC1CD1, 0x00003FBF // pi/2 lo
   data8 0x3E223A73BA576E48 // Q8
   data8 0x3DF54AD8D1F2CA43 // Q9
   data8 0x3EF66A8EE529A6AA // Q4
   data8 0x3EC2281050410EE6 // Q5
   data8 0x3E8D6BB992CC3CF5 // Q6
   data8 0x3E57F88DE34832E4 // Q7
   data8 0x3FD5555555555555 // Q0
   data8 0x3F96C16C16C16DB8 // Q1
   data8 0x3F61566ABBFFB489 // Q2
   data8 0x3F2BBD77945C1733 // Q3
   data8 0x3D927FB33E2B0E04 // Q10
LOCAL_OBJECT_END(double_Q_tan_constants)


.section .text

////////////////////////////////////////////////////////

LOCAL_LIBM_ENTRY(cot)
// The initial fnorm will take any unmasked faults and
// normalize any single/double unorms

{ .mlx
      cmp.eq    p12, p11 = r0, r0 // set p12=1, p11=0 for cot
      movl tan_GR_sig_inv_pi_by_2 = 0xA2F9836E4E44152A // significand of 2/pi
}
{ .mlx
      addl           tan_AD   = @ltoff(double_tan_constants), gp
      movl tan_GR_rshf_2to64 = 0x47e8000000000000 // 1.1000 2^(63+63+1)
}
;;

{ .mlx
      mov tan_GR_exp_2tom64 = 0xffff-64 // exponent of scaling factor 2^-64
      movl tan_GR_rshf = 0x43e8000000000000 // 1.1000 2^63 for right shift
}
{ .mfb
      ld8 tan_AD = [tan_AD]
      fnorm.s0  tan_NORM_f8  = f8
      br.cond.sptk COMMON_PATH
}
;;

LOCAL_LIBM_END(cot)


GLOBAL_IEEE754_ENTRY(tan)
// The initial fnorm will take any unmasked faults and
// normalize any single/double unorms

{ .mlx
      cmp.eq    p11, p12 = r0, r0 // set p11=1, p12=0 for tan
      movl tan_GR_sig_inv_pi_by_2 = 0xA2F9836E4E44152A // significand of 2/pi
}
{ .mlx
      addl           tan_AD   = @ltoff(double_tan_constants), gp
      movl tan_GR_rshf_2to64 = 0x47e8000000000000 // 1.1000 2^(63+63+1)
}
;;

{ .mlx
      mov tan_GR_exp_2tom64 = 0xffff-64 // exponent of scaling factor 2^-64
      movl tan_GR_rshf = 0x43e8000000000000 // 1.1000 2^63 for right shift
}
{ .mfi
      ld8 tan_AD = [tan_AD]
      fnorm.s0  tan_NORM_f8  = f8
      nop.i     0
}
;;


// Common path for both tan and cot
COMMON_PATH:
// Form two constants we need
//   2/pi * 2^1 * 2^63, scaled by 2^64 since we just loaded the significand
//   1.1000...000 * 2^(63+63+1) to right shift int(W) into the significand
{ .mmi
      setf.sig TAN_INV_PI_BY_2_2TO64 = tan_GR_sig_inv_pi_by_2
      setf.d TAN_RSHF_2TO64 = tan_GR_rshf_2to64
      mov       tan_GR_17_ones     = 0x1ffff             ;;
}


// Form another constant
//   2^-64 for scaling Nfloat
//   1.1000...000 * 2^63, the right shift constant
{ .mmf
      setf.exp TAN_2TOM64 = tan_GR_exp_2tom64
      adds tan_ADQ = double_Q_tan_constants - double_tan_constants, tan_AD
(p11) fclass.m.unc  p6,p0 = f8, 0x07  // Test for x=0 (tan)
}
;;


// Form another constant
//   2^-64 for scaling Nfloat
//   1.1000...000 * 2^63, the right shift constant
{ .mmf
      setf.d TAN_RSHF = tan_GR_rshf
      ldfe      tan_Pi_by_2_hi = [tan_AD],16
      fclass.m.unc  p7,p0 = f8, 0x23  // Test for x=inf
}
;;

{ .mfb
      ldfe      tan_Pi_by_2_lo = [tan_ADQ],16
      fclass.m.unc  p8,p0 = f8, 0xc3  // Test for x=nan
(p6)  br.ret.spnt    b0    ;;         // Exit for x=0 (tan only)
}

{ .mfi
      ldfpd     tan_P14,tan_P15 = [tan_AD],16
(p7)  frcpa.s0  f8,p9=f0,f0           // Set qnan indef if x=inf
      mov       tan_GR_10009 = 0x10009
}
{ .mib
      ldfpd      tan_Q8,tan_Q9  = [tan_ADQ],16
      nop.i 999
(p7)  br.ret.spnt    b0    ;;         // Exit for x=inf
}

{ .mfi
      ldfpd      tan_P12,tan_P13 = [tan_AD],16
(p12) fclass.m.unc  p6,p0 = f8, 0x07  // Test for x=0 (cot)
      nop.i 999
}
{ .mfb
      ldfpd      tan_Q4,tan_Q5  = [tan_ADQ],16
(p8)  fma.d.s0   f8=f8,f1,f8          // Set qnan if x=nan
(p8)  br.ret.spnt    b0    ;;         // Exit for x=nan
}

{ .mmf
      getf.exp  tan_signexp    = tan_NORM_f8
      ldfpd     tan_P8,tan_P9  = [tan_AD],16
      fmerge.s  arg_copy       = f8, f8 ;; // Save input for error call
}

// Multiply x by scaled 2/pi and add large const to shift integer part of W to
//   rightmost bits of significand
{ .mmf
      alloc      r32=ar.pfs,0,4,4,0
      ldfpd      tan_Q6,tan_Q7  = [tan_ADQ],16
      fma.s1 TAN_W_2TO64_RSH = tan_NORM_f8,TAN_INV_PI_BY_2_2TO64,TAN_RSHF_2TO64
};;

{ .mmf
      ldfpd      tan_P10,tan_P11 = [tan_AD],16
      and       tan_exp = tan_GR_17_ones, tan_signexp
(p6)  frcpa.s0  f8, p0 = f1, f8 ;;        // cot(+-0) = +-Inf
}


// p7 is true if we must call DBX TAN
// p7 is true if f8 exp is > 0x10009 (which includes all ones
//    NAN or inf)
{ .mmb
      ldfpd      tan_Q0,tan_Q1  = [tan_ADQ],16
      cmp.ge.unc  p7,p0 = tan_exp,tan_GR_10009
(p7)  br.cond.spnt   TAN_DBX ;;
}


{ .mmb
      ldfpd      tan_P4,tan_P5  = [tan_AD],16
(p6)  mov GR_Parameter_Tag = 226          // (cot)
(p6)  br.cond.spnt __libm_error_region ;; // call error support if cot(+-0)
}


{ .mmi
      ldfpd      tan_Q2,tan_Q3  = [tan_ADQ],16
      nop.m 999
      nop.i 999 ;;
}



// TAN_NFLOAT = Round_Int_Nearest(tan_W)
{ .mfi
      ldfpd      tan_P6,tan_P7  = [tan_AD],16
      fms.s1 TAN_NFLOAT = TAN_W_2TO64_RSH,TAN_2TOM64,TAN_RSHF
      nop.i 999 ;;
}


{ .mfi
      ldfd      tan_Q10 = [tan_ADQ]
      nop.f 999
      nop.i 999 ;;
}


{ .mfi
      ldfpd      tan_P0,tan_P1  = [tan_AD],16
      nop.f 999
      nop.i 999 ;;
}


{ .mmi
      getf.sig    tan_GR_n = TAN_W_2TO64_RSH
      ldfpd      tan_P2,tan_P3  = [tan_AD]
      nop.i 999 ;;
}

// tan_r          = -tan_Nfloat * tan_Pi_by_2_hi + x
{ .mfi
(p12) add        tan_GR_n = 0x1, tan_GR_n // N = N + 1 (for cot)
      fnma.s1  tan_r      = TAN_NFLOAT, tan_Pi_by_2_hi,  tan_NORM_f8
      nop.i 999 ;;
}


// p8 ==> even
// p9 ==> odd
{ .mmi
      and         tan_GR_N_odd_even = 0x1, tan_GR_n ;;
      nop.m 999
      cmp.eq.unc  p8,p9          = tan_GR_N_odd_even, r0      ;;
}


.pred.rel "mutex", p11, p12
// tan_r          = tan_r -tan_Nfloat * tan_Pi_by_2_lo (tan)
{ .mfi
      nop.m 999
(p11) fnma.s1  tan_r      = TAN_NFLOAT, tan_Pi_by_2_lo,  tan_r
      nop.i 999
}
// tan_r          = -(tan_r -tan_Nfloat * tan_Pi_by_2_lo) (cot)
{ .mfi
      nop.m 999
(p12) fms.s1   tan_r      = TAN_NFLOAT, tan_Pi_by_2_lo,  tan_r
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
      fma.s1   tan_rsq    = tan_r, tan_r,   f0
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
(p9)  frcpa.s1   tan_y0, p0 = f1,tan_r
      nop.i 999  ;;
}


{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v18 = tan_rsq, tan_P15, tan_P14
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v4  = tan_rsq, tan_P1, tan_P0
      nop.i 999  ;;
}



{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v16 = tan_rsq, tan_P13, tan_P12
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v17 = tan_rsq, tan_rsq, f0
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v12 = tan_rsq, tan_P9, tan_P8
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v13 = tan_rsq, tan_P11, tan_P10
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v7  = tan_rsq, tan_P5, tan_P4
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v8  = tan_rsq, tan_P7, tan_P6
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p9)  fnma.s1    tan_d   = tan_r, tan_y0, f1
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v5  = tan_rsq, tan_P3, tan_P2
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z11 = tan_rsq, tan_Q9, tan_Q8
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z12 = tan_rsq, tan_rsq, f0
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v15 = tan_v17, tan_v18, tan_v16
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z7 = tan_rsq, tan_Q5, tan_Q4
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v11 = tan_v17, tan_v13, tan_v12
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z8 = tan_rsq, tan_Q7, tan_Q6
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v14 = tan_v17, tan_v17, f0
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z3 = tan_rsq, tan_Q1, tan_Q0
      nop.i 999 ;;
}




{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v3 = tan_v17, tan_v5, tan_v4
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v6 = tan_v17, tan_v8, tan_v7
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p9)  fma.s1     tan_y1  = tan_y0, tan_d, tan_y0
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1     tan_dsq = tan_d, tan_d, f0
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z10 = tan_z12, tan_Q10, tan_z11
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z9  = tan_z12, tan_z12,f0
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z4 = tan_rsq, tan_Q3, tan_Q2
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z6  = tan_z12, tan_z8, tan_z7
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v10 = tan_v14, tan_v15, tan_v11
      nop.i 999 ;;
}



{ .mfi
      nop.m 999
(p9)  fma.s1     tan_y2  = tan_y1, tan_d, tan_y0
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1     tan_d4  = tan_dsq, tan_dsq, tan_d
      nop.i 999  ;;
}


{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v2 = tan_v14, tan_v6, tan_v3
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v9 = tan_v14, tan_v14, f0
      nop.i 999 ;;
}


{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z2  = tan_z12, tan_z4, tan_z3
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z5  = tan_z9, tan_z10, tan_z6
      nop.i 999  ;;
}


{ .mfi
      nop.m 999
(p9)  fma.s1     tan_inv_r = tan_d4, tan_y2, tan_y0
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1   tan_rcube  = tan_rsq, tan_r,   f0
      nop.i 999  ;;
}



{ .mfi
      nop.m 999
(p8)  fma.s1  tan_v1 = tan_v9, tan_v10, tan_v2
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1  tan_z1  = tan_z9, tan_z5, tan_z2
      nop.i 999   ;;
}



{ .mfi
      nop.m 999
(p8)  fma.d.s0  f8  = tan_v1, tan_rcube, tan_r
      nop.i 999
}
{ .mfb
      nop.m 999
(p9)  fms.d.s0  f8  = tan_r, tan_z1, tan_inv_r
      br.ret.sptk    b0 ;;
}
GLOBAL_IEEE754_END(tan)
libm_alias_double_other (__tan, tan)


LOCAL_LIBM_ENTRY(__libm_callout)
TAN_DBX:
.prologue

{ .mfi
      nop.m 0
      fmerge.s f9 = f0,f0
.save ar.pfs,GR_SAVE_PFS
      mov  GR_SAVE_PFS=ar.pfs
}
;;

{ .mfi
      mov GR_SAVE_GP=gp
      nop.f 0
.save b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0
}

.body
{ .mmb
      nop.m 999
      nop.m 999
(p11) br.cond.sptk.many  call_tanl ;;
}

// Here if we should call cotl
{ .mmb
      nop.m 999
      nop.m 999
      br.call.sptk.many  b0=__libm_cotl# ;;
}

{ .mfi
      mov gp        = GR_SAVE_GP
      fnorm.d.s0 f8 = f8
      mov b0        = GR_SAVE_B0
}
;;

{ .mib
      nop.m 999
      mov ar.pfs    = GR_SAVE_PFS
      br.ret.sptk     b0
;;
}

// Here if we should call tanl
call_tanl:
{ .mmb
      nop.m 999
      nop.m 999
      br.call.sptk.many  b0=__libm_tanl# ;;
}

{ .mfi
      mov gp        = GR_SAVE_GP
      fnorm.d.s0 f8 = f8
      mov b0        = GR_SAVE_B0
}
;;

{ .mib
      nop.m 999
      mov ar.pfs    = GR_SAVE_PFS
      br.ret.sptk     b0
;;
}

LOCAL_LIBM_END(__libm_callout)

.type __libm_tanl#,@function
.global __libm_tanl#
.type __libm_cotl#,@function
.global __libm_cotl#

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

// (1)
{ .mfi
      add           GR_Parameter_Y=-32,sp        // Parameter 2 value
      nop.f 0
.save   ar.pfs,GR_SAVE_PFS
      mov           GR_SAVE_PFS=ar.pfs           // Save ar.pfs
}
{ .mfi
.fframe 64
      add sp=-64,sp                              // Create new stack
      nop.f 0
      mov GR_SAVE_GP=gp                          // Save gp
};;

// (2)
{ .mmi
      stfd [GR_Parameter_Y] = f1,16              // STORE Parameter 2 on stack
      add GR_Parameter_X = 16,sp                 // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                          // Save b0
};;

.body
// (3)
{ .mib
      stfd [GR_Parameter_X] = arg_copy           // STORE Parameter 1 on stack
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
      nop.b 0
}
{ .mib
      stfd [GR_Parameter_Y] = f8                 // STORE Parameter 3 on stack
      add   GR_Parameter_Y = -16,GR_Parameter_Y
      br.call.sptk b0=__libm_error_support#      // Call error handling function
};;
{ .mmi
      nop.m 0
      nop.m 0
      add   GR_Parameter_RESULT = 48,sp
};;

// (4)
{ .mmi
      ldfd  f8 = [GR_Parameter_RESULT]           // Get return result off stack
.restore sp
      add   sp = 64,sp                           // Restore stack pointer
      mov   b0 = GR_SAVE_B0                      // Restore return address
};;
{ .mib
      mov   gp = GR_SAVE_GP                      // Restore gp
      mov   ar.pfs = GR_SAVE_PFS                 // Restore ar.pfs
      br.ret.sptk     b0                         // Return
};;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
