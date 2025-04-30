.file "sinh.s"


// Copyright (c) 2000 - 2005, Intel Corporation
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
// 10/12/00 Update to set denormal operand and underflow flags
// 01/22/01 Fixed to set inexact flag for small args.
// 05/02/01 Reworked to improve speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 11/20/02 Improved speed with new algorithm
// 03/31/05 Reformatted delimiters between data tables

// API
//==============================================================
// double sinh(double)

// Overview of operation
//==============================================================
// Case 1:  0 < |x| < 2^-60
//  Result = x, computed by x+sgn(x)*x^2) to handle flags and rounding
//
// Case 2:  2^-60 < |x| < 0.25
//  Evaluate sinh(x) by a 13th order polynomial
//  Care is take for the order of multiplication; and A1 is not exactly 1/3!,
//  A2 is not exactly 1/5!, etc.
//  sinh(x) = x + (A1*x^3 + A2*x^5 + A3*x^7 + A4*x^9 + A5*x^11 + A6*x^13)
//
// Case 3:  0.25 < |x| < 710.47586
//  Algorithm is based on the identity sinh(x) = ( exp(x) - exp(-x) ) / 2.
//  The algorithm for exp is described as below.  There are a number of
//  economies from evaluating both exp(x) and exp(-x).  Although we
//  are evaluating both quantities, only where the quantities diverge do we
//  duplicate the computations.  The basic algorithm for exp(x) is described
//  below.
//
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
//       Calculate exp(r) by 5th order polynomial
//          r = x - n (log2/128)_high
//          delta = - n (log2/128)_low
//       Calculate exp(delta) as 1 + delta


// Special values
//==============================================================
// sinh(+0)    = +0
// sinh(-0)    = -0

// sinh(+qnan) = +qnan
// sinh(-qnan) = -qnan
// sinh(+snan) = +qnan
// sinh(-snan) = -qnan

// sinh(-inf)  = -inf
// sinh(+inf)  = +inf

// Overflow and Underflow
//=======================
// sinh(x) = largest double normal when
//     |x| = 710.47586 = 0x408633ce8fb9f87d
//
// Underflow is handled as described in case 1 above

// Registers used
//==============================================================
// Floating Point registers used:
// f8, input, output
// f6 -> f15,  f32 -> f61

// General registers used:
// r14 -> r40

// Predicate registers used:
// p6 -> p15

// Assembly macros
//==============================================================

rRshf                 = r14
rN_neg                = r14
rAD_TB1               = r15
rAD_TB2               = r16
rAD_P                 = r17
rN                    = r18
rIndex_1              = r19
rIndex_2_16           = r20
rM                    = r21
rBiased_M             = r21
rSig_inv_ln2          = r22
rIndex_1_neg          = r22
rExp_bias             = r23
rExp_bias_minus_1     = r23
rExp_mask             = r24
rTmp                  = r24
rGt_ln                = r24
rIndex_2_16_neg       = r24
rM_neg                = r25
rBiased_M_neg         = r25
rRshf_2to56           = r26
rAD_T1_neg            = r26
rExp_2tom56           = r28
rAD_T2_neg            = r28
rAD_T1                = r29
rAD_T2                = r30
rSignexp_x            = r31
rExp_x                = r31

GR_SAVE_B0            = r33
GR_SAVE_PFS           = r34
GR_SAVE_GP            = r35

GR_Parameter_X        = r37
GR_Parameter_Y        = r38
GR_Parameter_RESULT   = r39
GR_Parameter_TAG      = r40


FR_X                  = f10
FR_Y                  = f1
FR_RESULT             = f8

fRSHF_2TO56           = f6
fINV_LN2_2TO63        = f7
fW_2TO56_RSH          = f9
f2TOM56               = f11
fP5                   = f12
fP4                   = f13
fP3                   = f14
fP2                   = f15

fLn2_by_128_hi        = f33
fLn2_by_128_lo        = f34

fRSHF                 = f35
fNfloat               = f36
fNormX                = f37
fR                    = f38
fF                    = f39

fRsq                  = f40
f2M                   = f41
fS1                   = f42
fT1                   = f42
fS2                   = f43
fT2                   = f43
fS                    = f43
fWre_urm_f8           = f44
fAbsX                 = f44

fMIN_DBL_OFLOW_ARG    = f45
fMAX_DBL_NORM_ARG     = f46
fXsq                  = f47
fX4                   = f48
fGt_pln               = f49
fTmp                  = f49

fP54                  = f50
fP5432                = f50
fP32                  = f51
fP                    = f52
fP54_neg              = f53
fP5432_neg            = f53
fP32_neg              = f54
fP_neg                = f55
fF_neg                = f56

f2M_neg               = f57
fS1_neg               = f58
fT1_neg               = f58
fS2_neg               = f59
fT2_neg               = f59
fS_neg                = f59
fExp                  = f60
fExp_neg              = f61

fA6                   = f50
fA65                  = f50
fA6543                = f50
fA654321              = f50
fA5                   = f51
fA4                   = f52
fA43                  = f52
fA3                   = f53
fA2                   = f54
fA21                  = f54
fA1                   = f55
fX3                   = f56

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
//   1. fRSHF_2TO56 = 1.1000..00 * 2^(63-7)
//        This constant is added to x*1/ln2 to shift the integer part of
//        x*128/ln2 into the rightmost bits of the significand.
//        The result of this fma is fW_2TO56_RSH.
//   2. fRSHF       = 1.1000..00 * 2^(63)
//        This constant is subtracted from fW_2TO56_RSH * 2^(-56) to give
//        the integer part of w, n, as a floating-point number.
//        The result of this fms is fNfloat.


LOCAL_OBJECT_START(exp_table_1)
data8 0x408633ce8fb9f87e // smallest dbl overflow arg
data8 0x408633ce8fb9f87d // largest dbl arg to give normal dbl result
data8 0xb17217f7d1cf79ab , 0x00003ff7 // ln2/128 hi
data8 0xc9e3b39803f2f6af , 0x00003fb7 // ln2/128 lo
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


LOCAL_OBJECT_START(exp_p_table)
data8 0x3f8111116da21757 //P5
data8 0x3fa55555d787761c //P4
data8 0x3fc5555555555414 //P3
data8 0x3fdffffffffffd6a //P2
LOCAL_OBJECT_END(exp_p_table)

LOCAL_OBJECT_START(sinh_p_table)
data8 0xB08AF9AE78C1239F, 0x00003FDE  // A6
data8 0xB8EF1D28926D8891, 0x00003FEC  // A4
data8 0x8888888888888412, 0x00003FF8  // A2
data8 0xD732377688025BE9, 0x00003FE5  // A5
data8 0xD00D00D00D4D39F2, 0x00003FF2  // A3
data8 0xAAAAAAAAAAAAAAAB, 0x00003FFC  // A1
LOCAL_OBJECT_END(sinh_p_table)


.section .text
GLOBAL_IEEE754_ENTRY(sinh)

{ .mlx
      getf.exp        rSignexp_x = f8  // Must recompute if x unorm
      movl            rSig_inv_ln2 = 0xb8aa3b295c17f0bc  // significand of 1/ln2
}
{ .mlx
      addl            rAD_TB1    = @ltoff(exp_table_1), gp
      movl            rRshf_2to56 = 0x4768000000000000   // 1.10000 2^(63+56)
}
;;

{ .mfi
      ld8             rAD_TB1    = [rAD_TB1]
      fclass.m        p6,p0 = f8,0x0b  // Test for x=unorm
      mov             rExp_mask = 0x1ffff
}
{ .mfi
      mov             rExp_bias = 0xffff
      fnorm.s1        fNormX   = f8
      mov             rExp_2tom56 = 0xffff-56
}
;;

// Form two constants we need
//  1/ln2 * 2^63  to compute  w = x * 1/ln2 * 128
//  1.1000..000 * 2^(63+63-7) to right shift int(w) into the significand

{ .mfi
      setf.sig        fINV_LN2_2TO63 = rSig_inv_ln2 // form 1/ln2 * 2^63
      fclass.m        p8,p0 = f8,0x07  // Test for x=0
      nop.i 999
}
{ .mlx
      setf.d          fRSHF_2TO56 = rRshf_2to56 // Form const 1.100 * 2^(63+56)
      movl            rRshf = 0x43e8000000000000 // 1.10000 2^63 for right shift
}
;;

{ .mfi
      ldfpd           fMIN_DBL_OFLOW_ARG, fMAX_DBL_NORM_ARG = [rAD_TB1],16
      fclass.m        p10,p0 = f8,0x1e3  // Test for x=inf, nan, NaT
      nop.i           0
}
{ .mfb
      setf.exp        f2TOM56 = rExp_2tom56 // form 2^-56 for scaling Nfloat
      nop.f           0
(p6)  br.cond.spnt    SINH_UNORM            // Branch if x=unorm
}
;;

SINH_COMMON:
{ .mfi
      ldfe            fLn2_by_128_hi  = [rAD_TB1],16
      nop.f           0
      nop.i           0
}
{ .mfb
      setf.d          fRSHF = rRshf // Form right shift const 1.100 * 2^63
      nop.f           0
(p8)  br.ret.spnt     b0                    // Exit for x=0, result=x
}
;;

{ .mfi
      ldfe            fLn2_by_128_lo  = [rAD_TB1],16
      nop.f           0
      nop.i           0
}
{ .mfb
      and             rExp_x = rExp_mask, rSignexp_x // Biased exponent of x
(p10) fma.d.s0        f8 = f8,f1,f0  // Result if x=inf, nan, NaT
(p10) br.ret.spnt     b0               // quick exit for x=inf, nan, NaT
}
;;

// After that last load rAD_TB1 points to the beginning of table 1
{ .mfi
      nop.m           0
      fcmp.eq.s0      p6,p0 = f8, f0       // Dummy to set D
      sub             rExp_x = rExp_x, rExp_bias // True exponent of x
}
;;

{ .mfi
      nop.m           0
      fmerge.s        fAbsX = f0, fNormX   // Form |x|
      nop.i           0
}
{ .mfb
      cmp.gt          p7, p0 = -2, rExp_x      // Test |x| < 2^(-2)
      fma.s1          fXsq = fNormX, fNormX, f0  // x*x for small path
(p7)  br.cond.spnt    SINH_SMALL               // Branch if 0 < |x| < 2^-2
}
;;

// W = X * Inv_log2_by_128
// By adding 1.10...0*2^63 we shift and get round_int(W) in significand.
// We actually add 1.10...0*2^56 to X * Inv_log2 to do the same thing.

{ .mfi
      add             rAD_P = 0x180, rAD_TB1
      fma.s1          fW_2TO56_RSH  = fNormX, fINV_LN2_2TO63, fRSHF_2TO56
      add             rAD_TB2 = 0x100, rAD_TB1
}
;;

// Divide arguments into the following categories:
//  Certain Safe                - 0.25 <= |x| <= MAX_DBL_NORM_ARG
//  Possible Overflow       p14 - MAX_DBL_NORM_ARG < |x| < MIN_DBL_OFLOW_ARG
//  Certain Overflow        p15 - MIN_DBL_OFLOW_ARG <= |x| < +inf
//
// If the input is really a double arg, then there will never be
// "Possible Overflow" arguments.
//

{ .mfi
      ldfpd           fP5, fP4  = [rAD_P] ,16
      fcmp.ge.s1      p15,p14 = fAbsX,fMIN_DBL_OFLOW_ARG
      nop.i           0
}
;;

// Nfloat = round_int(W)
// The signficand of fW_2TO56_RSH contains the rounded integer part of W,
// as a twos complement number in the lower bits (that is, it may be negative).
// That twos complement number (called N) is put into rN.

// Since fW_2TO56_RSH is scaled by 2^56, it must be multiplied by 2^-56
// before the shift constant 1.10000 * 2^63 is subtracted to yield fNfloat.
// Thus, fNfloat contains the floating point version of N

{ .mfi
      ldfpd           fP3, fP2  = [rAD_P]
(p14) fcmp.gt.unc.s1  p14,p0 = fAbsX,fMAX_DBL_NORM_ARG
      nop.i           0
}
{ .mfb
      nop.m           0
      fms.s1          fNfloat = fW_2TO56_RSH, f2TOM56, fRSHF
(p15) br.cond.spnt    SINH_CERTAIN_OVERFLOW
}
;;

{ .mfi
      getf.sig        rN        = fW_2TO56_RSH
      nop.f           0
      mov             rExp_bias_minus_1 = 0xfffe
}
;;

// rIndex_1 has index_1
// rIndex_2_16 has index_2 * 16
// rBiased_M has M

// rM has true M
// r = x - Nfloat * ln2_by_128_hi
// f = 1 - Nfloat * ln2_by_128_lo
{ .mfi
      and             rIndex_1 = 0x0f, rN
      fnma.s1         fR   = fNfloat, fLn2_by_128_hi, fNormX
      shr             rM = rN,  0x7
}
{ .mfi
      and             rIndex_2_16 = 0x70, rN
      fnma.s1         fF   = fNfloat, fLn2_by_128_lo, f1
      sub             rN_neg = r0, rN
}
;;

{ .mmi
      and             rIndex_1_neg = 0x0f, rN_neg
      add             rBiased_M = rExp_bias_minus_1, rM
      shr             rM_neg = rN_neg,  0x7
}
{ .mmi
      and             rIndex_2_16_neg = 0x70, rN_neg
      add             rAD_T2 = rAD_TB2, rIndex_2_16
      shladd          rAD_T1 = rIndex_1, 4, rAD_TB1
}
;;

// rAD_T1 has address of T1
// rAD_T2 has address if T2

{ .mmi
      setf.exp        f2M = rBiased_M
      ldfe            fT2  = [rAD_T2]
      nop.i           0
}
{ .mmi
      add             rBiased_M_neg = rExp_bias_minus_1, rM_neg
      add             rAD_T2_neg = rAD_TB2, rIndex_2_16_neg
      shladd          rAD_T1_neg = rIndex_1_neg, 4, rAD_TB1
}
;;

// Create Scale = 2^M
// Load T1 and T2
{ .mmi
      ldfe            fT1  = [rAD_T1]
      nop.m           0
      nop.i           0
}
{ .mmf
      setf.exp        f2M_neg = rBiased_M_neg
      ldfe            fT2_neg  = [rAD_T2_neg]
      fma.s1          fF_neg   = fNfloat, fLn2_by_128_lo, f1
}
;;

{ .mfi
      nop.m           0
      fma.s1          fRsq = fR, fR, f0
      nop.i           0
}
{ .mfi
      ldfe            fT1_neg  = [rAD_T1_neg]
      fma.s1          fP54 = fR, fP5, fP4
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP32 = fR, fP3, fP2
      nop.i           0
}
{ .mfi
      nop.m           0
      fnma.s1         fP54_neg = fR, fP5, fP4
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fnma.s1         fP32_neg = fR, fP3, fP2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP5432  = fRsq, fP54, fP32
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fS2  = fF,fT2,f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fS1  = f2M,fT1,f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fP5432_neg  = fRsq, fP54_neg, fP32_neg
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fS1_neg  = f2M_neg,fT1_neg,f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fS2_neg  = fF_neg,fT2_neg,f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP     = fRsq, fP5432, fR
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fS   = fS1,fS2,f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fms.s1          fP_neg     = fRsq, fP5432_neg, fR
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fS_neg   = fS1_neg,fS2_neg,f0
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fmpy.s0         fTmp = fLn2_by_128_lo, fLn2_by_128_lo // Force inexact
(p14) br.cond.spnt    SINH_POSSIBLE_OVERFLOW
}
;;

{ .mfi
      nop.m           0
      fma.s1          fExp = fS, fP, fS
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fExp_neg = fS_neg, fP_neg, fS_neg
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fms.d.s0        f8 = fExp, f1, fExp_neg
      br.ret.sptk     b0                  // Normal path exit
}
;;

// Here if 0 < |x| < 0.25
SINH_SMALL:
{ .mfi
      add             rAD_T1 = 0x1a0, rAD_TB1
      fcmp.lt.s1      p7, p8 = fNormX, f0       // Test sign of x
      cmp.gt          p6, p0 = -60, rExp_x      // Test |x| < 2^(-60)
}
{ .mfi
      add             rAD_T2 = 0x1d0, rAD_TB1
      nop.f           0
      nop.i           0
}
;;

{ .mmb
      ldfe            fA6 = [rAD_T1],16
      ldfe            fA5 = [rAD_T2],16
(p6)  br.cond.spnt    SINH_VERY_SMALL           // Branch if |x| < 2^(-60)
}
;;

{ .mmi
      ldfe            fA4 = [rAD_T1],16
      ldfe            fA3 = [rAD_T2],16
      nop.i           0
}
;;

{ .mmi
      ldfe            fA2 = [rAD_T1]
      ldfe            fA1 = [rAD_T2]
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fX3 = fNormX, fXsq, f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fX4 = fXsq, fXsq, f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fA65 = fXsq, fA6, fA5
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fA43 = fXsq, fA4, fA3
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fA21 = fXsq, fA2, fA1
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fA6543 = fX4, fA65, fA43
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fA654321 = fX4, fA6543, fA21
      nop.i           0
}
;;

// Dummy multiply to generate inexact
{ .mfi
      nop.m           0
      fmpy.s0         fTmp = fA6, fA6
      nop.i           0
}
{ .mfb
      nop.m           0
      fma.d.s0        f8 = fA654321, fX3, fNormX
      br.ret.sptk     b0                // Exit if 2^-60 < |x| < 0.25
}
;;

SINH_VERY_SMALL:
// Here if 0 < |x| < 2^-60
// Compute result by x + sgn(x)*x^2 to get properly rounded result
.pred.rel "mutex",p7,p8
{ .mfi
      nop.m           0
(p7)  fnma.d.s0       f8 = fNormX, fNormX, fNormX // If x<0 result ~ x-x^2
      nop.i           0
}
{ .mfb
      nop.m           0
(p8)  fma.d.s0        f8 = fNormX, fNormX, fNormX // If x>0 result ~ x+x^2
      br.ret.sptk     b0                          // Exit if |x| < 2^-60
}
;;


SINH_POSSIBLE_OVERFLOW:

// Here if fMAX_DBL_NORM_ARG < |x| < fMIN_DBL_OFLOW_ARG
// This cannot happen if input is a double, only if input higher precision.
// Overflow is a possibility, not a certainty.

// Recompute result using status field 2 with user's rounding mode,
// and wre set.  If result is larger than largest double, then we have
// overflow

{ .mfi
      mov             rGt_ln  = 0x103ff // Exponent for largest dbl + 1 ulp
      fsetc.s2        0x7F,0x42         // Get user's round mode, set wre
      nop.i           0
}
;;

{ .mfi
      setf.exp        fGt_pln = rGt_ln  // Create largest double + 1 ulp
      fma.d.s2        fWre_urm_f8 = fS, fP, fS    // Result with wre set
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fsetc.s2        0x7F,0x40                   // Turn off wre in sf2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fcmp.ge.s1      p6, p0 =  fWre_urm_f8, fGt_pln // Test for overflow
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      nop.f           0
(p6)  br.cond.spnt    SINH_CERTAIN_OVERFLOW // Branch if overflow
}
;;

{ .mfb
      nop.m           0
      fma.d.s0        f8 = fS, fP, fS
      br.ret.sptk     b0                     // Exit if really no overflow
}
;;

SINH_CERTAIN_OVERFLOW:
{ .mfi
      sub             rTmp = rExp_mask, r0, 1
      fcmp.lt.s1      p6, p7 = fNormX, f0    // Test for x < 0
      nop.i           0
}
;;

{ .mmf
      alloc           r32=ar.pfs,1,4,4,0
      setf.exp        fTmp = rTmp
      fmerge.s        FR_X = f8,f8
}
;;

{ .mfi
      mov             GR_Parameter_TAG = 127
(p6)  fnma.d.s0       FR_RESULT = fTmp, fTmp, f0    // Set I,O and -INF result
      nop.i           0
}
{ .mfb
      nop.m           0
(p7)  fma.d.s0        FR_RESULT = fTmp, fTmp, f0    // Set I,O and +INF result
      br.cond.sptk    __libm_error_region
}
;;

// Here if x unorm
SINH_UNORM:
{ .mfb
      getf.exp        rSignexp_x = fNormX    // Must recompute if x unorm
      fcmp.eq.s0      p6, p0 = f8, f0        // Set D flag
      br.cond.sptk    SINH_COMMON
}
;;

GLOBAL_IEEE754_END(sinh)
libm_alias_double_other (__sinh, sinh)


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
        stfd [GR_Parameter_X] = FR_X            // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT       // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
};;
{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
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
