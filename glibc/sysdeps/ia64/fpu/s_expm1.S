.file "exp_m1.s"


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
// 02/02/00 Initial Version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 07/07/01 Improved speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 11/20/02 Improved speed, algorithm based on exp
// 03/31/05 Reformatted delimiters between data tables

// API
//==============================================================
// double expm1(double)

// Overview of operation
//==============================================================
// 1. Inputs of Nan, Inf, Zero, NatVal handled with special paths
//
// 2. |x| < 2^-60
//    Result = x, computed by x + x*x to handle appropriate flags and rounding
//
// 3. 2^-60 <= |x| < 2^-2
//    Result determined by 13th order Taylor series polynomial
//    expm1f(x) = x + Q2*x^2 + ... + Q13*x^13
//
// 4. x < -48.0
//    Here we know result is essentially -1 + eps, where eps only affects
//    rounded result.  Set I.
//
// 5. x >= 709.7827
//    Result overflows.  Set I, O, and call error support
//
// 6. 2^-2 <= x < 709.7827  or  -48.0 <= x < -2^-2
//    This is the main path.  The algorithm is described below:

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
//       Calculate exp(r) by series by 5th order polynomial
//          r = x - n (log2/128)_high
//          delta = - n (log2/128)_low
//       Calculate exp(delta) as 1 + delta


// Special values
//==============================================================
// expm1(+0)    = +0.0
// expm1(-0)    = -0.0

// expm1(+qnan) = +qnan
// expm1(-qnan) = -qnan
// expm1(+snan) = +qnan
// expm1(-snan) = -qnan

// expm1(-inf)  = -1.0
// expm1(+inf)  = +inf

// Overflow and Underflow
//=======================
// expm1(x) = largest double normal when
//     x = 709.7827 = 40862e42fefa39ef
//
// Underflow is handled as described in case 2 above.


// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f9 -> f15,  f32 -> f75

// General registers used:
// r14 -> r40

// Predicate registers used:
// p6 -> p15

// Assembly macros
//==============================================================

rRshf                  = r14
rAD_TB1                = r15
rAD_T1                 = r15
rAD_TB2                = r16
rAD_T2                 = r16
rAD_Ln2_lo             = r17
rAD_P                  = r17

rN                     = r18
rIndex_1               = r19
rIndex_2_16            = r20

rM                     = r21
rBiased_M              = r21
rIndex_1_16            = r22
rSignexp_x             = r23
rExp_x                 = r24
rSig_inv_ln2           = r25

rAD_Q1                 = r26
rAD_Q2                 = r27
rTmp                   = r27
rExp_bias              = r28
rExp_mask              = r29
rRshf_2to56            = r30

rGt_ln                 = r31
rExp_2tom56            = r31


GR_SAVE_B0             = r33
GR_SAVE_PFS            = r34
GR_SAVE_GP             = r35
GR_SAVE_SP             = r36

GR_Parameter_X         = r37
GR_Parameter_Y         = r38
GR_Parameter_RESULT    = r39
GR_Parameter_TAG       = r40


FR_X                   = f10
FR_Y                   = f1
FR_RESULT              = f8

fRSHF_2TO56            = f6
fINV_LN2_2TO63         = f7
fW_2TO56_RSH           = f9
f2TOM56                = f11
fP5                    = f12
fP54                   = f50
fP5432                 = f50
fP4                    = f13
fP3                    = f14
fP32                   = f14
fP2                    = f15

fLn2_by_128_hi         = f33
fLn2_by_128_lo         = f34

fRSHF                  = f35
fNfloat                = f36
fW                     = f37
fR                     = f38
fF                     = f39

fRsq                   = f40
fRcube                 = f41

f2M                    = f42
fS1                    = f43
fT1                    = f44

fMIN_DBL_OFLOW_ARG     = f45
fMAX_DBL_MINUS_1_ARG   = f46
fMAX_DBL_NORM_ARG      = f47
fP_lo                  = f51
fP_hi                  = f52
fP                     = f53
fS                     = f54

fNormX                 = f56

fWre_urm_f8            = f57

fGt_pln                = f58
fTmp                   = f58

fS2                    = f59
fT2                    = f60
fSm1                   = f61

fXsq                   = f62
fX6                    = f63
fX4                    = f63
fQ7                    = f64
fQ76                   = f64
fQ7654                 = f64
fQ765432               = f64
fQ6                    = f65
fQ5                    = f66
fQ54                   = f66
fQ4                    = f67
fQ3                    = f68
fQ32                   = f68
fQ2                    = f69
fQD                    = f70
fQDC                   = f70
fQDCBA                 = f70
fQDCBA98               = f70
fQDCBA98765432         = f70
fQC                    = f71
fQB                    = f72
fQBA                   = f72
fQA                    = f73
fQ9                    = f74
fQ98                   = f74
fQ8                    = f75

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


LOCAL_OBJECT_START(exp_Table_1)
data8 0x40862e42fefa39f0 // smallest dbl overflow arg
data8 0xc048000000000000 // approx largest arg for minus one result
data8 0x40862e42fefa39ef // largest dbl arg to give normal dbl result
data8 0x0                // pad
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
LOCAL_OBJECT_END(exp_Table_1)

// Table 2 is 2^(index_1/8) where
// index_2 goes from 0 to 7
LOCAL_OBJECT_START(exp_Table_2)
data8 0x8000000000000000 , 0x00003FFF
data8 0x8B95C1E3EA8BD6E7 , 0x00003FFF
data8 0x9837F0518DB8A96F , 0x00003FFF
data8 0xA5FED6A9B15138EA , 0x00003FFF
data8 0xB504F333F9DE6484 , 0x00003FFF
data8 0xC5672A115506DADD , 0x00003FFF
data8 0xD744FCCAD69D6AF4 , 0x00003FFF
data8 0xEAC0C6E7DD24392F , 0x00003FFF
LOCAL_OBJECT_END(exp_Table_2)


LOCAL_OBJECT_START(exp_p_table)
data8 0x3f8111116da21757 //P5
data8 0x3fa55555d787761c //P4
data8 0x3fc5555555555414 //P3
data8 0x3fdffffffffffd6a //P2
LOCAL_OBJECT_END(exp_p_table)

LOCAL_OBJECT_START(exp_Q1_table)
data8 0x3de6124613a86d09 // QD = 1/13!
data8 0x3e21eed8eff8d898 // QC = 1/12!
data8 0x3ec71de3a556c734 // Q9 = 1/9!
data8 0x3efa01a01a01a01a // Q8 = 1/8!
data8 0x8888888888888889,0x3ff8 // Q5 = 1/5!
data8 0xaaaaaaaaaaaaaaab,0x3ffc // Q3 = 1/3!
data8 0x0,0x0            // Pad to avoid bank conflicts
LOCAL_OBJECT_END(exp_Q1_table)

LOCAL_OBJECT_START(exp_Q2_table)
data8 0x3e5ae64567f544e4 // QB = 1/11!
data8 0x3e927e4fb7789f5c // QA = 1/10!
data8 0x3f2a01a01a01a01a // Q7 = 1/7!
data8 0x3f56c16c16c16c17 // Q6 = 1/6!
data8 0xaaaaaaaaaaaaaaab,0x3ffa // Q4 = 1/4!
data8 0x8000000000000000,0x3ffe // Q2 = 1/2!
LOCAL_OBJECT_END(exp_Q2_table)


.section .text
GLOBAL_IEEE754_ENTRY(expm1)

{ .mlx
      getf.exp        rSignexp_x = f8  // Must recompute if x unorm
      movl            rSig_inv_ln2 = 0xb8aa3b295c17f0bc  // signif of 1/ln2
}
{ .mlx
      addl            rAD_TB1    = @ltoff(exp_Table_1), gp
      movl            rRshf_2to56 = 0x4768000000000000   // 1.10000 2^(63+56)
}
;;

// We do this fnorm right at the beginning to normalize
// any input unnormals so that SWA is not taken.
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
      nop.i           0
}
{ .mlx
      setf.d          fRSHF_2TO56 = rRshf_2to56 // Form 1.100 * 2^(63+56)
      movl            rRshf = 0x43e8000000000000   // 1.10000 2^63 for rshift
}
;;

{ .mfi
      setf.exp        f2TOM56 = rExp_2tom56 // form 2^-56 for scaling Nfloat
      fclass.m        p9,p0 = f8,0x22  // Test for x=-inf
      add             rAD_TB2 = 0x140, rAD_TB1 // Point to Table 2
}
{ .mib
      add             rAD_Q1 = 0x1e0, rAD_TB1 // Point to Q table for small path
      add             rAD_Ln2_lo = 0x30, rAD_TB1 // Point to ln2_by_128_lo
(p6)  br.cond.spnt    EXPM1_UNORM // Branch if x unorm
}
;;

EXPM1_COMMON:
{ .mfi
      ldfpd           fMIN_DBL_OFLOW_ARG, fMAX_DBL_MINUS_1_ARG = [rAD_TB1],16
      fclass.m        p10,p0 = f8,0x1e1  // Test for x=+inf, NaN, NaT
      add             rAD_Q2 = 0x50, rAD_Q1   // Point to Q table for small path
}
{ .mfb
      nop.m           0
      nop.f           0
(p8)  br.ret.spnt     b0                        // Exit for x=0, return x
}
;;

{ .mfi
      ldfd            fMAX_DBL_NORM_ARG = [rAD_TB1],16
      nop.f           0
      and             rExp_x = rExp_mask, rSignexp_x // Biased exponent of x
}
{ .mfb
      setf.d          fRSHF = rRshf // Form right shift const 1.100 * 2^63
(p9)  fms.d.s0        f8 = f0,f0,f1            // quick exit for x=-inf
(p9)  br.ret.spnt     b0
}
;;

{ .mfi
      ldfpd           fQD, fQC = [rAD_Q1], 16  // Load coeff for small path
      nop.f           0
      sub             rExp_x = rExp_x, rExp_bias // True exponent of x
}
{ .mfb
      ldfpd           fQB, fQA = [rAD_Q2], 16  // Load coeff for small path
(p10) fma.d.s0        f8 = f8, f1, f0          // For x=+inf, NaN, NaT
(p10) br.ret.spnt     b0                       // Exit for x=+inf, NaN, NaT
}
;;

{ .mfi
      ldfpd           fQ9, fQ8 = [rAD_Q1], 16  // Load coeff for small path
      fma.s1          fXsq = fNormX, fNormX, f0  // x*x for small path
      cmp.gt          p7, p8 = -2, rExp_x      // Test |x| < 2^(-2)
}
{ .mfi
      ldfpd           fQ7, fQ6 = [rAD_Q2], 16  // Load coeff for small path
      nop.f           0
      nop.i           0
}
;;

{ .mfi
      ldfe            fQ5 = [rAD_Q1], 16       // Load coeff for small path
      nop.f           0
      nop.i           0
}
{ .mib
      ldfe            fQ4 = [rAD_Q2], 16       // Load coeff for small path
(p7)  cmp.gt.unc      p6, p7 = -60, rExp_x     // Test |x| < 2^(-60)
(p7)  br.cond.spnt    EXPM1_SMALL              // Branch if 2^-60 <= |x| < 2^-2
}
;;

// W = X * Inv_log2_by_128
// By adding 1.10...0*2^63 we shift and get round_int(W) in significand.
// We actually add 1.10...0*2^56 to X * Inv_log2 to do the same thing.

{ .mfi
      ldfe            fLn2_by_128_hi  = [rAD_TB1],32
      fma.s1          fW_2TO56_RSH  = fNormX, fINV_LN2_2TO63, fRSHF_2TO56
      nop.i           0
}
{ .mfb
      ldfe            fLn2_by_128_lo  = [rAD_Ln2_lo]
(p6)  fma.d.s0        f8 = f8, f8, f8 // If x < 2^-60, result=x+x*x
(p6)  br.ret.spnt     b0              // Exit if x < 2^-60
}
;;

// Divide arguments into the following categories:
//  Certain minus one       p11 - -inf < x <= MAX_DBL_MINUS_1_ARG
//  Possible Overflow       p14 - MAX_DBL_NORM_ARG < x < MIN_DBL_OFLOW_ARG
//  Certain Overflow        p15 - MIN_DBL_OFLOW_ARG <= x < +inf
//
// If the input is really a double arg, then there will never be "Possible
// Overflow" arguments.
//

// After that last load, rAD_TB1 points to the beginning of table 1

{ .mfi
      nop.m           0
      fcmp.ge.s1      p15,p14 = fNormX,fMIN_DBL_OFLOW_ARG
      nop.i           0
}
;;

{ .mfi
      add             rAD_P = 0x80, rAD_TB2
      fcmp.le.s1      p11,p0 = fNormX,fMAX_DBL_MINUS_1_ARG
      nop.i           0
}
;;

{ .mfb
      ldfpd           fP5, fP4  = [rAD_P] ,16
(p14) fcmp.gt.unc.s1  p14,p0 = fNormX,fMAX_DBL_NORM_ARG
(p15) br.cond.spnt    EXPM1_CERTAIN_OVERFLOW
}
;;

// Nfloat = round_int(W)
// The signficand of fW_2TO56_RSH contains the rounded integer part of W,
// as a twos complement number in the lower bits (that is, it may be negative).
// That twos complement number (called N) is put into rN.

// Since fW_2TO56_RSH is scaled by 2^56, it must be multiplied by 2^-56
// before the shift constant 1.10000 * 2^63 is subtracted to yield fNfloat.
// Thus, fNfloat contains the floating point version of N

{ .mfb
      ldfpd           fP3, fP2  = [rAD_P]
      fms.s1          fNfloat = fW_2TO56_RSH, f2TOM56, fRSHF
(p11) br.cond.spnt    EXPM1_CERTAIN_MINUS_ONE
}
;;

{ .mfi
      getf.sig        rN = fW_2TO56_RSH
      nop.f           0
      nop.i           0
}
;;

// rIndex_1 has index_1
// rIndex_2_16 has index_2 * 16
// rBiased_M has M
// rIndex_1_16 has index_1 * 16

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
      nop.i           0
}
;;

// rAD_T1 has address of T1
// rAD_T2 has address if T2

{ .mmi
      add             rBiased_M = rExp_bias, rM
      add             rAD_T2 = rAD_TB2, rIndex_2_16
      shladd          rAD_T1 = rIndex_1, 4, rAD_TB1
}
;;

// Create Scale = 2^M
// Load T1 and T2
{ .mmi
      setf.exp        f2M = rBiased_M
      ldfe            fT2  = [rAD_T2]
      nop.i           0
}
;;

{ .mfi
      ldfe            fT1  = [rAD_T1]
      fmpy.s0         fTmp = fLn2_by_128_lo, fLn2_by_128_lo // Force inexact
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP54 = fR, fP5, fP4
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fP32 = fR, fP3, fP2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fRsq = fR, fR, f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP5432  = fRsq, fP54, fP32
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fS2  = fF,fT2,f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fS1  = f2M,fT1,f0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP = fRsq, fP5432, fR
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fms.s1          fSm1 = fS1,fS2,f1    // S - 1.0
      nop.i           0
}
{ .mfb
      nop.m           0
      fma.s1          fS   = fS1,fS2,f0
(p14) br.cond.spnt    EXPM1_POSSIBLE_OVERFLOW
}
;;

{ .mfb
      nop.m           0
      fma.d.s0        f8 = fS, fP, fSm1
      br.ret.sptk     b0                // Normal path exit
}
;;

// Here if 2^-60 <= |x| <2^-2
// Compute 13th order polynomial
EXPM1_SMALL:
{ .mmf
      ldfe            fQ3 = [rAD_Q1], 16
      ldfe            fQ2 = [rAD_Q2], 16
      fma.s1          fX4 = fXsq, fXsq, f0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fQDC = fQD, fNormX, fQC
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fQBA = fQB, fNormX, fQA
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fQ98 = fQ9, fNormX, fQ8
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fQ76= fQ7, fNormX, fQ6
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fQ54 = fQ5, fNormX, fQ4
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fX6 = fX4, fXsq, f0
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fQ32= fQ3, fNormX, fQ2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fQDCBA = fQDC, fXsq, fQBA
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fQ7654 = fQ76, fXsq, fQ54
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fQDCBA98 = fQDCBA, fXsq, fQ98
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fQ765432 = fQ7654, fXsq, fQ32
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fQDCBA98765432 = fQDCBA98, fX6, fQ765432
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fma.d.s0        f8 = fQDCBA98765432, fXsq, fNormX
      br.ret.sptk     b0                   // Exit small branch
}
;;


EXPM1_POSSIBLE_OVERFLOW:

// Here if fMAX_DBL_NORM_ARG < x < fMIN_DBL_OFLOW_ARG
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
      fma.d.s2        fWre_urm_f8 = fS, fP, fSm1  // Result with wre set
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
(p6)  br.cond.spnt    EXPM1_CERTAIN_OVERFLOW // Branch if overflow
}
;;

{ .mfb
      nop.m           0
      fma.d.s0        f8 = fS, fP, fSm1
      br.ret.sptk     b0                     // Exit if really no overflow
}
;;

EXPM1_CERTAIN_OVERFLOW:
{ .mmi
      sub             rTmp = rExp_mask, r0, 1
;;
      setf.exp        fTmp = rTmp
      nop.i           0
}
;;

{ .mfi
      alloc           r32=ar.pfs,1,4,4,0
      fmerge.s        FR_X = f8,f8
      nop.i           0
}
{ .mfb
      mov             GR_Parameter_TAG = 41
      fma.d.s0        FR_RESULT = fTmp, fTmp, f0    // Set I,O and +INF result
      br.cond.sptk    __libm_error_region
}
;;

// Here if x unorm
EXPM1_UNORM:
{ .mfb
      getf.exp        rSignexp_x = fNormX    // Must recompute if x unorm
      fcmp.eq.s0      p6, p0 = f8, f0        // Set D flag
      br.cond.sptk    EXPM1_COMMON
}
;;

// here if result will be -1 and inexact, x <= -48.0
EXPM1_CERTAIN_MINUS_ONE:
{ .mmi
      mov             rTmp = 1
;;
      setf.exp        fTmp = rTmp
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fms.d.s0        FR_RESULT = fTmp, fTmp, f1 // Set I, rounded -1+eps result
      br.ret.sptk     b0
}
;;

GLOBAL_IEEE754_END(expm1)
libm_alias_double_other (__expm1, expm1)


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
