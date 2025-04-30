.file "sinhf.s"


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

// History
//*********************************************************************
// 02/02/00 Initial version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 10/12/00 Update to set denormal operand and underflow flags
// 01/22/01 Fixed to set inexact flag for small args.
// 05/02/01 Reworked to improve speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 11/20/02 Improved algorithm based on expf
// 03/31/05 Reformatted delimiters between data tables
//
// API
//*********************************************************************
// float sinhf(float)
//
// Overview of operation
//*********************************************************************
// Case 1:  0 < |x| < 2^-60
//  Result = x, computed by x+sgn(x)*x^2) to handle flags and rounding
//
// Case 2:  2^-60 < |x| < 0.25
//  Evaluate sinh(x) by a 9th order polynomial
//  Care is take for the order of multiplication; and A2 is not exactly 1/5!,
//  A3 is not exactly 1/7!, etc.
//  sinh(x) = x + (A1*x^3 + A2*x^5 + A3*x^7 + A4*x^9)
//
// Case 3:  0.25 < |x| < 89.41598
//  Algorithm is based on the identity sinh(x) = ( exp(x) - exp(-x) ) / 2.
//  The algorithm for exp is described as below.  There are a number of
//  economies from evaluating both exp(x) and exp(-x).  Although we
//  are evaluating both quantities, only where the quantities diverge do we
//  duplicate the computations.  The basic algorithm for exp(x) is described
//  below.
//
// Take the input x. w is "how many log2/128 in x?"
//  w = x * 64/log2
//  NJ = int(w)
//  x = NJ*log2/64 + R

//  NJ = 64*n + j
//  x = n*log2 + (log2/64)*j + R
//
//  So, exp(x) = 2^n * 2^(j/64)* exp(R)
//
//  T =  2^n * 2^(j/64)
//       Construct 2^n
//       Get 2^(j/64) table
//           actually all the entries of 2^(j/64) table are stored in DP and
//           with exponent bits set to 0 -> multiplication on 2^n can be
//           performed by doing logical "or" operation with bits presenting 2^n

//  exp(R) = 1 + (exp(R) - 1)
//  P = exp(R) - 1 approximated by Taylor series of 3rd degree
//      P = A3*R^3 + A2*R^2 + R, A3 = 1/6, A2 = 1/2
//

//  The final result is reconstructed as follows
//  exp(x) = T + T*P

// Special values
//*********************************************************************
// sinhf(+0)    = +0
// sinhf(-0)    = -0

// sinhf(+qnan) = +qnan
// sinhf(-qnan) = -qnan
// sinhf(+snan) = +qnan
// sinhf(-snan) = -qnan

// sinhf(-inf)  = -inf
// sinhf(+inf)  = +inf

// Overflow and Underflow
//*********************************************************************
// sinhf(x) = largest single normal when
//     x = 89.41598 = 0x42b2d4fc
//
// Underflow is handled as described in case 1 above

// Registers used
//*********************************************************************
// Floating Point registers used:
// f8 input, output
// f6,f7, f9 -> f15,  f32 -> f45

// General registers used:
// r2, r3, r16 -> r38

// Predicate registers used:
// p6 -> p15

// Assembly macros
//*********************************************************************
// integer registers used
// scratch
rNJ                   = r2
rNJ_neg               = r3

rJ_neg                = r16
rN_neg                = r17
rSignexp_x            = r18
rExp_x                = r18
rExp_mask             = r19
rExp_bias             = r20
rAd1                  = r21
rAd2                  = r22
rJ                    = r23
rN                    = r24
rTblAddr              = r25
rA3                   = r26
rExpHalf              = r27
rLn2Div64             = r28
rGt_ln                = r29
r17ones_m1            = r29
rRightShifter         = r30
rJ_mask               = r30
r64DivLn2             = r31
rN_mask               = r31
// stacked
GR_SAVE_PFS           = r32
GR_SAVE_B0            = r33
GR_SAVE_GP            = r34
GR_Parameter_X        = r35
GR_Parameter_Y        = r36
GR_Parameter_RESULT   = r37
GR_Parameter_TAG      = r38

// floating point registers used
FR_X                  = f10
FR_Y                  = f1
FR_RESULT             = f8
// scratch
fRightShifter         = f6
f64DivLn2             = f7
fNormX                = f9
fNint                 = f10
fN                    = f11
fR                    = f12
fLn2Div64             = f13
fA2                   = f14
fA3                   = f15
// stacked
fP                    = f32
fT                    = f33
fMIN_SGL_OFLOW_ARG    = f34
fMAX_SGL_NORM_ARG     = f35
fRSqr                 = f36
fA1                   = f37
fA21                  = f37
fA4                   = f38
fA43                  = f38
fA4321                = f38
fX4                   = f39
fTmp                  = f39
fGt_pln               = f39
fWre_urm_f8           = f40
fXsq                  = f40
fP_neg                = f41
fX3                   = f41
fT_neg                = f42
fExp                  = f43
fExp_neg              = f44
fAbsX                 = f45


RODATA
.align 16

LOCAL_OBJECT_START(_sinhf_table)
data4 0x42b2d4fd         // Smallest single arg to overflow single result
data4 0x42b2d4fc         // Largest single arg to give normal single result
data4 0x00000000         // pad
data4 0x00000000         // pad
//
// 2^(j/64) table, j goes from 0 to 63
data8 0x0000000000000000 // 2^(0/64)
data8 0x00002C9A3E778061 // 2^(1/64)
data8 0x000059B0D3158574 // 2^(2/64)
data8 0x0000874518759BC8 // 2^(3/64)
data8 0x0000B5586CF9890F // 2^(4/64)
data8 0x0000E3EC32D3D1A2 // 2^(5/64)
data8 0x00011301D0125B51 // 2^(6/64)
data8 0x0001429AAEA92DE0 // 2^(7/64)
data8 0x000172B83C7D517B // 2^(8/64)
data8 0x0001A35BEB6FCB75 // 2^(9/64)
data8 0x0001D4873168B9AA // 2^(10/64)
data8 0x0002063B88628CD6 // 2^(11/64)
data8 0x0002387A6E756238 // 2^(12/64)
data8 0x00026B4565E27CDD // 2^(13/64)
data8 0x00029E9DF51FDEE1 // 2^(14/64)
data8 0x0002D285A6E4030B // 2^(15/64)
data8 0x000306FE0A31B715 // 2^(16/64)
data8 0x00033C08B26416FF // 2^(17/64)
data8 0x000371A7373AA9CB // 2^(18/64)
data8 0x0003A7DB34E59FF7 // 2^(19/64)
data8 0x0003DEA64C123422 // 2^(20/64)
data8 0x0004160A21F72E2A // 2^(21/64)
data8 0x00044E086061892D // 2^(22/64)
data8 0x000486A2B5C13CD0 // 2^(23/64)
data8 0x0004BFDAD5362A27 // 2^(24/64)
data8 0x0004F9B2769D2CA7 // 2^(25/64)
data8 0x0005342B569D4F82 // 2^(26/64)
data8 0x00056F4736B527DA // 2^(27/64)
data8 0x0005AB07DD485429 // 2^(28/64)
data8 0x0005E76F15AD2148 // 2^(29/64)
data8 0x0006247EB03A5585 // 2^(30/64)
data8 0x0006623882552225 // 2^(31/64)
data8 0x0006A09E667F3BCD // 2^(32/64)
data8 0x0006DFB23C651A2F // 2^(33/64)
data8 0x00071F75E8EC5F74 // 2^(34/64)
data8 0x00075FEB564267C9 // 2^(35/64)
data8 0x0007A11473EB0187 // 2^(36/64)
data8 0x0007E2F336CF4E62 // 2^(37/64)
data8 0x00082589994CCE13 // 2^(38/64)
data8 0x000868D99B4492ED // 2^(39/64)
data8 0x0008ACE5422AA0DB // 2^(40/64)
data8 0x0008F1AE99157736 // 2^(41/64)
data8 0x00093737B0CDC5E5 // 2^(42/64)
data8 0x00097D829FDE4E50 // 2^(43/64)
data8 0x0009C49182A3F090 // 2^(44/64)
data8 0x000A0C667B5DE565 // 2^(45/64)
data8 0x000A5503B23E255D // 2^(46/64)
data8 0x000A9E6B5579FDBF // 2^(47/64)
data8 0x000AE89F995AD3AD // 2^(48/64)
data8 0x000B33A2B84F15FB // 2^(49/64)
data8 0x000B7F76F2FB5E47 // 2^(50/64)
data8 0x000BCC1E904BC1D2 // 2^(51/64)
data8 0x000C199BDD85529C // 2^(52/64)
data8 0x000C67F12E57D14B // 2^(53/64)
data8 0x000CB720DCEF9069 // 2^(54/64)
data8 0x000D072D4A07897C // 2^(55/64)
data8 0x000D5818DCFBA487 // 2^(56/64)
data8 0x000DA9E603DB3285 // 2^(57/64)
data8 0x000DFC97337B9B5F // 2^(58/64)
data8 0x000E502EE78B3FF6 // 2^(59/64)
data8 0x000EA4AFA2A490DA // 2^(60/64)
data8 0x000EFA1BEE615A27 // 2^(61/64)
data8 0x000F50765B6E4540 // 2^(62/64)
data8 0x000FA7C1819E90D8 // 2^(63/64)
LOCAL_OBJECT_END(_sinhf_table)

LOCAL_OBJECT_START(sinh_p_table)
data8 0x3ec749d84bc96d7d // A4
data8 0x3f2a0168d09557cf // A3
data8 0x3f811111326ed15a // A2
data8 0x3fc55555552ed1e2 // A1
LOCAL_OBJECT_END(sinh_p_table)


.section .text
GLOBAL_IEEE754_ENTRY(sinhf)

{ .mlx
      getf.exp        rSignexp_x = f8  // Must recompute if x unorm
      movl            r64DivLn2 = 0x40571547652B82FE // 64/ln(2)
}
{ .mlx
      addl            rTblAddr = @ltoff(_sinhf_table),gp
      movl            rRightShifter = 0x43E8000000000000 // DP Right Shifter
}
;;

{ .mfi
      // point to the beginning of the table
      ld8             rTblAddr = [rTblAddr]
      fclass.m        p6, p0 = f8, 0x0b   // Test for x=unorm
      addl            rA3 = 0x3E2AA, r0   // high bits of 1.0/6.0 rounded to SP
}
{ .mfi
      nop.m           0
      fnorm.s1        fNormX = f8 // normalized x
      addl            rExpHalf = 0xFFFE, r0 // exponent of 1/2
}
;;

{ .mfi
      setf.d          f64DivLn2 = r64DivLn2 // load 64/ln(2) to FP reg
      fclass.m        p15, p0 = f8, 0x1e3   // test for NaT,NaN,Inf
      nop.i           0
}
{ .mlx
      // load Right Shifter to FP reg
      setf.d          fRightShifter = rRightShifter
      movl            rLn2Div64 = 0x3F862E42FEFA39EF // DP ln(2)/64 in GR
}
;;

{ .mfi
      mov             rExp_mask = 0x1ffff
      fcmp.eq.s1      p13, p0 = f0, f8 // test for x = 0.0
      shl             rA3 = rA3, 12    // 0x3E2AA000, approx to 1.0/6.0 in SP
}
{ .mfb
      nop.m           0
      nop.f           0
(p6)  br.cond.spnt    SINH_UNORM            // Branch if x=unorm
}
;;

SINH_COMMON:
{ .mfi
      setf.exp        fA2 = rExpHalf        // load A2 to FP reg
      nop.f           0
      mov             rExp_bias = 0xffff
}
{ .mfb
      setf.d          fLn2Div64 = rLn2Div64 // load ln(2)/64 to FP reg
(p15) fma.s.s0        f8 = f8, f1, f0       // result if x = NaT,NaN,Inf
(p15) br.ret.spnt     b0                    // exit here if x = NaT,NaN,Inf
}
;;

{ .mfi
      // min overflow and max normal threshold
      ldfps           fMIN_SGL_OFLOW_ARG, fMAX_SGL_NORM_ARG = [rTblAddr], 8
      nop.f           0
      and             rExp_x = rExp_mask, rSignexp_x // Biased exponent of x
}
{ .mfb
      setf.s          fA3 = rA3                  // load A3 to FP reg
      nop.f           0
(p13) br.ret.spnt     b0                         // exit here if x=0.0, return x
}
;;

{ .mfi
      sub             rExp_x = rExp_x, rExp_bias // True exponent of x
      fmerge.s        fAbsX = f0, fNormX         // Form |x|
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      // x*(64/ln(2)) + Right Shifter
      fma.s1          fNint = fNormX, f64DivLn2, fRightShifter
      add             rTblAddr = 8, rTblAddr
}
{ .mfb
      cmp.gt          p7, p0 = -2, rExp_x        // Test |x| < 2^(-2)
      fma.s1          fXsq = fNormX, fNormX, f0  // x*x for small path
(p7)  br.cond.spnt    SINH_SMALL                 // Branch if 0 < |x| < 2^-2
}
;;

{ .mfi
      nop.m           0
      // check for overflow
      fcmp.ge.s1      p12, p13 = fAbsX, fMIN_SGL_OFLOW_ARG
      mov             rJ_mask = 0x3f             // 6-bit mask for J
}
;;

{ .mfb
      nop.m           0
      fms.s1          fN = fNint, f1, fRightShifter // n in FP register
      // branch out if overflow
(p12) br.cond.spnt    SINH_CERTAIN_OVERFLOW
}
;;

{ .mfi
      getf.sig        rNJ = fNint                   // bits of n, j
      // check for possible overflow
      fcmp.gt.s1      p13, p0 = fAbsX, fMAX_SGL_NORM_ARG
      nop.i           0
}
;;

{ .mfi
      addl            rN = 0xFFBF - 63, rNJ      // biased and shifted n-1,j
      fnma.s1         fR = fLn2Div64, fN, fNormX // R = x - N*ln(2)/64
      and             rJ = rJ_mask, rNJ          // bits of j
}
{ .mfi
      sub             rNJ_neg = r0, rNJ          // bits of n, j for -x
      nop.f           0
      andcm           rN_mask = -1, rJ_mask      // 0xff...fc0 to mask N
}
;;

{ .mfi
      shladd          rJ = rJ, 3, rTblAddr // address in the 2^(j/64) table
      nop.f           0
      and             rN = rN_mask, rN     // biased, shifted n-1
}
{ .mfi
      addl            rN_neg = 0xFFBF - 63, rNJ_neg // -x biased, shifted n-1,j
      nop.f           0
      and             rJ_neg = rJ_mask, rNJ_neg     // bits of j for -x
}
;;

{ .mfi
      ld8             rJ = [rJ]                    // Table value
      nop.f           0
      shl             rN = rN, 46 // 2^(n-1) bits in DP format
}
{ .mfi
      shladd          rJ_neg = rJ_neg, 3, rTblAddr // addr in 2^(j/64) table -x
      nop.f           0
      and             rN_neg = rN_mask, rN_neg     // biased, shifted n-1 for -x
}
;;

{ .mfi
      ld8             rJ_neg = [rJ_neg]            // Table value for -x
      nop.f           0
      shl             rN_neg = rN_neg, 46 // 2^(n-1) bits in DP format for -x
}
;;

{ .mfi
      or              rN = rN, rJ // bits of 2^n * 2^(j/64) in DP format
      nop.f           0
      nop.i           0
}
;;

{ .mmf
      setf.d          fT = rN            // 2^(n-1) * 2^(j/64)
      or              rN_neg = rN_neg, rJ_neg // -x bits of 2^n * 2^(j/64) in DP
      fma.s1          fRSqr = fR, fR, f0 // R^2
}
;;

{ .mfi
      setf.d          fT_neg = rN_neg    // 2^(n-1) * 2^(j/64) for -x
      fma.s1          fP = fA3, fR, fA2  // A3*R + A2
      nop.i           0
}
{ .mfi
      nop.m           0
      fnma.s1         fP_neg = fA3, fR, fA2  // A3*R + A2 for -x
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP = fP, fRSqr, fR // P = (A3*R + A2)*R^2 + R
      nop.i           0
}
{ .mfi
      nop.m           0
      fms.s1          fP_neg = fP_neg, fRSqr, fR // P = (A3*R + A2)*R^2 + R, -x
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fmpy.s0         fTmp = fLn2Div64, fLn2Div64       // Force inexact
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fExp = fP, fT, fT                 // exp(x)/2
      nop.i           0
}
{ .mfb
      nop.m           0
      fma.s1          fExp_neg = fP_neg, fT_neg, fT_neg // exp(-x)/2
      // branch out if possible overflow result
(p13) br.cond.spnt    SINH_POSSIBLE_OVERFLOW
}
;;

{ .mfb
      nop.m           0
      // final result in the absence of overflow
      fms.s.s0        f8 = fExp, f1, fExp_neg  // result = (exp(x)-exp(-x))/2
      // exit here in the absence of overflow
      br.ret.sptk     b0              // Exit main path, 0.25 <= |x| < 89.41598
}
;;

// Here if 0 < |x| < 0.25.  Evaluate 9th order polynomial.
SINH_SMALL:
{ .mfi
      add             rAd1 = 0x200, rTblAddr
      fcmp.lt.s1      p7, p8 = fNormX, f0       // Test sign of x
      cmp.gt          p6, p0 = -60, rExp_x      // Test |x| < 2^(-60)
}
{ .mfi
      add             rAd2 = 0x210, rTblAddr
      nop.f           0
      nop.i           0
}
;;

{ .mmb
      ldfpd           fA4, fA3 = [rAd1]
      ldfpd           fA2, fA1 = [rAd2]
(p6)  br.cond.spnt    SINH_VERY_SMALL           // Branch if |x| < 2^(-60)
}
;;

{ .mfi
      nop.m           0
      fma.s1          fX3 = fXsq, fNormX, f0
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
      fma.s1          fA43 = fXsq, fA4, fA3
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fA21 = fXsq, fA2, fA1
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fA4321 = fX4, fA43, fA21
      nop.i           0
}
;;

// Dummy multiply to generate inexact
{ .mfi
      nop.m           0
      fmpy.s0         fTmp = fA4, fA4
      nop.i           0
}
{ .mfb
      nop.m           0
      fma.s.s0        f8 = fA4321, fX3, fNormX
      br.ret.sptk     b0                // Exit if 2^-60 < |x| < 0.25
}
;;

SINH_VERY_SMALL:
// Here if 0 < |x| < 2^-60
// Compute result by x + sgn(x)*x^2 to get properly rounded result
.pred.rel "mutex",p7,p8
{ .mfi
      nop.m           0
(p7)  fnma.s.s0       f8 = fNormX, fNormX, fNormX // If x<0 result ~ x-x^2
      nop.i           0
}
{ .mfb
      nop.m           0
(p8)  fma.s.s0        f8 = fNormX, fNormX, fNormX // If x>0 result ~ x+x^2
      br.ret.sptk     b0                          // Exit if |x| < 2^-60
}
;;

SINH_POSSIBLE_OVERFLOW:

// Here if fMAX_SGL_NORM_ARG < x < fMIN_SGL_OFLOW_ARG
// This cannot happen if input is a single, only if input higher precision.
// Overflow is a possibility, not a certainty.

// Recompute result using status field 2 with user's rounding mode,
// and wre set.  If result is larger than largest single, then we have
// overflow

{ .mfi
      mov             rGt_ln  = 0x1007f // Exponent for largest single + 1 ulp
      fsetc.s2        0x7F,0x42         // Get user's round mode, set wre
      nop.i           0
}
;;

{ .mfi
      setf.exp        fGt_pln = rGt_ln  // Create largest single + 1 ulp
      fma.s.s2        fWre_urm_f8 = fP, fT, fT    // Result with wre set
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
      fma.s.s0        f8 = fP, fT, fT
      br.ret.sptk     b0                     // Exit if really no overflow
}
;;

// here if overflow
SINH_CERTAIN_OVERFLOW:
{ .mfi
      addl            r17ones_m1 = 0x1FFFE, r0
      fcmp.lt.s1      p6, p7 = fNormX, f0     // Test for x < 0
      nop.i           0
}
;;

{ .mmf
      alloc           r32 = ar.pfs, 0, 3, 4, 0 // get some registers
      setf.exp        fTmp = r17ones_m1
      fmerge.s        FR_X = f8,f8
}
;;

{ .mfi
      mov             GR_Parameter_TAG = 128
(p6)  fnma.s.s0       FR_RESULT = fTmp, fTmp, f0 // Set I,O and -INF result
      nop.i           0
}
{ .mfb
      nop.m           0
(p7)  fma.s.s0        FR_RESULT = fTmp, fTmp, f0 // Set I,O and +INF result
      br.cond.sptk    __libm_error_region
}
;;

// Here if x unorm
SINH_UNORM:
{ .mfb
      getf.exp        rSignexp_x = fNormX    // Must recompute if x unorm
      fcmp.eq.s0      p6, p0 = f8, f0        // Set D flag
      br.cond.sptk    SINH_COMMON            // Return to main path
}
;;

GLOBAL_IEEE754_END(sinhf)
libm_alias_float_other (__sinh, sinh)


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
      stfs [GR_Parameter_Y] = FR_Y,16         // Store Parameter 2 on stack
      add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mfi
      stfs [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
      nop.f 0
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y // Parameter 3 address
}
{ .mib
      stfs [GR_Parameter_Y] = FR_RESULT       // Store Parameter 3 on stack
      add   GR_Parameter_Y = -16,GR_Parameter_Y
      br.call.sptk b0=__libm_error_support#   // Call error handling function
};;

{ .mmi
      add   GR_Parameter_RESULT = 48,sp
      nop.m 0
      nop.i 0
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
