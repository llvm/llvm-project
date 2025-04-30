.file "expf.s"


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
// 02/02/00 Original version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 08/21/00 Improvements to save 2 cycles on main path, and shorten x=0 case
// 12/07/00 Widen main path, shorten x=inf, nan paths
// 03/15/01 Fix monotonicity problem around x=0 for round to +inf
// 02/05/02 Corrected uninitialize predicate in POSSIBLE_UNDERFLOW path
// 05/20/02 Cleaned up namespace and sf0 syntax
// 07/26/02 Algorithm changed, accuracy improved
// 09/26/02 support of higher precision inputs added, underflow threshold
//          corrected
// 11/15/02 Improved performance on Itanium 2, added possible over/under paths
// 05/30/03 Set inexact flag on unmasked overflow/underflow
// 03/31/05 Reformatted delimiters between data tables
//
//
// API
//*********************************************************************
// float expf(float)
//
// Overview of operation
//*********************************************************************
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
// expf(+0)    = 1.0
// expf(-0)    = 1.0

// expf(+qnan) = +qnan
// expf(-qnan) = -qnan
// expf(+snan) = +qnan
// expf(-snan) = -qnan

// expf(-inf)  = +0
// expf(+inf)  = +inf

// Overflow and Underflow
//*********************************************************************
// expf(x) = largest single normal when
//     x = 88.72283 = 0x42b17217

// expf(x) = smallest single normal when
//     x = -87.33654 = 0xc2aeac4f

// expf(x) = largest round-to-nearest single zero when
//     x = -103.97208 = 0xc2cff1b5


// Registers used
//*********************************************************************
// Floating Point registers used:
// f8, input
// f6,f7, f9 -> f15,  f32 -> f40

// General registers used:
// r3, r23 -> r38

// Predicate registers used:
// p10 -> p15

// Assembly macros
//*********************************************************************
// integer registers used
// scratch
rNJ                   = r3

rTmp                  = r23
rJ                    = r23
rN                    = r24
rTblAddr              = r25
rA3                   = r26
rExpHalf              = r27
rLn2Div64             = r28
r17ones_m1            = r29
rGt_ln                = r29
rRightShifter         = r30
r64DivLn2             = r31
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
fMAX_SGL_ZERO_ARG     = f35
fMAX_SGL_NORM_ARG     = f36
fMIN_SGL_NORM_ARG     = f37
fRSqr                 = f38
fTmp                  = f39
fGt_pln               = f39
fWre_urm_f8           = f40
fFtz_urm_f8           = f40


RODATA
.align 16

LOCAL_OBJECT_START(_expf_table)
data4 0x42b17218         // Smallest sgl arg to overflow sgl result, +88.7228
data4 0xc2cff1b5         // Largest sgl for rnd-to-nearest 0 result, -103.9720
data4 0x42b17217         // Largest sgl arg to give normal sgl result, +88.7228
data4 0xc2aeac4f         // Smallest sgl arg to give normal sgl result, -87.3365
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
LOCAL_OBJECT_END(_expf_table)


.section .text
GLOBAL_IEEE754_ENTRY(expf)

{ .mlx
      addl            rTblAddr = @ltoff(_expf_table),gp
      movl            r64DivLn2 = 0x40571547652B82FE // 64/ln(2)
}
{ .mlx
      addl            rA3 = 0x3E2AA, r0 // high bits of 1.0/6.0 rounded to SP
      movl            rRightShifter = 0x43E8000000000000 // DP Right Shifter
}
;;

{ .mfi
      // point to the beginning of the table
      ld8             rTblAddr = [rTblAddr]
      fclass.m        p14, p0 = f8, 0x22    // test for -INF
      shl             rA3 = rA3, 12  // 0x3E2AA000, approx to 1.0/6.0 in SP
}
{ .mfi
      nop.m           0
      fnorm.s1        fNormX = f8           // normalized x
      addl            rExpHalf = 0xFFFE, r0 // exponent of 1/2
}
;;

{ .mfi
      setf.d          f64DivLn2 = r64DivLn2 // load 64/ln(2) to FP reg
      fclass.m        p15, p0 = f8, 0x1e1   // test for NaT,NaN,+Inf
      nop.i           0
}
{ .mlx
      // load Right Shifter to FP reg
      setf.d          fRightShifter = rRightShifter
      movl            rLn2Div64 = 0x3F862E42FEFA39EF // DP ln(2)/64 in GR
}
;;

{ .mfi
      nop.m           0
      fcmp.eq.s1      p13, p0 = f0, f8      // test for x = 0.0
      nop.i           0
}
{ .mfb
      setf.s          fA3 = rA3             // load A3 to FP reg
(p14) fma.s.s0        f8 = f0, f1, f0       // result if x = -inf
(p14) br.ret.spnt     b0                    // exit here if x = -inf
}
;;

{ .mfi
      setf.exp        fA2 = rExpHalf        // load A2 to FP reg
      fcmp.eq.s0      p6, p0 = f8, f0       // Dummy to flag denorm
      nop.i           0
}
{ .mfb
      setf.d          fLn2Div64 = rLn2Div64 // load ln(2)/64 to FP reg
(p15) fma.s.s0        f8 = f8, f1, f0       // result if x = NaT,NaN,+Inf
(p15) br.ret.spnt     b0                    // exit here if x = NaT,NaN,+Inf
}
;;

{ .mfb
      // overflow and underflow_zero threshold
      ldfps           fMIN_SGL_OFLOW_ARG, fMAX_SGL_ZERO_ARG = [rTblAddr], 8
(p13) fma.s.s0        f8 = f1, f1, f0       // result if x = 0.0
(p13) br.ret.spnt     b0                    // exit here if x =0.0
}
;;

      // max normal and underflow_denorm threshold
{ .mfi
      ldfps           fMAX_SGL_NORM_ARG, fMIN_SGL_NORM_ARG = [rTblAddr], 8
      nop.f           0
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      // x*(64/ln(2)) + Right Shifter
      fma.s1          fNint = fNormX, f64DivLn2, fRightShifter
      nop.i           0
}
;;

// Divide arguments into the following categories:
//  Certain Underflow       p11 - -inf < x <= MAX_SGL_ZERO_ARG
//  Possible Underflow      p13 - MAX_SGL_ZERO_ARG < x < MIN_SGL_NORM_ARG
//  Certain Safe                - MIN_SGL_NORM_ARG <= x <= MAX_SGL_NORM_ARG
//  Possible Overflow       p14 - MAX_SGL_NORM_ARG < x < MIN_SGL_OFLOW_ARG
//  Certain Overflow        p15 - MIN_SGL_OFLOW_ARG <= x < +inf
//
// If the input is really a single arg, then there will never be
// "Possible Overflow" arguments.
//

{ .mfi
      nop.m           0
      // check for overflow
      fcmp.ge.s1      p15, p0 = fNormX, fMIN_SGL_OFLOW_ARG
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      // check for underflow and tiny (+0) result
      fcmp.le.s1      p11, p0 = fNormX, fMAX_SGL_ZERO_ARG
      nop.i           0
}
{ .mfb
      nop.m           0
      fms.s1          fN = fNint, f1, fRightShifter // n in FP register
      // branch out if overflow
(p15) br.cond.spnt    EXP_CERTAIN_OVERFLOW
}
;;

{ .mfb
      getf.sig        rNJ = fNint           // bits of n, j
      // check for underflow and deno result
      fcmp.lt.s1      p13, p0 = fNormX, fMIN_SGL_NORM_ARG
      // branch out if underflow and tiny (+0) result
(p11) br.cond.spnt    EXP_CERTAIN_UNDERFLOW
}
;;

{ .mfi
      nop.m           0
      // check for possible overflow
      fcmp.gt.s1      p14, p0 = fNormX, fMAX_SGL_NORM_ARG
      extr.u          rJ = rNJ, 0, 6        // bits of j
}
{ .mfi
      addl            rN = 0xFFFF - 63, rNJ // biased and shifted n
      fnma.s1         fR = fLn2Div64, fN, fNormX // R = x - N*ln(2)/64
      nop.i           0
}
;;

{ .mfi
      shladd          rJ = rJ, 3, rTblAddr  // address in the 2^(j/64) table
      nop.f           0
      shr             rN = rN, 6            // biased n
}
;;

{ .mfi
      ld8             rJ = [rJ]
      nop.f           0
      shl             rN = rN, 52           // 2^n bits in DP format
}
;;

{ .mfi
      or              rN = rN, rJ // bits of 2^n * 2^(j/64) in DP format
      nop.f           0
      nop.i           0
}
;;

{ .mfi
      setf.d          fT = rN               // 2^n * 2^(j/64)
      fma.s1          fP = fA3, fR, fA2     // A3*R + A2
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fRSqr = fR, fR, f0    // R^2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP = fP, fRSqr, fR    // P = (A3*R + A2)*R^2 + R
      nop.i           0
}
;;

{ .mbb
      nop.m           0
      // branch out if possible underflow
(p13) br.cond.spnt    EXP_POSSIBLE_UNDERFLOW
      // branch out if possible overflow result
(p14) br.cond.spnt    EXP_POSSIBLE_OVERFLOW
}
;;

{ .mfb
      nop.m           0
      // final result in the absence of over- and underflow
      fma.s.s0        f8 = fP, fT, fT
      // exit here in the absence of over- and underflow
      br.ret.sptk     b0
}
;;

EXP_POSSIBLE_OVERFLOW:

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
(p6)  br.cond.spnt    EXP_CERTAIN_OVERFLOW // Branch if overflow
}
;;

{ .mfb
      nop.m           0
      fma.s.s0        f8 = fP, fT, fT
      br.ret.sptk     b0                     // Exit if really no overflow
}
;;

// here if overflow
EXP_CERTAIN_OVERFLOW:
{ .mmi
      addl            r17ones_m1 = 0x1FFFE, r0
;;
      setf.exp        fTmp = r17ones_m1
      nop.i           0
}
;;

{ .mfi
      alloc           r32=ar.pfs,0,3,4,0
      fmerge.s        FR_X = f8,f8
      nop.i           0
}
{ .mfb
      mov             GR_Parameter_TAG = 16
      fma.s.s0        FR_RESULT = fTmp, fTmp, fTmp // Set I,O and +INF result
      br.cond.sptk    __libm_error_region
}
;;

EXP_POSSIBLE_UNDERFLOW:

// Here if fMAX_SGL_ZERO_ARG < x < fMIN_SGL_NORM_ARG
// Underflow is a possibility, not a certainty

// We define an underflow when the answer with
//    ftz set
// is zero (tiny numbers become zero)

// Notice (from below) that if we have an unlimited exponent range,
// then there is an extra machine number E between the largest denormal and
// the smallest normal.

// So if with unbounded exponent we round to E or below, then we are
// tiny and underflow has occurred.

// But notice that you can be in a situation where we are tiny, namely
// rounded to E, but when the exponent is bounded we round to smallest
// normal. So the answer can be the smallest normal with underflow.

//                           E
// -----+--------------------+--------------------+-----
//      |                    |                    |
//   1.1...10 2^-3fff    1.1...11 2^-3fff    1.0...00 2^-3ffe
//   0.1...11 2^-3ffe                                   (biased, 1)
//    largest dn                               smallest normal

{ .mfi
      nop.m           0
      fsetc.s2        0x7F,0x41                // Get user's round mode, set ftz
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fma.s.s2        fFtz_urm_f8 = fP, fT, fT // Result with ftz set
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fsetc.s2        0x7F,0x40                // Turn off ftz in sf2
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fcmp.eq.s1      p6, p7 = fFtz_urm_f8, f0 // Test for underflow
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s.s0        f8 = fP, fT, fT          // Compute result, set I, maybe U
      nop.i           0
}
;;

{ .mbb
      nop.m           0
(p6)  br.cond.spnt    EXP_UNDERFLOW_COMMON     // Branch if really underflow
(p7)  br.ret.sptk     b0                       // Exit if really no underflow
}
;;

EXP_CERTAIN_UNDERFLOW:
// Here if  x < fMAX_SGL_ZERO_ARG
// Result will be zero (or smallest denorm if round to +inf) with I, U set
{ .mmi
      mov             rTmp = 1
;;
      setf.exp        fTmp = rTmp               // Form small normal
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fmerge.se       fTmp = fTmp, f64DivLn2    // Small with non-trial signif
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fma.s.s0        f8 = fTmp, fTmp, f0 // Set I,U, tiny (+0.0) result
      br.cond.sptk    EXP_UNDERFLOW_COMMON
}
;;

EXP_UNDERFLOW_COMMON:
// Determine if underflow result is zero or nonzero
{ .mfi
      alloc           r32=ar.pfs,0,3,4,0
      fcmp.eq.s1      p6, p0 =  f8, f0
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fmerge.s        FR_X = fNormX,fNormX
(p6)  br.cond.spnt    EXP_UNDERFLOW_ZERO
}
;;

EXP_UNDERFLOW_NONZERO:
// Here if  x < fMIN_SGL_NORM_ARG and result nonzero;
// I, U are set
{ .mfb
      mov             GR_Parameter_TAG = 17
      nop.f           0                         // FR_RESULT already set
      br.cond.sptk    __libm_error_region
}
;;

EXP_UNDERFLOW_ZERO:
// Here if x < fMIN_SGL_NORM_ARG and result zero;
// I, U are set
{ .mfb
      mov             GR_Parameter_TAG = 17
      nop.f           0                         // FR_RESULT already set
      br.cond.sptk    __libm_error_region
}
;;

GLOBAL_IEEE754_END(expf)
libm_alias_float_other (__exp, exp)
#ifdef SHARED
.symver expf,expf@@GLIBC_2.27
.weak __expf_compat
.set __expf_compat,__expf
.symver __expf_compat,expf@GLIBC_2.2
#endif


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
