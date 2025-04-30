.file "expf_m1.s"


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
// 02/02/00 Initial Version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 07/07/01 Improved speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 11/20/02 Improved speed, algorithm based on expf
// 03/31/05 Reformatted delimiters between data tables
//
//
// API
//*********************************************************************
// float expm1f(float)
//
// Overview of operation
//*********************************************************************
// 1. Inputs of Nan, Inf, Zero, NatVal handled with special paths
//
// 2. |x| < 2^-40
//    Result = x, computed by x + x*x to handle appropriate flags and rounding
//
// 3. 2^-40 <= |x| < 2^-2
//    Result determined by 8th order Taylor series polynomial
//    expm1f(x) = x + A2*x^2 + ... + A8*x^8
//
// 4. x < -24.0
//    Here we know result is essentially -1 + eps, where eps only affects
//    rounded result.  Set I.
//
// 5. x >= 88.7228
//    Result overflows.  Set I, O, and call error support
//
// 6. 2^-2 <= x < 88.7228  or  -24.0 <= x < -2^-2
//    This is the main path.  The algorithm is described below:

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
//  expm1f(x) = T*P + (T - 1.0)

// Special values
//*********************************************************************
// expm1f(+0)    = +0.0
// expm1f(-0)    = -0.0

// expm1f(+qnan) = +qnan
// expm1f(-qnan) = -qnan
// expm1f(+snan) = +qnan
// expm1f(-snan) = -qnan

// expm1f(-inf)  = -1.0
// expm1f(+inf)  = +inf

// Overflow and Underflow
//*********************************************************************
// expm1f(x) = largest single normal when
//     x = 88.7228 = 0x42b17217
//
// Underflow is handled as described in case 2 above.


// Registers used
//*********************************************************************
// Floating Point registers used:
// f8, input
// f6,f7, f9 -> f15,  f32 -> f45

// General registers used:
// r3, r20 -> r38

// Predicate registers used:
// p9 -> p15

// Assembly macros
//*********************************************************************
// integer registers used
// scratch
rNJ                   = r3

rExp_half             = r20
rSignexp_x            = r21
rExp_x                = r22
rExp_mask             = r23
rExp_bias             = r24
rTmp                  = r25
rM1_lim               = r25
rGt_ln                = r25
rJ                    = r26
rN                    = r27
rTblAddr              = r28
rLn2Div64             = r29
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
fX3                   = f33
fT                    = f34
fMIN_SGL_OFLOW_ARG    = f35
fMAX_SGL_NORM_ARG     = f36
fMAX_SGL_MINUS_1_ARG  = f37
fA4                   = f38
fA43                  = f38
fA432                 = f38
fRSqr                 = f39
fA5                   = f40
fTmp                  = f41
fGt_pln               = f41
fXsq                  = f41
fA7                   = f42
fA6                   = f43
fA65                  = f43
fTm1                  = f44
fA8                   = f45
fA87                  = f45
fA8765                = f45
fA8765432             = f45
fWre_urm_f8           = f45

RODATA
.align 16
LOCAL_OBJECT_START(_expf_table)
data8 0x3efa01a01a01a01a // A8 = 1/8!
data8 0x3f2a01a01a01a01a // A7 = 1/7!
data8 0x3f56c16c16c16c17 // A6 = 1/6!
data8 0x3f81111111111111 // A5 = 1/5!
data8 0x3fa5555555555555 // A4 = 1/4!
data8 0x3fc5555555555555 // A3 = 1/3!
//
data4 0x42b17218         // Smallest sgl arg to overflow sgl result
data4 0x42b17217         // Largest sgl arg to give sgl result
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
GLOBAL_IEEE754_ENTRY(expm1f)

{ .mlx
      getf.exp        rSignexp_x = f8      // Must recompute if x unorm
      movl            r64DivLn2 = 0x40571547652B82FE // 64/ln(2)
}
{ .mlx
      addl            rTblAddr = @ltoff(_expf_table),gp
      movl            rRightShifter = 0x43E8000000000000 // DP Right Shifter
}
;;

{ .mfi
      // point to the beginning of the table
      ld8             rTblAddr = [rTblAddr]
      fclass.m        p14, p0 = f8 , 0x22  // test for -INF
      mov             rExp_mask = 0x1ffff   // Exponent mask
}
{ .mfi
      nop.m           0
      fnorm.s1        fNormX = f8 // normalized x
      nop.i           0
}
;;

{ .mfi
      setf.d          f64DivLn2 = r64DivLn2 // load 64/ln(2) to FP reg
      fclass.m        p9, p0 = f8 , 0x0b    // test for x unorm
      mov             rExp_bias = 0xffff    // Exponent bias
}
{ .mlx
      // load Right Shifter to FP reg
      setf.d          fRightShifter = rRightShifter
      movl            rLn2Div64 = 0x3F862E42FEFA39EF // DP ln(2)/64 in GR
}
;;

{ .mfi
      ldfpd           fA8, fA7 = [rTblAddr], 16
      fcmp.eq.s1      p13, p0 = f0, f8      // test for x = 0.0
      mov             rExp_half = 0xfffe
}
{ .mfb
      setf.d          fLn2Div64 = rLn2Div64 // load ln(2)/64 to FP reg
      nop.f           0
(p9)  br.cond.spnt    EXPM1_UNORM // Branch if x unorm
}
;;

EXPM1_COMMON:
{ .mfb
      ldfpd           fA6, fA5 = [rTblAddr], 16
(p14) fms.s.s0        f8 = f0, f0, f1       // result if x = -inf
(p14) br.ret.spnt     b0                    // exit here if x = -inf
}
;;

{ .mfb
      ldfpd           fA4, fA3 = [rTblAddr], 16
      fclass.m        p15, p0 = f8 , 0x1e1  // test for NaT,NaN,+Inf
(p13) br.ret.spnt     b0                    // exit here if x =0.0, result is x
}
;;

{ .mfi
      // overflow thresholds
      ldfps           fMIN_SGL_OFLOW_ARG, fMAX_SGL_NORM_ARG = [rTblAddr], 8
      fma.s1          fXsq = fNormX, fNormX, f0      // x^2 for small path
      and             rExp_x = rExp_mask, rSignexp_x // Biased exponent of x
}
{ .mlx
      nop.m           0
      movl            rM1_lim = 0xc1c00000  // Minus -1 limit (-24.0), SP
}
;;

{ .mfi
      setf.exp        fA2 = rExp_half
      // x*(64/ln(2)) + Right Shifter
      fma.s1          fNint = fNormX, f64DivLn2, fRightShifter
      sub             rExp_x = rExp_x, rExp_bias     // True exponent of x
}
{ .mfb
      nop.m           0
(p15) fma.s.s0        f8 = f8, f1, f0       // result if x = NaT,NaN,+Inf
(p15) br.ret.spnt     b0                    // exit here if x = NaT,NaN,+Inf
}
;;

{ .mfi
      setf.s          fMAX_SGL_MINUS_1_ARG = rM1_lim // -1 threshold, -24.0
      nop.f           0
      cmp.gt          p7, p8 = -2, rExp_x      // Test |x| < 2^(-2)
}
;;

{ .mfi
(p7)  cmp.gt.unc      p6, p7 = -40, rExp_x     // Test |x| < 2^(-40)
      fma.s1          fA87 = fA8, fNormX, fA7  // Small path, A8*x+A7
      nop.i           0
}
{ .mfi
      nop.m           0
      fma.s1          fA65 = fA6, fNormX, fA5  // Small path, A6*x+A5
      nop.i           0
}
;;

{ .mfb
      nop.m           0
(p6)  fma.s.s0        f8 = f8, f8, f8          // If x < 2^-40, result=x+x*x
(p6)  br.ret.spnt     b0                       // Exit if x < 2^-40
}
;;

{ .mfi
      nop.m           0
      // check for overflow
      fcmp.gt.s1      p15, p14 = fNormX, fMIN_SGL_OFLOW_ARG
      nop.i           0
}
{ .mfi
      nop.m           0
      fms.s1          fN = fNint, f1, fRightShifter // n in FP register
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p7)  fma.s1          fA43 = fA4, fNormX, fA3   // Small path, A4*x+A3
      nop.i           0
}
;;

{ .mfi
      getf.sig        rNJ = fNint               // bits of n, j
(p7)  fma.s1          fA8765 = fA87, fXsq, fA65 // Small path, A87*xsq+A65
      nop.i           0
}
{ .mfb
      nop.m           0
(p7)  fma.s1          fX3 = fXsq, fNormX, f0    // Small path, x^3
      // branch out if overflow
(p15) br.cond.spnt    EXPM1_CERTAIN_OVERFLOW
}
;;

{ .mfi
      addl            rN = 0xffff-63, rNJ    // biased and shifted n
      fnma.s1         fR = fLn2Div64, fN, fNormX // R = x - N*ln(2)/64
      extr.u          rJ = rNJ , 0 , 6       // bits of j
}
;;

{ .mfi
      shladd          rJ = rJ, 3, rTblAddr   // address in the 2^(j/64) table
      // check for certain -1
      fcmp.le.s1      p13, p0 = fNormX, fMAX_SGL_MINUS_1_ARG
      shr             rN = rN, 6             // biased n
}
{ .mfi
      nop.m           0
(p7)  fma.s1          fA432 = fA43, fNormX, fA2 // Small path, A43*x+A2
      nop.i           0
}
;;

{ .mfi
      ld8             rJ = [rJ]
      nop.f           0
      shl             rN = rN , 52           // 2^n bits in DP format
}
;;

{ .mmi
      or              rN = rN, rJ        // bits of 2^n * 2^(j/64) in DP format
(p13) mov             rTmp = 1           // Make small value for -1 path
      nop.i           0
}
;;

{ .mfi
      setf.d          fT = rN            // 2^n
      // check for possible overflow (only happens if input higher precision)
(p14) fcmp.gt.s1      p14, p0 = fNormX, fMAX_SGL_NORM_ARG
      nop.i           0
}
{ .mfi
      nop.m           0
(p7)  fma.s1          fA8765432 = fA8765, fX3, fA432 // A8765*x^3+A432
      nop.i           0
}
;;

{ .mfi
(p13) setf.exp        fTmp = rTmp        // Make small value for -1 path
      fma.s1          fP = fA3, fR, fA2  // A3*R + A2
      nop.i           0
}
{ .mfb
      nop.m           0
      fma.s1          fRSqr = fR, fR, f0 // R^2
(p13) br.cond.spnt    EXPM1_CERTAIN_MINUS_ONE // Branch if x < -24.0
}
;;

{ .mfb
      nop.m           0
(p7)  fma.s.s0        f8 = fA8765432, fXsq, fNormX // Small path,
                                         // result=xsq*A8765432+x
(p7)  br.ret.spnt     b0                 // Exit if 2^-40 <= |x| < 2^-2
}
;;

{ .mfi
      nop.m           0
      fma.s1          fP = fP, fRSqr, fR // P = (A3*R + A2)*Rsqr + R
      nop.i           0
}
;;

{ .mfb
      nop.m           0
      fms.s1          fTm1 = fT, f1, f1  // T - 1.0
(p14) br.cond.spnt    EXPM1_POSSIBLE_OVERFLOW
}
;;

{ .mfb
      nop.m           0
      fma.s.s0        f8 = fP, fT, fTm1
      br.ret.sptk     b0                 // Result for main path
                                         // minus_one_limit < x < -2^-2
                                         // and +2^-2 <= x < overflow_limit
}
;;

// Here if x unorm
EXPM1_UNORM:
{ .mfb
      getf.exp        rSignexp_x = fNormX // Must recompute if x unorm
      fcmp.eq.s0      p6, p0 = f8, f0     // Set D flag
      br.cond.sptk    EXPM1_COMMON
}
;;

// here if result will be -1 and inexact, x <= -24.0
EXPM1_CERTAIN_MINUS_ONE:
{ .mfb
      nop.m           0
      fms.s.s0        f8 = fTmp, fTmp, f1  // Result -1, and Inexact set
      br.ret.sptk     b0
}
;;

EXPM1_POSSIBLE_OVERFLOW:

// Here if fMAX_SGL_NORM_ARG < x < fMIN_SGL_OFLOW_ARG
// This cannot happen if input is a single, only if input higher precision.
// Overflow is a possibility, not a certainty.

// Recompute result using status field 2 with user's rounding mode,
// and wre set.  If result is larger than largest single, then we have
// overflow

{ .mfi
      mov             rGt_ln  = 0x1007f // Exponent for largest sgl + 1 ulp
      fsetc.s2        0x7F,0x42         // Get user's round mode, set wre
      nop.i           0
}
;;

{ .mfi
      setf.exp        fGt_pln = rGt_ln  // Create largest single + 1 ulp
      fma.s.s2        fWre_urm_f8 = fP, fT, fTm1  // Result with wre set
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
      fma.s.s0        f8 = fP, fT, fTm1
      br.ret.sptk     b0                     // Exit if really no overflow
}
;;

// here if overflow
EXPM1_CERTAIN_OVERFLOW:
{ .mmi
      addl            rTmp = 0x1FFFE, r0;;
      setf.exp        fTmp = rTmp
      nop.i 999
}
;;

{ .mfi
      alloc           r32 = ar.pfs, 0, 3, 4, 0 // get some registers
      fmerge.s        FR_X = fNormX,fNormX
      nop.i           0
}
{ .mfb
      mov             GR_Parameter_TAG = 43
      fma.s.s0        FR_RESULT = fTmp, fTmp, f0 // Set I,O and +INF result
      br.cond.sptk    __libm_error_region
}
;;

GLOBAL_IEEE754_END(expm1f)
libm_alias_float_other (__expm1, expm1)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
      add   GR_Parameter_Y=-32,sp             // Parameter 2 value
      nop.f 999
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
