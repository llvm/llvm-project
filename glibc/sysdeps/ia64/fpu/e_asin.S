.file "asin.s"


// Copyright (c) 2000 - 2003 Intel Corporation
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
//==============================================================
// 02/02/00 Initial version
// 08/17/00 New and much faster algorithm.
// 08/31/00 Avoided bank conflicts on loads, shortened |x|=1 path,
//          fixed mfb split issue stalls.
// 12/19/00 Fixed small arg cases to force inexact, or inexact and underflow.
// 08/02/02 New and much faster algorithm II
// 02/06/03 Reordered header: .section, .global, .proc, .align

// Description
//=========================================
// The asin function computes the principal value of the arc sine of x.
// asin(0) returns 0, asin(1) returns pi/2, asin(-1) returns -pi/2.
// A doman error occurs for arguments not in the range [-1,+1].
//
// The asin function returns the arc sine in the range [-pi/2, +pi/2] radians.
//
// There are 8 paths:
// 1. x = +/-0.0
//    Return asin(x) = +/-0.0
//
// 2. 0.0 < |x| < 0.625
//    Return asin(x) = x + x^3 *PolA(x^2)
//    where PolA(x^2) = A3 + A5*x^2 + A7*x^4 +...+ A35*x^32
//
// 3. 0.625 <=|x| < 1.0
//    Return asin(x) = sign(x) * ( Pi/2 - sqrt(R) * PolB(R))
//    Where R = 1 - |x|,
//          PolB(R) = B0 + B1*R + B2*R^2 +...+B12*R^12
//
//    sqrt(R) is approximated using the following sequence:
//        y0 = (1 + eps)/sqrt(R) - initial approximation by frsqrta,
//             |eps| < 2^(-8)
//        Then 3 iterations are used to refine the result:
//        H0 = 0.5*y0
//        S0 = R*y0
//
//        d0 = 0.5 - H0*S0
//        H1 = H0 + d0*H0
//        S1 = S0 + d0*S0
//
//        d1 = 0.5 - H1*S1
//        H2 = H1 + d0*H1
//        S2 = S1 + d0*S1
//
//        d2 = 0.5 - H2*S2
//        S3 = S3 + d2*S3
//
//        S3 approximates sqrt(R) with enough accuracy for this algorithm
//
//    So, the result should be reconstracted as follows:
//    asin(x) = sign(x) * (Pi/2 - S3*PolB(R))
//
//    But for optimization perposes the reconstruction step is slightly
//    changed:
//    asin(x) = sign(x)*(Pi/2 - PolB(R)*S2) + sign(x)*d2*S2*PolB(R)
//
// 4. |x| = 1.0
//    Return asin(x) = sign(x)*Pi/2
//
// 5. 1.0 < |x| <= +INF
//    A doman error occurs for arguments not in the range [-1,+1]
//
// 6. x = [S,Q]NaN
//    Return asin(x) = QNaN
//
// 7. x is denormal
//    Return asin(x) = x + x^3,
//
// 8. x is unnormal
//    Normalize input in f8 and return to the very beginning of the function
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input, output
// f6, f7, f9 -> f15, f32 -> f63

// General registers used:
// r3, r21 -> r31, r32 -> r38

// Predicate registers used:
// p0, p6 -> p14

//
// Assembly macros
//=========================================
// integer registers used
// scratch
rTblAddr                      = r3

rPiBy2Ptr                     = r21
rTmpPtr3                      = r22
rDenoBound                    = r23
rOne                          = r24
rAbsXBits                     = r25
rHalf                         = r26
r0625                         = r27
rSign                         = r28
rXBits                        = r29
rTmpPtr2                      = r30
rTmpPtr1                      = r31

// stacked
GR_SAVE_PFS                   = r32
GR_SAVE_B0                    = r33
GR_SAVE_GP                    = r34
GR_Parameter_X                = r35
GR_Parameter_Y                = r36
GR_Parameter_RESULT           = r37
GR_Parameter_TAG              = r38

// floating point registers used
FR_X                          = f10
FR_Y                          = f1
FR_RESULT                     = f8


// scratch
fXSqr                         = f6
fXCube                        = f7
fXQuadr                       = f9
f1pX                          = f10
f1mX                          = f11
f1pXRcp                       = f12
f1mXRcp                       = f13
fH                            = f14
fS                            = f15
// stacked
fA3                           = f32
fB1                           = f32
fA5                           = f33
fB2                           = f33
fA7                           = f34
fPiBy2                        = f34
fA9                           = f35
fA11                          = f36
fB10                          = f35
fB11                          = f36
fA13                          = f37
fA15                          = f38
fB4                           = f37
fB5                           = f38
fA17                          = f39
fA19                          = f40
fB6                           = f39
fB7                           = f40
fA21                          = f41
fA23                          = f42
fB3                           = f41
fB8                           = f42
fA25                          = f43
fA27                          = f44
fB9                           = f43
fB12                          = f44
fA29                          = f45
fA31                          = f46
fA33                          = f47
fA35                          = f48
fBaseP                        = f49
fB0                           = f50
fSignedS                      = f51
fD                            = f52
fHalf                         = f53
fR                            = f54
fCloseTo1Pol                  = f55
fSignX                        = f56
fDenoBound                    = f57
fNormX                        = f58
fX8                           = f59
fRSqr                         = f60
fRQuadr                       = f61
fR8                           = f62
fX16                          = f63
// Data tables
//==============================================================
RODATA
.align 16
LOCAL_OBJECT_START(asin_base_range_table)
// Ai: Polynomial coefficients for the asin(x), |x| < .625000
// Bi: Polynomial coefficients for the asin(x), |x| > .625000
data8 0xBFDAAB56C01AE468 //A29
data8 0x3FE1C470B76A5B2B //A31
data8 0xBFDC5FF82A0C4205 //A33
data8 0x3FC71FD88BFE93F0 //A35
data8 0xB504F333F9DE6487, 0x00003FFF //B0
data8 0xAAAAAAAAAAAAFC18, 0x00003FFC //A3
data8 0x3F9F1C71BC4A7823 //A9
data8 0x3F96E8BBAAB216B2 //A11
data8 0x3F91C4CA1F9F8A98 //A13
data8 0x3F8C9DDCEDEBE7A6 //A15
data8 0x3F877784442B1516 //A17
data8 0x3F859C0491802BA2 //A19
data8 0x9999999998C88B8F, 0x00003FFB //A5
data8 0x3F6BD7A9A660BF5E //A21
data8 0x3F9FC1659340419D //A23
data8 0xB6DB6DB798149BDF, 0x00003FFA //A7
data8 0xBFB3EF18964D3ED3 //A25
data8 0x3FCD285315542CF2 //A27
data8 0xF15BEEEFF7D2966A, 0x00003FFB //B1
data8 0x3EF0DDA376D10FB3 //B10
data8 0xBEB83CAFE05EBAC9 //B11
data8 0x3F65FFB67B513644 //B4
data8 0x3F5032FBB86A4501 //B5
data8 0x3F392162276C7CBA //B6
data8 0x3F2435949FD98BDF //B7
data8 0xD93923D7FA08341C, 0x00003FF9 //B2
data8 0x3F802995B6D90BDB //B3
data8 0x3F10DF86B341A63F //B8
data8 0xC90FDAA22168C235, 0x00003FFF // Pi/2
data8 0x3EFA3EBD6B0ECB9D //B9
data8 0x3EDE18BA080E9098 //B12
LOCAL_OBJECT_END(asin_base_range_table)


.section .text
GLOBAL_LIBM_ENTRY(asin)
asin_unnormal_back:
{ .mfi
      getf.d             rXBits = f8 // grab bits of input value
      // set p12 = 1 if x is a NaN, denormal, or zero
      fclass.m           p12, p0 = f8, 0xcf
      adds               rSign = 1, r0
}
{ .mfi
      addl               rTblAddr = @ltoff(asin_base_range_table),gp
      // 1 - x = 1 - |x| for positive x
      fms.s1             f1mX = f1, f1, f8
      addl               rHalf = 0xFFFE, r0 // exponent of 1/2
}
;;
{ .mfi
      addl               r0625 = 0x3FE4, r0 // high 16 bits of 0.625
      // set p8 = 1 if x < 0
      fcmp.lt.s1         p8, p9 = f8, f0
      shl                rSign = rSign, 63 // sign bit
}
{ .mfi
      // point to the beginning of the table
      ld8                rTblAddr = [rTblAddr]
      // 1 + x = 1 - |x| for negative x
      fma.s1             f1pX = f1, f1, f8
      adds               rOne = 0x3FF, r0
}
;;
{ .mfi
      andcm              rAbsXBits = rXBits, rSign // bits of |x|
      fmerge.s           fSignX = f8, f1 // signum(x)
      shl                r0625 = r0625, 48 // bits of DP representation of 0.625
}
{ .mfb
      setf.exp           fHalf = rHalf // load A2 to FP reg
      fma.s1             fXSqr = f8, f8, f0 // x^2
      // branch on special path if x is a NaN, denormal, or zero
(p12) br.cond.spnt       asin_special
}
;;
{ .mfi
      adds               rPiBy2Ptr = 272, rTblAddr
      nop.f              0
      shl                rOne = rOne, 52 // bits of 1.0
}
{ .mfi
      adds               rTmpPtr1 = 16, rTblAddr
      nop.f              0
      // set p6 = 1 if |x| < 0.625
      cmp.lt             p6, p7 = rAbsXBits, r0625
}
;;
{ .mfi
      ldfpd              fA29, fA31 = [rTblAddr] // A29, fA31
      // 1 - x = 1 - |x| for positive x
(p9)  fms.s1             fR = f1, f1, f8
      // point to coefficient of "near 1" polynomial
(p7)  adds               rTmpPtr2 = 176, rTblAddr
}
{ .mfi
      ldfpd              fA33, fA35 = [rTmpPtr1], 16 // A33, fA35
      // 1 + x = 1 - |x| for negative x
(p8)  fma.s1             fR = f1, f1, f8
(p6)  adds               rTmpPtr2 = 48, rTblAddr
}
;;
{ .mfi
      ldfe               fB0 = [rTmpPtr1], 16 // B0
      nop.f              0
      nop.i              0
}
{ .mib
      adds               rTmpPtr3 = 16, rTmpPtr2
      // set p10 = 1 if |x| = 1.0
      cmp.eq             p10, p0 = rAbsXBits, rOne
      // branch on special path for |x| = 1.0
(p10) br.cond.spnt       asin_abs_1
}
;;
{ .mfi
      ldfe               fA3 = [rTmpPtr2], 48 // A3 or B1
      nop.f              0
      adds               rTmpPtr1 = 64, rTmpPtr3
}
{ .mib
      ldfpd              fA9, fA11 = [rTmpPtr3], 16 // A9, A11 or B10, B11
      // set p11 = 1 if |x| > 1.0
      cmp.gt             p11, p0 = rAbsXBits, rOne
      // branch on special path for |x| > 1.0
(p11) br.cond.spnt       asin_abs_gt_1
}
;;
{ .mfi
      ldfpd              fA17, fA19 = [rTmpPtr2], 16 // A17, A19 or B6, B7
      // initial approximation of 1 / sqrt(1 - x)
      frsqrta.s1         f1mXRcp, p0 = f1mX
      nop.i              0
}
{ .mfi
      ldfpd              fA13, fA15 = [rTmpPtr3] // A13, A15 or B4, B5
      fma.s1             fXCube = fXSqr, f8, f0 // x^3
      nop.i              0
}
;;
{ .mfi
      ldfe               fA5 = [rTmpPtr2], 48 // A5 or B2
      // initial approximation of 1 / sqrt(1 + x)
      frsqrta.s1         f1pXRcp, p0 = f1pX
      nop.i              0
}
{ .mfi
      ldfpd              fA21, fA23 = [rTmpPtr1], 16 // A21, A23 or B3, B8
      fma.s1             fXQuadr = fXSqr, fXSqr, f0 // x^4
      nop.i              0
}
;;
{ .mfi
      ldfe               fA7 = [rTmpPtr1] // A7 or Pi/2
      fma.s1             fRSqr = fR, fR, f0 // R^2
      nop.i              0
}
{ .mfb
      ldfpd              fA25, fA27 = [rTmpPtr2] // A25, A27 or B9, B12
      nop.f              0
(p6)  br.cond.spnt       asin_base_range;
}
;;

{ .mfi
      nop.m              0
(p9)  fma.s1             fH = fHalf, f1mXRcp, f0 // H0 for x > 0
      nop.i              0
}
{ .mfi
      nop.m              0
(p9)  fma.s1             fS = f1mX, f1mXRcp, f0  // S0 for x > 0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
(p8)  fma.s1             fH = fHalf, f1pXRcp, f0 // H0 for x < 0
      nop.i              0
}
{ .mfi
      nop.m              0
(p8)  fma.s1             fS = f1pX, f1pXRcp, f0  // S0 for x > 0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fRQuadr = fRSqr, fRSqr, f0 // R^4
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB11 = fB11, fR, fB10
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB1 = fB1, fR, fB0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB5 = fB5, fR, fB4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB7 = fB7, fR, fB6
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB3 = fB3, fR, fB2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fnma.s1            fD = fH, fS, fHalf // d0 = 1/2 - H0*S0
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fR8 = fRQuadr, fRQuadr, f0 // R^4
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB9 = fB9, fR, fB8
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fB12 = fB12, fRSqr, fB11
      nop.i              0
}
{.mfi
      nop.m              0
      fma.s1             fB7 = fB7, fRSqr, fB5
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fB3 = fB3, fRSqr, fB1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fH = fH, fD, fH // H1 = H0 + H0*d0
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fS = fS, fD, fS // S1 = S0 + S0*d0
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fma.s1             fPiBy2 = fPiBy2, fSignX, f0 // signum(x)*Pi/2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fB12 = fB12, fRSqr, fB9
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fB7 = fB7, fRQuadr, fB3
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fnma.s1            fD = fH, fS, fHalf // d1 = 1/2 - H1*S1
      nop.i              0
}
{ .mfi
      nop.m              0
      fnma.s1            fSignedS = fSignX, fS, f0 // -signum(x)*S1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fCloseTo1Pol = fB12, fR8, fB7
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fH = fH, fD, fH // H2 = H1 + H1*d1
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fS = fS, fD, fS // S2 = S1 + S1*d1
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // -signum(x)* S2 = -signum(x)*(S1 + S1*d1)
      fma.s1             fSignedS = fSignedS, fD, fSignedS
      nop.i              0
}
;;
{.mfi
      nop.m              0
      fnma.s1            fD = fH, fS, fHalf // d2 = 1/2 - H2*S2
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // signum(x)*(Pi/2 - PolB*S2)
      fma.s1             fPiBy2 = fSignedS, fCloseTo1Pol, fPiBy2
      nop.i              0
}
{ .mfi
      nop.m              0
      // -signum(x)*PolB * S2
      fma.s1             fCloseTo1Pol = fSignedS, fCloseTo1Pol, f0
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for 0.625 <= |x| < 1
      fma.d.s0           f8 = fCloseTo1Pol, fD, fPiBy2
      // exit here for  0.625 <= |x| < 1
      br.ret.sptk        b0
}
;;


// here if |x| < 0.625
.align 32
asin_base_range:
{ .mfi
      nop.m              0
      fma.s1             fA33 = fA33, fXSqr, fA31
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA15 = fA15, fXSqr, fA13
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA29 = fA29, fXSqr, fA27
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fXSqr, fA23
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA21 = fA21, fXSqr, fA19
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, fXSqr, fA7
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA5 = fA5, fXSqr, fA3
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA35 = fA35, fXQuadr, fA33
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, fXQuadr, fA15
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fX8 = fXQuadr, fXQuadr, f0 // x^8
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA25 = fA25, fXQuadr, fA21
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA9 = fA9, fXQuadr, fA5
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA35 = fA35, fXQuadr, fA29
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, fXSqr, fA11
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fX16 = fX8, fX8, f0 // x^16
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fA35 = fA35, fX8, fA25
      nop.i              0
}
{ .mfi
      nop.m              0
      fma.s1             fA17 = fA17, fX8, fA9
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      fma.s1             fBaseP = fA35, fX16, fA17
      nop.i              0
}
;;
{ .mfb
      nop.m              0
      // final result for |x| < 0.625
      fma.d.s0           f8 = fBaseP, fXCube, f8
      // exit here for |x| < 0.625 path
      br.ret.sptk        b0
}
;;

// here if |x| = 1
// asin(x) = sign(x) * Pi/2
.align 32
asin_abs_1:
{ .mfi
      ldfe               fPiBy2 = [rPiBy2Ptr] // Pi/2
      nop.f              0
      nop.i              0
}
;;
{.mfb
      nop.m              0
      // result for |x| = 1.0
      fma.d.s0           f8 = fPiBy2, fSignX, f0
      // exit here for |x| = 1.0
      br.ret.sptk        b0
}
;;

// here if x is a NaN, denormal, or zero
.align 32
asin_special:
{ .mfi
      nop.m              0
      // set p12 = 1 if x is a NaN
      fclass.m           p12, p0 = f8, 0xc3
      nop.i              0
}
{ .mlx
      nop.m              0
      // smallest positive DP normalized number
      movl               rDenoBound = 0x0010000000000000
}
;;
{ .mfi
      nop.m              0
      // set p13 = 1 if x = 0.0
      fclass.m           p13, p0 = f8, 0x07
      nop.i              0
}
{ .mfi
      nop.m              0
      fnorm.s1           fNormX = f8
      nop.i              0
}
;;
{ .mfb
      // load smallest normal to FP reg
      setf.d             fDenoBound = rDenoBound
      // answer if x is a NaN
(p12) fma.d.s0           f8 = f8,f1,f0
      // exit here if x is a NaN
(p12) br.ret.spnt        b0
}
;;
{ .mfb
      nop.m              0
      nop.f              0
      // exit here if x = 0.0
(p13) br.ret.spnt        b0
}
;;
// if we still here then x is denormal or unnormal
{ .mfi
      nop.m              0
      // absolute value of normalized x
      fmerge.s           fNormX = f1, fNormX
      nop.i              0
}
;;
{ .mfi
      nop.m              0
      // set p14 = 1 if normalized x is greater than or
      // equal to the smallest denormalized value
      // So, if p14 is set to 1 it means that we deal with
      // unnormal rather than with "true" denormal
      fcmp.ge.s1         p14, p0 = fNormX, fDenoBound
      nop.i              0
}
;;
{ .mfi
      nop.m              0
(p14) fcmp.eq.s0         p6, p0 = f8, f0      // Set D flag if x unnormal
      nop.i              0
}
{ .mfb
      nop.m              0
      // normalize unnormal input
(p14) fnorm.s1           f8 = f8
      // return to the main path
(p14) br.cond.sptk       asin_unnormal_back
}
;;
// if we still here it means that input is "true" denormal
{ .mfb
      nop.m              0
      // final result if x is denormal
      fma.d.s0           f8 = f8, fXSqr, f8
      // exit here if x is denormal
      br.ret.sptk        b0
}
;;

// here if |x| > 1.0
// error handler should be called
.align 32
asin_abs_gt_1:
{ .mfi
      alloc              r32 = ar.pfs, 0, 3, 4, 0 // get some registers
      fmerge.s           FR_X = f8,f8
      nop.i              0
}
{ .mfb
      mov                GR_Parameter_TAG = 61 // error code
      frcpa.s0           FR_RESULT, p0 = f0,f0
      // call error handler routine
      br.cond.sptk       __libm_error_region
}
;;
GLOBAL_LIBM_END(asin)
libm_alias_double_other (asin, asin)



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
        stfd [GR_Parameter_X] = FR_X                  // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#         // Call error handling function
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
