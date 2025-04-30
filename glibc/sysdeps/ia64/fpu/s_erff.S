.file "erff.s"


// Copyright (c) 2001 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2001 by the Intel Numerics Group, Intel Corporation
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
// 08/14/01 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// float erff(float)
//
// Overview of operation
//==============================================================
// Background
//
//
// There are 8 paths:
// 1. x = +/-0.0
//    Return erff(x) = +/-0.0
//
// 2. 0.0 < |x| < 0.125
//    Return erff(x) = x *Pol3(x^2),
//    where Pol3(x^2) = C3*x^6 + C2*x^4 + C1*x^2 + C0
//
// 3. 0.125 <= |x| < 4.0
//    Return erff(x) = sign(x)*PolD(x)*PolC(|x|) + sign(x)*PolA(|x|),
//    where sign(x)*PolD(x) = sign(x)*(|x|^7 + D2*x^6 + D1*|x|^5 + D0*x^4),
//          PolC(|x|) = B0*x^4 + C3*|x|^3 + C2*|x|^2 + C1*|x| + C0,
//          PolA(|x|) = A3|x|^3 + A2*x^2 + A1*|x| + A0
//
//    Actually range 0.125<=|x|< 4.0 is splitted to 5 subranges.
//    For each subrange there is particular set of coefficients.
//    Below is the list of subranges:
//    3.1 0.125 <= |x| < 0.25
//    3.2 0.25 <= |x| < 0.5
//    3.3 0.5 <= |x| < 1.0
//    3.4 1.0 <= |x| < 2.0
//    3.5 2.0 <= |x| < 4.0
//
// 4. 4.0 <= |x| < +INF
//    Return erff(x) = sign(x)*(1.0d - 2^(-52))
//
// 5. |x| = INF
//    Return erff(x) = sign(x) * 1.0
//
// 6. x = [S,Q]NaN
//    Return erff(x) = QNaN
//
// 7. x is positive denormal
//    Return erff(x) = C0*x - x^2,
//    where C0 = 2.0/sqrt(Pi)
//
// 8. x is negative denormal
//    Return erff(x) = C0*x + x^2,
//    where C0 = 2.0/sqrt(Pi)
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f32 -> f59

// General registers used:
// r32 -> r45, r2, r3

// Predicate registers used:
// p0, p6 -> p12, p14, p15

// p6           to filter out case when x = [Q,S]NaN or +/-0
// p7           to filter out case when x = denormal
// p8           set if |x| >= 0.3125, used also to process denormal input
// p9           to filter out case when |x| = inf
// p10          to filter out case when |x| < 0.125
// p11          to filter out case when 0.125 <= |x| < 4.0
// p12          to filter out case when |x| >= 4.0
// p14          set to 1 for positive x
// p15          set to 1 for negative x

// Assembly macros
//==============================================================
rDataPtr           = r2
rDataPtr1          = r3

rBias              = r33
rCoeffAddr3        = r34
rCoeffAddr1        = r35
rCoeffAddr2        = r36
rOffset2           = r37
rBias2             = r38
rMask              = r39
rArg               = r40
rBound             = r41
rSignBit           = r42
rAbsArg            = r43
rDataPtr2          = r44
rSaturation        = r45

//==============================================================
fA0                = f32
fA1                = f33
fA2                = f34
fA3                = f35
fC0                = f36
fC1                = f37
fC2                = f38
fC3                = f39
fD0                = f40
fD1                = f41
fD2                = f42
fB0                = f43
fArgSqr            = f44
fAbsArg            = f45
fSignumX           = f46
fArg4              = f47
fArg4Sgn           = f48
fArg3              = f49
fArg3Sgn           = f50
fArg7Sgn           = f51
fArg6Sgn           = f52
fPolC              = f53
fPolCTmp           = f54
fPolA              = f55
fPolATmp           = f56
fPolD              = f57
fPolDTmp           = f58
fArgSqrSgn         = f59

// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(erff_data)
// Polynomial coefficients for the erf(x), 0.125 <= |x| < 0.25
data8 0xBE4218BB56B49E66 // C0
data8 0x3F7AFB8315DA322B // C1
data8 0x3F615D6EBEE0CA32 // C2
data8 0xBF468D71CF4F0918 // C3
data8 0x40312115B0932F24 // D0
data8 0xC0160D6CD0991EA3 // D1
data8 0xBFE04A567A6DBE4A // D2
data8 0xBF4207BC640D1509 // B0
// Polynomial coefficients for the erf(x), 0.25 <= |x| < 0.5
data8 0x3F90849356383F58 // C0
data8 0x3F830BD5BA240F09 // C1
data8 0xBF3FA4970E2BCE23 // C2
data8 0xBF6061798E58D0FD // C3
data8 0xBF68C0D83DD22E02 // D0
data8 0x401C0A9EE4108F94 // D1
data8 0xC01056F9B5E387F5 // D2
data8 0x3F1C9744E36A5706 // B0
// Polynomial coefficients for the erf(x), 0.5 <= |x| < 1.0
data8 0x3F85F7D419A13DE3 // C0
data8 0x3F791A13FF66D45A // C1
data8 0x3F46B17B16B5929F // C2
data8 0xBF5124947A8BF45E // C3
data8 0x3FA1B3FD95EA9564 // D0
data8 0x40250CECD79A020A // D1
data8 0xC0190DC96FF66CCD // D2
data8 0x3F4401AE28BA4DD5 // B0
// Polynomial coefficients for the erf(x), 1.0 <= |x| < 2.0
data8 0xBF49E07E3584C3AE // C0
data8 0x3F3166621131445C // C1
data8 0xBF65B7FC1EAC2099 // C2
data8 0x3F508C6BD211D736 // C3
data8 0xC053FABD70601067 // D0
data8 0x404A06640EE87808 // D1
data8 0xC0283F30817A3F08 // D2
data8 0xBF2F6DBBF4D6257F // B0
// Polynomial coefficients for the erf(x), 2.0 <= |x| < 4.0
data8 0xBF849855D67E9407 // C0
data8 0x3F5ECA5FEC01C70C // C1
data8 0xBF483110C30FABA4 // C2
data8 0x3F1618DA72860403 // C3
data8 0xC08A5C9D5FE8B9F6 // D0
data8 0x406EFF5F088CEC4B // D1
data8 0xC03A5743DF38FDE0 // D2
data8 0xBEE397A9FA5686A2 // B0
// Polynomial coefficients for the erf(x), -0.125 < x < 0.125
data8 0x3FF20DD7504270CB // C0
data8 0xBFD8127465AFE719 // C1
data8 0x3FBCE2D77791DD77 // C2
data8 0xBF9B582755CDF345 // C3
// Polynomial coefficients for the erf(x), 0.125 <= |x| < 0.25
data8 0xBD54E7E451AF0E36 // A0
data8 0x3FF20DD75043FE20 // A1
data8 0xBE05680ACF8280E4 // A2
data8 0xBFD812745E92C3D3 // A3
// Polynomial coefficients for the erf(x), 0.25 <= |x| < 0.5
data8 0xBE1ACEC2859CB55F // A0
data8 0x3FF20DD75E8D2B64 // A1
data8 0xBEABC6A83208FCFC // A2
data8 0xBFD81253E42E7B99 // A3
// Polynomial coefficients for the erf(x), 0.5 <= |x| < 1.0
data8 0x3EABD5A2482B4979 // A0
data8 0x3FF20DCAA52085D5 // A1
data8 0x3F13A994A348795B // A2
data8 0xBFD8167B2DFCDE44 // A3
// Polynomial coefficients for the erf(x), 1.0 <= |x| < 2.0
data8 0xBF5BA377DDAB4E17 // A0
data8 0x3FF2397F1D8FC0ED // A1
data8 0xBF9945BFC1915C21 // A2
data8 0xBFD747AAABB690D8 // A3
// Polynomial coefficients for the erf(x), 2.0 <= |x| < 4.0
data8 0x3FF0E2920E0391AF // A0
data8 0xC00D249D1A95A5AE // A1
data8 0x40233905061C3803 // A2
data8 0xC027560B851F7690 // A3
//
data8 0x3FEFFFFFFFFFFFFF // 1.0 - epsilon
data8 0x3FF20DD750429B6D // C0 = 2.0/sqrt(Pi)
LOCAL_OBJECT_END(erff_data)


.section .text
GLOBAL_LIBM_ENTRY(erff)

{ .mfi
      alloc          r32 = ar.pfs, 0, 14, 0, 0
      fmerge.s       fAbsArg = f1, f8             // |x|
      addl           rMask = 0x806, r0
}
{ .mfi
      addl           rDataPtr = @ltoff(erff_data), gp
      fma.s1         fArgSqr = f8, f8, f0         // x^2
      adds           rSignBit = 0x1, r0
}
;;

{ .mfi
      getf.s         rArg = f8                    // x in GR
      fclass.m       p7,p0 = f8, 0x0b             // is x denormal ?
      // sign bit and 2 most bits in significand
      shl            rMask = rMask, 20
}
{ .mfi
      ld8            rDataPtr = [rDataPtr]
      nop.f          0
      adds           rBias2 = 0x1F0, r0
}
;;

{ .mfi
      nop.m          0
      fmerge.s       fSignumX = f8, f1            // signum(x)
      shl            rSignBit = rSignBit, 31      // mask for sign bit
}
{ .mfi
      adds           rBound = 0x3E0, r0
      nop.f          0
      adds           rSaturation = 0x408, r0
}
;;

{ .mfi
      andcm          rOffset2 = rArg, rMask
      fclass.m       p6,p0 = f8, 0xc7             // is x [S,Q]NaN or +/-0 ?
      shl            rBound = rBound, 20          // 0.125f in GR
}
{ .mfb
      andcm          rAbsArg = rArg, rSignBit     // |x| in GR
      nop.f          0
(p7)  br.cond.spnt   erff_denormal               // branch out if x is denormal
}
;;

{ .mfi
      adds           rCoeffAddr2 = 352, rDataPtr
      fclass.m       p9,p0 = f8, 0x23            // is x +/- inf?
      shr            rOffset2 = rOffset2, 21
}
{ .mfi
      cmp.lt         p10, p8 = rAbsArg, rBound   // |x| < 0.125?
      nop.f          0
      adds           rCoeffAddr3 = 16, rDataPtr
}
;;

{ .mfi
(p8)  sub            rBias = rOffset2, rBias2
      fma.s1         fArg4 = fArgSqr, fArgSqr, f0 // x^4
      shl            rSaturation = rSaturation, 20// 4.0 in GR (saturation bound)
}
{ .mfb
(p10) adds           rBias = 0x14, r0
(p6)  fma.s.s0       f8 = f8,f1,f8                // NaN or +/-0
(p6)  br.ret.spnt    b0                           // exit for x = NaN or +/-0
}
;;

{ .mfi
      shladd         rCoeffAddr1 = rBias, 4, rDataPtr
      fma.s1         fArg3Sgn = fArgSqr, f8, f0  // sign(x)*|x|^3
      // is |x| < 4.0?
      cmp.lt         p11, p12 = rAbsArg, rSaturation
}
{ .mfi
      shladd         rCoeffAddr3 = rBias, 4, rCoeffAddr3
      fma.s1         fArg3 = fArgSqr, fAbsArg, f0 // |x|^3
      shladd         rCoeffAddr2 = rBias, 3, rCoeffAddr2
}
;;

{ .mfi
(p11) ldfpd          fC0, fC1 = [rCoeffAddr1]
(p9)  fmerge.s       f8 = f8,f1                   // +/- inf
(p12) adds           rDataPtr = 512, rDataPtr
}
{ .mfb
(p11) ldfpd          fC2, fC3 = [rCoeffAddr3], 16
      nop.f          0
(p9)  br.ret.spnt    b0                           // exit for x = +/- inf
}
;;

{ .mfi
(p11) ldfpd          fA0, fA1 = [rCoeffAddr2], 16
      nop.f          0
      nop.i          0
}
{ .mfi
      add            rCoeffAddr1 = 48, rCoeffAddr1
      nop.f          0
      nop.i          0
}
;;

{ .mfi
(p11) ldfpd          fD0, fD1 = [rCoeffAddr3]
      nop.f          0
      nop.i          0
}
{ .mfb
(p11) ldfpd          fD2, fB0 = [rCoeffAddr1]
      // sign(x)*|x|^2
      fma.s1         fArgSqrSgn = fArgSqr, fSignumX, f0
(p10) br.cond.spnt   erff_near_zero
}
;;

{ .mfi
(p11) ldfpd          fA2, fA3 = [rCoeffAddr2], 16
      fcmp.lt.s1     p15, p14 = f8,f0
      nop.i          0
}
{ .mfb
(p12) ldfd           fA0 = [rDataPtr]
      fma.s1         fArg4Sgn = fArg4, fSignumX, f0 // sign(x)*|x|^4
(p12) br.cond.spnt   erff_saturation
}
;;
{ .mfi
      nop.m          0
      fma.s1         fArg7Sgn = fArg4, fArg3Sgn, f0  // sign(x)*|x|^7
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fArg6Sgn = fArg3, fArg3Sgn, f0  // sign(x)*|x|^6
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         fPolC = fC3, fAbsArg, fC2    // C3*|x| + C2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fPolCTmp = fC1, fAbsArg, fC0 // C1*|x| + C0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fPolA = fA1, fAbsArg, fA0    // A1*|x| + A0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         fPolD = fD1, fAbsArg, fD0    // D1*|x| + D0
      nop.i          0
}
{ .mfi
      nop.m          0
      // sign(x)*(|x|^7 + D2*x^6)
      fma.s1         fPolDTmp = fArg6Sgn, fD2, fArg7Sgn
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fPolATmp = fA3, fAbsArg, fA2  // A3*|x| + A2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fB0 = fB0, fArg4, f0          // B0*x^4
      nop.i          0
};;

{ .mfi
      nop.m          0
      // C3*|x|^3 + C2*x^2 + C1*|x| + C0
      fma.s1         fPolC = fPolC, fArgSqr, fPolCTmp
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      // PolD = sign(x)*(|x|^7 + D2*x^6 + D1*|x|^5 + D0*x^4)
      fma.d.s1       fPolD = fPolD, fArg4Sgn, fPolDTmp
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      // PolA = A3|x|^3 + A2*x^2 + A1*|x| + A0
      fma.d.s1       fPolA = fPolATmp, fArgSqr, fPolA
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      // PolC = B0*x^4 + C3*|x|^3 + C2*|x|^2 + C1*|x| + C0
      fma.d.s1       fPolC = fPolC, f1, fB0
      nop.i          0
}
;;

{ .mfi
      nop.m          0
(p14) fma.s.s0       f8 = fPolC, fPolD, fPolA     // for positive x
      nop.i          0
}
{ .mfb
      nop.m          0
(p15) fms.s.s0       f8 = fPolC, fPolD, fPolA     // for negative x
      br.ret.sptk    b0                           // Exit for 0.125 <=|x|< 4.0
};;


// Here if |x| < 0.125
erff_near_zero:
{ .mfi
      nop.m          0
      fma.s1         fPolC = fC3, fArgSqr, fC2    // C3*x^2 + C2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fPolCTmp = fC1, fArgSqr, fC0  // C1*x^2 + C0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fPolC = fPolC, fArg4, fPolCTmp // C3*x^6 + C2*x^4 + C1*x^2 + C0
      nop.i          0
};;

{ .mfb
      nop.m          0
      // x*(C3*x^6 + C2*x^4 + C1*x^2 + C0)
      fma.s.s0       f8 = fPolC, f8, f0
      br.ret.sptk    b0                           // Exit for |x| < 0.125
};;

// Here if 4.0 <= |x| < +inf
erff_saturation:
{ .mfb
      nop.m          0
      fma.s.s0       f8 = fA0, fSignumX, f0       // sign(x)*(1.0d - 2^(-52))
      // Exit for 4.0 <= |x| < +inf
      br.ret.sptk    b0                           // Exit for 4.0 <=|x|< +inf
}
;;

// Here if x is single precision denormal
erff_denormal:
{ .mfi
      adds           rDataPtr = 520, rDataPtr     // address of C0
      fclass.m       p7,p8 = f8, 0x0a             // is x -denormal ?
      nop.i          0
}
;;
{ .mfi
      ldfd           fC0 = [rDataPtr]             // C0
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fC0 = fC0,f8,f0              // C0*x
      nop.i          0
}
;;
{ .mfi
      nop.m          0
(p7)  fma.s.s0       f8 = f8,f8,fC0               // -denormal
      nop.i          0
}
{ .mfb
      nop.m          0
(p8)  fnma.s.s0      f8 = f8,f8,fC0               // +denormal
      br.ret.sptk    b0                           // Exit for denormal
}
;;

GLOBAL_LIBM_END(erff)
libm_alias_float_other (erf, erf)
