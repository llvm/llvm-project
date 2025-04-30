.file "tanhf.s"


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
// 05/30/01 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// float tanhf(float)
//
// Overview of operation
//==============================================================
// Background
//
//
// There are 9 paths:
// 1. x = +/-0.0
//    Return tanhf(x) = +/-0.0
//
// 2. 0.0 < |x| < 0.3125
//    Return tanhf(x) = x + x^3*Pol3(x^2),
//    where Pol3(x^2) = C3*x^6 + C2*x^4 + C1*x^2 + C0
//
// 3. 0.3125 <= |x| < 8.0
//    Return tanhf(x) = sign(x)*PolD(x)*PolC(|x|) + sign(x)*PolA(|x|),
//    where sign(x)*PolD(x) = sign(x)*(|x|^7 + D2*x^6 + D1*|x|^5 + D0*x^4),
//          PolC(|x|) = B0*x^4 + C3*|x|^3 + C2*|x|^2 + C1*|x| + C0,
//          PolA(|x|) = A3|x|^3 + A2*x^2 + A1*|x| + A0
//
//    Actually range 0.3125<=|x|< 8.0 is split to 5 subranges.
//    For each subrange there is particular set of coefficients.
//    Below is the list of subranges:
//    3.1 0.3125 <= |x| < 0.5
//    3.2 0.5 <= |x| < 1.0
//    3.3 1.0 <= |x| < 2.0
//    3.4 2.0 <= |x| < 4.0
//    3.5 4.0 <= |x| < 8.0
//
// 4. 8.0 <= |x| < 9.125
//    Return tanhf(x) = sign(x)*(A3|x|^3 + A2*x^2 + A1*|x| + A0)
//
// 5. 9.125 <= |x| < +INF
//    Return tanhf(x) = sign(x)*(1.0d - 2^(-52))
//
// 6. |x| = INF
//    Return tanhf(x) = sign(x) * 1.0
//
// 7. x = [S,Q]NaN
//    Return tanhf(x) = QNaN
//
// 8. x is positive denormal
//    Return tanhf(x) = x - x^2
//
// 9. x is negative denormal
//    Return tanhf(x) = x + x^2
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f32 -> f59

// General registers used:
// r32 -> r46, r2, r3

// Predicate registers used:
// p0, p6 -> p15

// p6           to filter out case when x = [Q,S]NaN or +/-0
// p7           to filter out case when x = denormal
// p8           set if |x| >= 0.3125, used also to process denormal input
// p9           to filter out case when |x| = inf
// p10          to filter out case when |x| < 0.3125
// p11          to filter out case when 0.3125 <= |x| < 9.125
// p12          to filter out case when |x| >= 9.125
// p13          to filter out case when 8.0 <= |x| < 9.125
// p14          set to 1 for positive x
// p15          set to 1 for negative x

// Assembly macros
//==============================================================
rDataPtr           = r2
rDataPtr1          = r3

rBias              = r33
rCoeffAddr3        = r34
rNearSaturation    = r35
rCoeffAddr1        = r36
rCoeffAddr2        = r37
rOffset2           = r38
rBias2             = r39
rMask              = r40
rArg               = r41
rBound             = r42
rSignBit           = r43
rAbsArg            = r44
rDataPtr2          = r45
rSaturation        = r46

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

LOCAL_OBJECT_START(tanhf_data)
// Polynomial coefficients for the tanh(x), 0.3125 <= |x| < 0.5
data8 0x3F9BEEDFDD177D7B // C0
data8 0x3F970D10C7F32458 // C1
data8 0x3F766D6B051F3A38 // C2
data8 0xBF732F2001B23402 // C3
data8 0xBF854BE1CE1ED499 // D0
data8 0x4013C944F3999A16 // D1
data8 0xC01106C6975222C0 // D2
data8 0x3F783D5ACCF9EBE8 // B0
// Polynomial coefficients for the tanh(x), 0.5 <= |x| < 1.0
data8 0xBF5D631440786869 // C0
data8 0xBF575D79A0D52069 // C1
data8 0xBF7E2237B7EFC705 // C2
data8 0x3F6A7ACBC273041F // C3
data8 0xC040E32EA52D91EB // D0
data8 0x403D19463E5DB4D7 // D1
data8 0xC02216F61F759F39 // D2
data8 0xBF55B4EA0B844BE7 // B0
// Polynomial coefficients for the tanh(x), 1.0 <= |x| < 2.0
data8 0x3F8637DBE5B3E690 // C0
data8 0xBF7F7FEC158C07F5 // C1
data8 0x3F711C586706838A // C2
data8 0xBF50EF7EF605554E // C3
data8 0xC054D45448354E25 // D0
data8 0x404ADFEEA282E730 // D1
data8 0xC028AEE456D59549 // D2
data8 0x3F25232D1BED59A8 // B0
// Polynomial coefficients for the tanh(x), 2.0 <= |x| < 4.0
data8 0xBF52602285F2D06C // C0
data8 0x3F2E57C298FFE1E0 // C1
data8 0xBF15ED575DB3C811 // C2
data8 0x3EE428878A08525C // C3
data8 0xC0895A26849039C1 // D0
data8 0x406E3C60BBFBB575 // D1
data8 0xC03A06F62867C75A // D2
data8 0xBEB114C70F1C723E // B0
// Polynomial coefficients for the tanh(x), 4.0 <= |x| < 8.0
data8 0x3EF4B22BD17039A3 // C0
data8 0xBEB704ADC040C57F // C1
data8 0x3E937A98288AFE1A // C2
data8 0xBE4F33B2C9FFE7E7 // C3
data8 0xC0BE48CFADE2431E // D0
data8 0x4090E74249760FDD // D1
data8 0xC04B6F537FCF2F1E // D2
data8 0x3E0DCD879C91ADEA // B0
// Polynomial coefficients for the tanh(x), -0.3125 < x < 0.3125
data8 0xBFD555551E8245B7 // A0
data8 0x3FC110E63F52E689 // A1
data8 0xBFAB8CD6A5B7BAFA // A2
data8 0x3F945D467FCEB553 // A3
// Polynomial coefficients for the tanh(x), 0.3125 <= |x| < 0.5
data8 0xBE3DCC92FCAECBB6 // A0
data8 0x3FF0000043B7D267 // A1
data8 0xBED18BF28ACFC4B1 // A2
data8 0xBFD554A56F82837E // A3
// Polynomial coefficients for the tanh(x), 0.5 <= |x| < 1.0
data8 0x3EFD6054758539F9 // A0
data8 0x3FEFFBFC77198EBE // A1
data8 0x3F700327CA98D237 // A2
data8 0xBFD68955F5BB2FA1 // A3
// Polynomial coefficients for the tanh(x), 1.0 <= |x| < 2.0
data8 0xBF71A53F229DF01B // A0
data8 0x3FF0AECFD730DE50 // A1
data8 0xBFC882F88E5DF3BA // A2
data8 0x3FC6EDF212CA2A8D // A3
// Polynomial coefficients for the tanh(x), 2.0 <= |x| < 4.0
data8 0xBFAF0B712E9EDA47 // A0
data8 0x3FF1C208080BEA64 // A1
data8 0x3FC3D29B20C8946E // A2
data8 0xBFF04514ED900A6A // A3
// Polynomial coefficients for the tanh(x), 4.0 <= |x| < 8.0
data8 0xBFB1DEA49A831CBC // A0
data8 0x3FFA729FC7085674 // A1
data8 0xBFF2F44D923A8FA4 // A2
data8 0x3FE092FC5712227E // A3
// Polynomial coefficients for the tanh(x), 8.0 <= |x| <= 9.125
data8 0x3FEFFF5769EE3041 // A0
data8 0x3EFBBF148D850891 // A1
data8 0xBEC86BCEF0F5C2FE // A2
data8 0x3E7CBA4F3A885A5C // A3
//
data8 0x3FEFFFFFFFFFFFFF // 1.0 - epsilon
LOCAL_OBJECT_END(tanhf_data)

.section .text
GLOBAL_LIBM_ENTRY(tanhf)

{ .mfi
      alloc          r32 = ar.pfs, 1, 14, 0, 0
      fmerge.s       fAbsArg = f1, f8             // |x|
      addl           rMask = 0x806, r0
}
{ .mfi
      addl           rDataPtr = @ltoff(tanhf_data), gp
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
      adds           rBias2 = 0x1F4, r0
}
;;

{ .mfi
      adds           rNearSaturation = 0x14, r0
      fmerge.s       fSignumX = f8, f1            // signum(x)
      shl            rSignBit = rSignBit, 31      // mask for sign bit
}
{ .mfi
      adds           rBound = 0x3EA, r0
      nop.f          0
      addl           rSaturation = 0x4112, r0
}
;;

{ .mfi
      andcm          rOffset2 = rArg, rMask
      fclass.m       p6,p0 = f8, 0xc7             // is x [S,Q]NaN or +/-0 ?
      shl            rBound = rBound, 20          // 1.0f in GR
}
{ .mfb
      andcm          rAbsArg = rArg, rSignBit     // |x| in GR
      nop.f          0
(p7)  br.cond.spnt   tanhf_denormal               // branch out if x is denormal
}
;;

{ .mfi
      adds           rCoeffAddr2 = 352, rDataPtr
      fclass.m       p9,p0 = f8, 0x23            // is x +/- inf?
      shr            rOffset2 = rOffset2, 21
}
{ .mfi
      cmp.lt         p10, p8 = rAbsArg, rBound   // |x| < 0.3125?
      nop.f          0
      adds           rCoeffAddr3 = 16, rDataPtr
}
;;

{ .mfi
(p8)  sub            rBias = rOffset2, rBias2
      fma.s1         fArg4 = fArgSqr, fArgSqr, f0 // x^4
      shl            rSaturation = rSaturation, 16
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
      // is |x| < 9.125?
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
(p12) adds           rDataPtr = 544, rDataPtr
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
(p8)  cmp.eq.unc     p13, p0 = rBias, rNearSaturation
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
(p10) br.cond.spnt   tanhf_near_zero
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
(p12) br.cond.spnt   tanhf_saturation
}
;;
{ .mfi
      nop.m          0
      fma.s1         fArg7Sgn = fArg4, fArg3Sgn, f0  // sign(x)*|x|^7
      nop.i          0
}
{ .mfb
      nop.m          0
      fma.s1         fArg6Sgn = fArg3, fArg3Sgn, f0  // sign(x)*|x|^6
(p13) br.cond.spnt   tanhf_close_to_saturation
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
      br.ret.sptk    b0                           // Exit for 0.3125 <=|x|< 8.0
};;


// Here if |x| < 0.3125
tanhf_near_zero:
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
      // x + x^3*(C3*x^6 + C2*x^4 + C1*x^2 + C0)
      fma.s.s0       f8 = fPolC, fArg3Sgn, f8
      br.ret.sptk    b0                           // Exit for |x| < 0.3125
};;

// Here if 9.125 <= |x| < +inf
tanhf_saturation:
{ .mfb
      nop.m          0
      fma.s.s0       f8 = fA0, fSignumX, f0       // sign(x)*(1.0d - 2^(-52))
      // Exit for 9.125 <= |x| < +inf
      br.ret.sptk    b0                           // Exit for 9.125 <=|x|< +inf
}
;;

// Here if  8.0 <= |x| < 9.125
tanhf_close_to_saturation:
{ .mfi
      nop.m          0
      fma.s1         fPolATmp = fA1, fAbsArg, fA0 // A1*|x| + A0
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fPolA = fA3, fAbsArg, fA2    // A3*|x| + A2
      nop.i          0
}
;;

.pred.rel "mutex", p14, p15
{ .mfi
      nop.m          0
      // for positive x
(p14) fma.s.s0       f8 = fPolA, fArgSqr, fPolATmp
      nop.i          0
}
{ .mfb
      nop.m          0
      // for negative x
(p15) fms.s.s0       f8 = fPolA, fArgSqrSgn, fPolATmp
      br.ret.sptk    b0                           // Exit for 8.0 <=|x|< 9.125
};;

// Here if x is single precision denormal
tanhf_denormal:
{ .mfi
      nop.m          0
      fclass.m       p7,p8 = f8, 0x0a             // is x -denormal ?
      nop.i          0
}
;;

{ .mfi
      nop.m          0
(p7)  fma.s.s0       f8 = f8,f8,f8                // -denormal
      nop.i          0
}
{ .mfb
      nop.m          0
(p8)  fnma.s.s0      f8 = f8,f8,f8                // +denormal
      br.ret.sptk    b0                           // Exit for denormal
}
;;

GLOBAL_LIBM_END(tanhf)
libm_alias_float_other (tanh, tanh)
