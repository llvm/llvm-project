.file "erf.s"


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
// 08/15/01 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// double erf(double)
//
// Overview of operation
//==============================================================
// Background
//
//
// There are 9 paths:
// 1. x = +/-0.0
//    Return erf(x) = +/-0.0
//
// 2. 0.0 < |x| < 0.5
//    Return erf(x) = x *Pol9(x^2)
//
// 3. For several subranges of 0.5 <= |x| < 5.90625
//    Return erf(x) = sign(x)*Pol19(y),
//    where y = (|x|-b)/a, Pol19(y) = A0 + A1*y^1 + A2*y^2 + ... + A19*y^19
//
//    For each subrange there is particular set of coefficients.
//    Below is the list of subranges:
//    3.1 0.5 <= |x| < 1.0     b = a = 0.5
//    3.2 1.0 <= |x| < 2.0,    b = a = 1.0
//    3.3 2.0 <= |x| < 3.25    b = a = 2.0
//    3.4 4.0 <= |x| < 5.90625 b = 4.0, a = 2.0
//
// 4. 3.25 <= |x| < 4.0
//    Return erf(x) = sign(x)*Pol14(|x| - 3.25)
//
// 5. 5.90625 <= |x| < +INF
//    Return erf(x) = sign(x)*(1.0d - 2^(-63))
//
// 6. |x| = INF
//    Return erf(x) = sign(x) * 1.0
//
// 7. x = [S,Q]NaN
//    Return erf(x) = QNaN
//
// 8. x is positive denormal
//    Return erf(x) = A0*x - x^2,
//    where A0 = 2.0/sqrt(Pi)
//
// 9. x is negative denormal
//    Return erf(x) = A0*x + x^2,
//    where A0 = 2.0/sqrt(Pi)
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input, output
// f32 -> f63

// General registers used:
// r32 -> r48, r2, r3

// Predicate registers used:
// p0, p6 -> p15

// p6           to filter out case when x = denormal
// p7           to filter out case when x = [Q,S]NaN or +/-0,
//              used also to process denormals
// p8           to filter out case when 3.25 <= |x| < 4.0,
//              used also to process denormals
// p9           to filter out case when |x| = inf
// p10          to filter out case when |x| < 0.5
// p11          set when |x| < 3.25 or |x| > 4.0
// p12          to filter out case when |x| >= 5.90625
// p13          set if 4.0 <=|x| < 5.90625
// p14          set to 1 for positive x
// p15          set to 1 for negative x

// Assembly macros
//==============================================================
rDataPtr           = r2
rDataPtr1          = r3

rBias              = r33
rCoeffAddr3        = r34
rThreeAndQ         = r35
rCoeffAddr2        = r36
rMask              = r37
rArg               = r38
rSignBit           = r39
rAbsArg            = r40
rSaturation        = r41
rIndex             = r42
rCoeffAddr1        = r43
rCoeffAddr4        = r44
rShiftedArg        = r45
rShiftedArgMasked  = r46
rBiasedExpOf4      = r47
rShiftedAbsArg     = r48

//==============================================================
fA0                = f32
fA1                = f33
fA2                = f34
fA3                = f35
fA4                = f36
fA5                = f37
fA6                = f38
fA7                = f39
fA8                = f40
fA9                = f41
fA10               = f42
fA11               = f43
fA12               = f44
fA13               = f45
fA14               = f46
fA15               = f47
fA16               = f48
fA17               = f49
fA18               = f50
fA19               = f51
fArgSqr            = f52
fArgAbsNorm        = f53
fSignumX           = f54
fRes               = f55
fThreeAndQ         = f56
fArgAbs            = f57
fTSqr              = f58
fTQuadr            = f59
fTDeg3             = f60
fTDeg7             = f61
fArgAbsNormSgn     = f62
fTQuadrSgn         = f63

// Data tables
//==============================================================
RODATA

.align 64

LOCAL_OBJECT_START(erf_data)
// Coefficients ##0..15
// Polynomial coefficients for the erf(x), 0.5 <= |x| < 1.0
data8 0xB69AC40646D1F6C1, 0x00003FD2 //A19
data8 0x90AD48C0118FA10C, 0x00003FD7 //A18
data8 0x826FBAD055EA4AB8, 0x0000BFDB //A17
data8 0x8DAB171246CC2B89, 0x00003FDC //A16
data8 0xC0B1D6662F8A7564, 0x00003FDF //A15
data8 0xA46374AC35099BAF, 0x0000BFE1 //A14
data8 0xB2F230996346EF27, 0x0000BFE4 //A13
data8 0xCDEC50950FACE04A, 0x00003FE6 //A12
data8 0x826014649396E9D2, 0x00003FE9 //A11
data8 0xCDB787DC718B13F9, 0x0000BFEB //A10
data8 0x8E0B23C24EE0C8EE, 0x0000BFED //A9
data8 0xA49EA40A4E5A3F76, 0x00003FF0 //A8
data8 0xB11E30BE912617D3, 0x00003FF0 //A7
data8 0xCCF89D9351CE26E3, 0x0000BFF4 //A6
data8 0xEFF75AD1F0F22809, 0x00003FF2 //A5
data8 0xBB793EF404C09A22, 0x00003FF8 //A4
// Polynomial coefficients for the erf(x), 1.0 <= |x| < 2.0
data8 0xBAE93FF4174EA59B, 0x00003FE6 //A19
data8 0x8A0FD46092F95D44, 0x0000BFEA //A18
data8 0xA37B3242B7809E12, 0x00003FEC //A17
data8 0xA0330A5CD2E91689, 0x0000BFED //A16
data8 0x8E34A678F3497D17, 0x0000BFEC //A15
data8 0xAC185D45A2772384, 0x00003FEF //A14
data8 0xB0C11347CE7EEDE8, 0x00003FEF //A13
data8 0xD3330DC14EA0E4EB, 0x0000BFF2 //A12
data8 0xB4A6DFDE578A428F, 0x00003FF1 //A11
data8 0xA0B4034310D2D9CB, 0x00003FF5 //A10
data8 0xF71662D3132B7759, 0x0000BFF5 //A9
data8 0x9C88BF157695E9EC, 0x0000BFF7 //A8
data8 0xF84B80EFCA43895D, 0x00003FF8 //A7
data8 0x9722D22DA628A17B, 0x00003FF7 //A6
data8 0x8DB0A586F8F3381F, 0x0000BFFB //A5
data8 0x8DB0A5879F87E5BE, 0x00003FFB //A4
// Polynomial coefficients for the erf(x), 2.0 <= |x| < 3.25
data8 0x9C4AF1F3A4B21AFC, 0x00003FF6 //A19
data8 0x8D40D5D5DB741AB8, 0x0000BFF9 //A18
data8 0xDEBE7099E0A75BA4, 0x00003FFA //A17
data8 0xB99A33294D32429D, 0x0000BFFB //A16
data8 0x8109D9C7197BC7C9, 0x00003FFB //A15
data8 0xC30DE8E2EFC2D760, 0x00003FFA //A14
data8 0x80DDA28C5B35DC73, 0x0000BFFC //A13
data8 0x9BE4DE5095BACE0D, 0x00003FF9 //A12
data8 0xDA4092509EE7D111, 0x00003FFC //A11
data8 0x89D98C561B0C9040, 0x0000BFFD //A10
data8 0xD20B26EB2F0881D4, 0x0000BFF9 //A9
data8 0xD089C56948731561, 0x00003FFD //A8
data8 0xDD704DEFFB21B7E7, 0x0000BFFD //A7
data8 0xF0C9A6BBDE469115, 0x00003FF9 //A6
data8 0xD673A02CB5766633, 0x00003FFD //A5
data8 0x8D162CBAD8A12649, 0x0000BFFE //A4
// Polynomial coefficients for the erf(x), 4.0 <= |x| < 6.0
data8 0xD4428B75C6FE8FD1, 0x0000BFFC //A19
data8 0xF76BE1935675D5C8, 0x00003FFE //A18
data8 0xFD6BB3B14AA7A8E6, 0x0000BFFF //A17
data8 0x8BE8F573D348DDA4, 0x00004000 //A16
data8 0x81E91923A1030502, 0x0000BFFF //A15
data8 0xCE7FE87B26CFD286, 0x0000BFFE //A14
data8 0x84EF6B4E17404384, 0x00004000 //A13
data8 0x91FEF33015404991, 0x0000C000 //A12
data8 0xDEDF6A9370747E56, 0x00003FFF //A11
data8 0x8397E6FF56CDFD9D, 0x0000BFFF //A10
data8 0xFAD1CE912473937B, 0x00003FFD //A9
data8 0xC48C1EA8AAA624EA, 0x0000BFFC //A8
data8 0xFECAF0097ACF981B, 0x00003FFA //A7
data8 0x8829A394065E4B95, 0x0000BFF9 //A6
data8 0xED3003E477A53EE7, 0x00003FF6 //A5
data8 0xA4C07E9BB3FCB0F3, 0x0000BFF4 //A4
//
// Coefficients ##16..19
// Polynomial coefficients for the erf(x), 0.5 <= |x| < 1.0
data8 0x95FA98C337005D13, 0x0000BFF9 //A3
data8 0xE0F7E524D2808A97, 0x0000BFFB //A2
data8 0xE0F7E524D2808A98, 0x00003FFD //A1
data8 0x853F7AE0C76E915F, 0x00003FFE //A0
// Polynomial coefficients for the erf(x), 1.0 <= |x| < 2.0
data8 0x8DB0A587A96ABCF0, 0x00003FFC //A3
data8 0xD488F84B7DE18DA8, 0x0000BFFD //A2
data8 0xD488F84B7DE12E9C, 0x00003FFD //A1
data8 0xD7BB3D3A08445636, 0x00003FFE //A0
// Polynomial coefficients for the erf(x), 2.0 <= |x| < 3.25
data8 0xC58571D23D5C4B3A, 0x00003FFD //A3
data8 0xA94DCF467CD6AFF3, 0x0000BFFC //A2
data8 0xA94DCF467CD10A16, 0x00003FFA //A1
data8 0xFECD70A13CAF1997, 0x00003FFE //A0
// Polynomial coefficients for the erf(x), 4.0 <= |x| < 6.0
data8 0xB01D2B4F0D5AB8B0, 0x00003FF1 //A3
data8 0x8858A465CE594BD1, 0x0000BFEE //A2
data8 0x8858A447456DE61D, 0x00003FEA //A1
data8 0xFFFFFFBDC88BB107, 0x00003FFE //A0
// Polynomial coefficients for the erf(x), 0.0 <= |x| < 0.5
data8 0xBE839EDBB36C7FCE //A9
data8 0x3EBB7745A18DD242 //A8
data8 0xBF4C02DB238F2AFC //A5
data8 0x3F7565BCD0A9A3EA //A4
data8 0xC093A3581BCF3333, 0x0000BFFD //A1
data8 0xBEEF4BB82AD8AE22 //A7
data8 0x3F1F9A2A57A218CD //A6
data8 0xBF9B82CE3127F4E4 //A3
data8 0x3FBCE2F21A042B25 //A2
data8 0x906EBA8214DB688D, 0x00003FFF //A0
// 1.0 - 2^(-63)
data8 0xFFFFFFFFFFFFFFFF, 0x00003FFE
// Polynomial coefficients for the erf(x), 3.25 <= |x| < 4.0
data8 0x95E91576C7A12250, 0x00003FE7 //A14
data8 0x8E5E0D0E1F5D3CB5, 0x0000BFEA //A13
data8 0xED761DAFAF814DE9, 0x00003FEB //A12
data8 0xB3A77D921D0ACFC7, 0x0000BFEC //A11
data8 0xA662D27096B08D7C, 0x0000BFEC //A10
data8 0xDA0F410AE6233EA5, 0x00003FEF //A9
data8 0xAB4A8B16B3124327, 0x0000BFF1 //A8
data8 0xB241E236A5EDCED3, 0x00003FF2 //A7
data8 0x8A2A65BA1F551F77, 0x0000BFF3 //A6
data8 0xA4852D0B1D87000A, 0x00003FF3 //A5
data8 0x963EB00039489476, 0x0000BFF3 //A4
data8 0xCD5244FF4F7313A5, 0x00003FF2 //A3
data8 0xC6F1E695363BCB26, 0x0000BFF1 //A2
data8 0xF4DAF4680DA54C02, 0x00003FEF //A1
data8 0xFFFFB7CFB3F2ABBE, 0x00003FFE //A0
// A = 2.0/sqrt(Pi)
data8 0x906EBA8214DB688D, 0x00003FFF
LOCAL_OBJECT_END(erf_data)


.section .text
GLOBAL_LIBM_ENTRY(erf)

{ .mfi
      alloc          r32 = ar.pfs, 0, 17, 0, 0
      fmerge.se      fArgAbsNorm = f1, f8         // normalized x
      adds           rSignBit = 0x1, r0
}
{ .mfi
      addl           rDataPtr = @ltoff(erf_data), gp
      fma.s1         fArgSqr = f8, f8, f0         // x^2
      addl           rThreeAndQ = 0x400A0, r0     // shifted bits of 3.25
}
;;
{ .mfi
      getf.d         rArg = f8                    // x in GR
      fclass.m       p6,p0 = f8, 0x0b             // is x denormal ?
      shl            rThreeAndQ = rThreeAndQ, 44  // bits of 3.25
}
{ .mfi
      ld8            rDataPtr = [rDataPtr]
      nop.f          0
      addl           rBiasedExpOf4 = 0x40100, r0  // shifted bits of 4.0
}
;;
{ .mfi
      addl           rSaturation = 0x4017A, r0    // shifted bits of 5.90625
      fclass.m       p7,p0 = f8, 0xc7             // is x [S,Q]NaN or +/-0 ?
      shl            rSignBit = rSignBit, 63      // mask for sign bit
}
{ .mfi
      addl           rMask = 0x7FF00, r0          // Mask for index bits
      nop.f          0
      addl           rBias = 0x3FE00, r0          // bias of 0.5 << 8
}
;;
{ .mfi
      setf.d         fThreeAndQ = rThreeAndQ      // 3.25 if FP register
      fclass.m       p9,p0 = f8, 0x23             // is x +/- inf?
      shr.u          rShiftedArg = rArg, 44
}
{ .mfb
      andcm          rAbsArg = rArg, rSignBit     // |x| in GR
      nop.f          0
(p6)  br.cond.spnt   erf_denormal                 // branch out if x is denormal
}
;;
{ .mfi
      and            rShiftedArgMasked = rShiftedArg, rMask // bias of x << 8
      fmerge.s       fArgAbs = f1, f8             // |x|
      shr            rShiftedAbsArg = rAbsArg, 44
}
{ .mfb
      cmp.lt         p8, p11 = rThreeAndQ, rAbsArg // p8 = 1 if |x| >= 3.25
(p7)  fma.d.s0       f8 = f8,f1,f8                // NaN or +/-0
(p7)  br.ret.spnt    b0                           // exit for x = NaN or +/-0
}
;;
{ .mfi
      sub            rIndex = rShiftedArgMasked, rBias // index << 8
      nop.f          0
      cmp.lt         p10, p0 = rShiftedArgMasked, rBias // p10 = 1 if |x| < 0.5
}
{ .mfb
      // p8 = 1 if 3.25 <= |x| < 4.0
(p8)  cmp.lt         p8, p11 = rShiftedAbsArg, rBiasedExpOf4
      fms.s1         fArgAbsNorm = fArgAbsNorm, f1, f1
(p10) br.cond.spnt   erf_near_zero // branch out if |x| < 0.5
}
;;
.pred.rel "mutex", p8, p11
{ .mfi
(p8)  adds           rCoeffAddr1 = 1392, rDataPtr // coeff. for 3.25 <=|x|<4.0
(p9)  fmerge.s       f8 = f8,f1                   // +/- inf
      nop.i          0
}
{ .mfb
(p11) add            rCoeffAddr1 = rDataPtr, rIndex// coeff. ##0,2,..14
      nop.f          0
(p9)  br.ret.spnt    b0                            // exit for x = +/- inf
}
;;
{ .mfi
      adds           rCoeffAddr2 = 16, rCoeffAddr1
      fmerge.s       fSignumX = f8, f1            // signum(x)
      nop.i          0
}
{ .mfb
      cmp.lt         p12, p0 = rSaturation, rShiftedAbsArg // |x| > 5.90625?
      nop.f          0
(p12) br.cond.spnt   erf_saturation               // branch out if x |x| >= 6.0
}
;;
// Here if paths #3,4
// if path #4 we'll branch out after loading of 14 necessary coefficients
{.mfi
      ldfe           fA19 = [rCoeffAddr1], 32
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA18 = [rCoeffAddr2], 32
      nop.f          0
      adds           rCoeffAddr3 = 1024, rDataPtr
}
;;
{.mfi
      ldfe           fA17 = [rCoeffAddr1], 32
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA16 = [rCoeffAddr2], 32
      nop.f          0
      nop.i          0
}
;;
{.mfi
      ldfe           fA15 = [rCoeffAddr1], 32
      fma.s1         fTSqr = fArgAbsNorm, fArgAbsNorm, f0
      shr.u          rIndex = rIndex, 2
}
{.mfi
      ldfe           fA14 = [rCoeffAddr2], 32
      nop.f          0
      adds           rCoeffAddr4 = 16, r0
}
;;
{.mfi
      ldfe           fA13 = [rCoeffAddr1], 32
      nop.f          0
      // address of coefficients ##16..23
      add            rCoeffAddr3 = rCoeffAddr3, rIndex
}
{.mfi
      ldfe           fA12 = [rCoeffAddr2], 32
      nop.f          0
      cmp.lt         p15, p14 = rArg, r0
}
;;
{.mfi
      ldfe           fA11 = [rCoeffAddr1], 32
      nop.f          0
      add            rCoeffAddr4 = rCoeffAddr3, rCoeffAddr4
}
{.mfi
      ldfe           fA10 = [rCoeffAddr2], 32
      nop.f          0
      nop.i          0
}
;;
{.mfi
      ldfe           fA9 = [rCoeffAddr1], 32
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA8 = [rCoeffAddr2], 32
      nop.f          0
      nop.i          0
}
;;
{.mfi
      ldfe           fA7 = [rCoeffAddr1], 32
      fms.s1         fArgAbs = fArgAbs, f1, fThreeAndQ
      nop.i          0
}
{.mfb
      ldfe           fA6 = [rCoeffAddr2], 32
      nop.f          0
(p8)  br.cond.spnt   erf_3q_4 // branch out if  3.25 < |x| < 4.0
}
;;
{.mfi
      ldfe           fA5 = [rCoeffAddr1], 32
      fma.s1         fTDeg3 = fArgAbsNorm, fTSqr, f0
      nop.i          0
}
{.mfi
      ldfe           fA4 = [rCoeffAddr2], 32
      fma.s1         fTQuadr = fTSqr, fTSqr, f0
      nop.i          0
}
;;
// Path #3 Polynomial Pol19(y) computation; y = fArgAbsNorm
{.mfi
      ldfe           fA3 = [rCoeffAddr3], 32
      fma.s1         fArgAbsNormSgn = fArgAbsNorm, fSignumX, f0
      nop.i          0
}
{.mfi
      ldfe           fA2 = [rCoeffAddr4], 32
      nop.f          0
      nop.i          0
}
;;
{.mfi
      ldfe           fA1 = [rCoeffAddr3], 32
      fma.s1         fRes = fA19, fArgAbsNorm, fA18
      nop.i          0
}
{.mfi
      ldfe           fA0 = [rCoeffAddr4], 32
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA17 = fA17, fArgAbsNorm, fA16
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA15 = fA15, fArgAbsNorm, fA14
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fTDeg7 = fTDeg3, fTQuadr, f0
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA13 = fA13, fArgAbsNorm, fA12
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA11 = fA11, fArgAbsNorm, fA10
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA9 = fA9, fArgAbsNorm, fA8
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTSqr, fA17
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA7 = fA7, fArgAbsNorm, fA6
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA5 = fA5, fArgAbsNorm, f0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA15 = fA15, fTSqr, fA13
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA4 = fA4, fArgAbsNorm, fA3
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA2 = fA2, fArgAbsNorm, fA1
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA11 = fA11, fTSqr, fA9
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA7 = fA7, fTSqr, fA5
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTQuadr, fA15
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA4 = fA4, fTSqr, fA2
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTQuadr, fA11
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA4 = fA7, fTDeg3, fA4
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes,  fTDeg7, fA4
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      // result for negative argument
(p15) fms.d.s0       f8 = fRes, fArgAbsNormSgn, fA0
      nop.i          0
}
{ .mfb
      nop.m          0
      // result for positive argument
(p14) fma.d.s0       f8 = fRes, fArgAbsNormSgn, fA0
      br.ret.sptk    b0
}

// Here if  3.25 < |x| < 4.0
.align 32
erf_3q_4:
.pred.rel "mutex", p14, p15
{ .mfi
      ldfe           fA5 = [rCoeffAddr1], 32
      fma.s1         fTSqr = fArgAbs, fArgAbs, f0
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fRes = fA19, fArgAbs, fA18
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA17 = fA17, fArgAbs, fA16
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA15 = fA15, fArgAbs, fA14
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA13 = fA13, fArgAbs, fA12
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA11 = fA11, fArgAbs, fA10
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA9 = fA9, fArgAbs, fA8
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fArgAbsNormSgn = fArgAbs, fSignumX, f0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fTQuadr = fTSqr, fTSqr, f0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTSqr, fA17
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA15 = fA15, fTSqr, fA13
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA11 = fA11, fTSqr, fA9
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA7 = fA7, fArgAbs, fA6
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fTDeg7 = fTQuadr, fTSqr, f0
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTQuadr, fA15
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA11 = fA11, fTSqr, fA7
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTDeg7, fA11
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      // result for negative argument
(p15) fms.d.s0       f8 = fRes, fArgAbsNormSgn, fA5
      nop.i          0
}
{ .mfb
      nop.m          0
      // result for positive argument
(p14) fma.d.s0       f8 = fRes, fArgAbsNormSgn, fA5
      br.ret.sptk    b0
}
;;

// Here if |x| < 0.5
.align 32
erf_near_zero:
{ .mfi
      adds           rCoeffAddr1 = 1280, rDataPtr // address of A9
      fma.s1         fTSqr = fArgSqr, fArgSqr, f0 // x^4
      nop.i          0
}
{ .mfi
      adds           rCoeffAddr2 = 1328, rDataPtr // address of A7
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      ldfpd          fA9, fA8 = [rCoeffAddr1], 16
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfpd          fA7, fA6 = [rCoeffAddr2], 16
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      ldfpd          fA5, fA4 = [rCoeffAddr1], 16
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfpd          fA3, fA2 = [rCoeffAddr2], 16
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      ldfe           fA1 = [rCoeffAddr1]
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfe           fA0 = [rCoeffAddr2]
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fTQuadr = fTSqr, fTSqr, f0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fA9, fArgSqr, fA8
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA7 = fA7, fArgSqr, fA6
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA3 = fA3, fArgSqr, fA2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA5 = fA5, fArgSqr, fA4
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA1 = fA1, fArgSqr, fA0
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fTQuadrSgn = fTQuadr, f8, f0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTSqr, fA7
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA1 = fA3, fTSqr, fA1
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fRes = fRes, fTSqr, fA5
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA1 = fA1, f8, f0
      nop.i          0
}
;;
{ .mfb
      nop.m          0
      fma.d.s0       f8 = fRes, fTQuadrSgn, fA1 // x*Pol9(x^2)
      br.ret.sptk    b0                              // Exit for |x| < 0.5
};;

// Here if 5.90625 <= |x| < +inf
.align 32
erf_saturation:
{ .mfi
      adds           rDataPtr = 1376, rDataPtr     // address of A0
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      ldfe           fA0 = [rDataPtr]
      nop.f          0
      nop.i          0
}
;;
{ .mfb
      nop.m          0
      fma.d.s0       f8 = fA0, fSignumX, f0       // sign(x)*(1.0 - 2^(-63))
      // Exit for 5.90625 <= |x| < +inf
      br.ret.sptk    b0                          // Exit for 5.90625 <=|x|< +inf
}
;;

// Here if x is double precision denormal
.align 32
erf_denormal:
{ .mfi
      adds           rDataPtr = 1632, rDataPtr    // address of A0
      fclass.m       p7,p8 = f8, 0x0a             // is x -denormal ?
      nop.i          0
}
;;
{ .mfi
      ldfe           fA0 = [rDataPtr]             // A0
      nop.f          0
      nop.i          0
}
;;
{ .mfi
      nop.m          0
      fma.s1         fA0 = fA0,f8,f0              // A0*x
      nop.i          0
}
;;
{ .mfi
      nop.m          0
(p7)  fma.d.s0       f8 = f8,f8,fA0               // -denormal
      nop.i          0
}
{ .mfb
      nop.m          0
(p8)  fnma.d.s0      f8 = f8,f8,fA0               // +denormal
      br.ret.sptk    b0                           // Exit for denormal
}
;;

GLOBAL_LIBM_END(erf)
libm_alias_double_other (erf, erf)
