.file "powl.s"


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
//*********************************************************************
//
// Function:   powl(x,y), where
//                          y
//             powl(x,y) = x , for double extended precision x and y values
//
//*********************************************************************
//
// History:
// 02/02/00 (Hand Optimized)
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 01/22/01 Corrected results for powl(1,inf), powl(1,nan), and
//          powl(snan,0) to be 1 per C99, not nan.  Fixed many flag settings.
// 02/06/01 Call __libm_error support if over/underflow when y=2.
// 04/17/01 Support added for y close to 1 and x a non-special value.
//          Shared software under/overflow detection for all paths
// 02/07/02 Corrected sf3 setting to disable traps
// 05/13/02 Improved performance of all paths
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
// 04/17/03 Added missing mutex directive
// 10/13/03 Corrected .endp names to match .proc names
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers:
//                        f8  (Input x and Return Value)
//                        f9  (Input y)
//                        f10-f15,f32-f79
//
//    General Purpose Registers:
//                        Locals r14-24,r32-r65
//                        Parameters to __libm_error_support r62,r63,r64,r65
//
//    Predicate Registers: p6-p15
//
//*********************************************************************
//
//  Special Cases and IEEE special conditions:
//
//    Denormal fault raised on denormal inputs
//    Overflow exceptions raised when appropriate for pow
//    Underflow exceptions raised when appropriate for pow
//    (Error Handling Routine called for overflow and Underflow)
//    Inexact raised when appropriate by algorithm
//
//  1.  (anything) ** NatVal or (NatVal) ** anything  is NatVal
//  2.  X or Y unsupported or sNaN                    is qNaN/Invalid
//  3.  (anything) ** 0  is 1
//  4.  (anything) ** 1  is itself
//  5.  (anything except 1) ** qNAN is qNAN
//  6.  qNAN ** (anything except 0) is qNAN
//  7.  +-(|x| > 1) **  +INF is +INF
//  8.  +-(|x| > 1) **  -INF is +0
//  9.  +-(|x| < 1) **  +INF is +0
//  10. +-(|x| < 1) **  -INF is +INF
//  11. +-1         ** +-INF is +1
//  12. +0 ** (+anything except 0, NAN)               is +0
//  13. -0 ** (+anything except 0, NAN, odd integer)  is +0
//  14. +0 ** (-anything except 0, NAN)               is +INF/div_0
//  15. -0 ** (-anything except 0, NAN, odd integer)  is +INF/div_0
//  16. -0 ** (odd integer) = -( +0 ** (odd integer) )
//  17. +INF ** (+anything except 0,NAN) is +INF
//  18. +INF ** (-anything except 0,NAN) is +0
//  19. -INF ** (anything except NAN)  = -0 ** (-anything)
//  20. (-anything) ** (integer) is (-1)**(integer)*(+anything**integer)
//  21. (-anything except 0 and inf) ** (non-integer) is qNAN/Invalid
//  22. X or Y denorm/unorm and denorm/unorm operand trap is enabled,
//      generate denorm/unorm fault except if invalid or div_0 raised.
//
//*********************************************************************
//
//  Algorithm
//  =========
//
//  Special Cases
//
//    If Y = 2,    return X*X.
//    If Y = 0.5,  return sqrt(X).
//
//  Compute log(X) to extra precision.
//
//  ker_log_80( X, logX_hi, logX_lo, Safe );
//
//   ...logX_hi + logX_lo approximates log(X) to roughly 80
//   ...significant bits of accuracy.
//
//  Compute Y*log(X) to extra precision.
//
//    P_hi := Y * logX_hi
//    P_lo := Y * logX_hi - P_hi       ...using FMA
//    P_lo := Y * logX_lo + P_lo       ...using FMA
//
//  Compute exp(P_hi + P_lo)
//
//    Flag := 2;
//    Expo_Range := 2; (assuming double-extended power function)
//    ker_exp_64( P_hi, P_lo, Flag, Expo_Range,
//                Z_hi, Z_lo, scale, Safe )
//
//    scale := sgn * scale
//
//    If (Safe) then ...result will not over/underflow
//       return scale*Z_hi + (scale*Z_lo)
//       quickly
//    Else
//       take necessary precaution in computing
//       scale*Z_hi + (scale*Z_lo)
//       to set possible exceptions correctly.
//    End If
//
//  Case_Y_Special
//
//   ...Follow the order of the case checks
//
//   If Y is +-0, return +1 without raising any exception.
//   If Y is +1,  return X  without raising any exception.
//   If Y is qNaN, return Y without exception.
//   If X is qNaN, return X without exception.
//
//   At this point, X is real and Y is +-inf.
//   Thus |X| can only be 1, strictly bigger than 1, or
//   strictly less than 1.
//
//   If |X| < 1, then
//   return ( Y == +inf?  +0 : +inf )
//   elseif |X| > 1, then
//   return ( Y == +inf? +0 : +inf )
//   else
//   goto Case_Invalid
//
//  Case_X_Special
//
//   ...Follow the order of the case checks
//   ...Note that Y is real, finite, non-zero, and not +1.
//
//   If X is qNaN, return X without exception.
//
//   If X is +-0,
//   return ( Y > 0 ? +0 : +inf )
//
//   If X is +inf
//   return ( Y > 0 ? +inf : +0 )
//
//   If X is -inf
//   return -0 ** -Y
//   return ( Y > 0 ? +inf : +0 )
//
//  Case_Invalid
//
//   Return 0 * inf to generate a quiet NaN together
//   with an invalid exception.
//
//  Implementation
//  ==============
//
//   We describe the quick branch since this part is important
//   in reaching the normal case efficiently.
//
//  STAGE 1
//  -------
//   This stage contains two threads.
//
//   Stage1.Thread1
//
//     fclass.m   X_excep,  X_ok   = X, (NatVal or s/qNaN) or
//                              +-0, +-infinity
//
//     fclass.nm  X_unsupp, X_supp = X, (NatVal or s/qNaN) or
//                              +-(0, unnorm, norm, infinity)
//
//     X_norm := fnorm( X ) with traps disabled
//
//     If (X_excep)  goto Filtering (Step 2)
//     If (X_unsupp) goto Filtering (Step 2)
//
//     Stage1.Thread2
//     ..............
//
//     fclass.m   Y_excep,  Y_ok   = Y, (NatVal or s/qNaN) or
//                              +-0, +-infinity
//
//     fclass.nm  Y_unsupp, Y_supp = Y, (NatVal or s/qNaN) or
//                              +-(0, unnorm, norm, infinity)
//
//     Y_norm := fnorm( Y ) with traps disabled
//
//     If (Y_excep)  goto Filtering (Step 2)
//     If (Y_unsupp) goto Filtering (Step 2)
//
//
//  STAGE 2
//  -------
//  This stage contains two threads.
//
//     Stage2.Thread1
//     ..............
//
//     Set X_lt_0 if X < 0 (using fcmp)
//     sgn := +1.0
//     If (X_lt_0) goto Filtering (Step 2)
//
//     Stage2.Thread2
//     ..............
//
//     Set Y_is_1 if Y = +1 (using fcmp)
//     If (Y_is_1) goto Filtering (Step 2)
//
//   STAGE 3
//   -------
//   This stage contains two threads.
//
//
//   Stage3.Thread1
//   ..............
//
//     X := fnorm(X) in prevailing traps
//
//
//     Stage3.Thread2
//     ..............
//
//     Y := fnorm(Y) in prevailing traps
//
//   STAGE 4
//   -------
//
//   Go to Case_Normal.
//


// ************* DO NOT CHANGE ORDER OF THESE TABLES ********************

// double-extended 1/ln(2)
// 3fff b8aa 3b29 5c17 f0bb be87fed0691d3e88
// 3fff b8aa 3b29 5c17 f0bc
// For speed the significand will be loaded directly with a movl and setf.sig
//   and the exponent will be bias+63 instead of bias+0.  Thus subsequent
//   computations need to scale appropriately.
// The constant 2^12/ln(2) is needed for the computation of N.  This is also
//   obtained by scaling the computations.
//
// Two shifting constants are loaded directly with movl and setf.d.
//   1. RSHF_2TO51 = 1.1000..00 * 2^(63-12)
//        This constant is added to x*1/ln2 to shift the integer part of
//        x*2^12/ln2 into the rightmost bits of the significand.
//        The result of this fma is N_signif.
//   2. RSHF       = 1.1000..00 * 2^(63)
//        This constant is subtracted from N_signif * 2^(-51) to give
//        the integer part of N, N_fix, as a floating-point number.
//        The result of this fms is float_N.
RODATA

.align 16
// L_hi, L_lo
LOCAL_OBJECT_START(Constants_exp_64_Arg)
data8 0xB17217F400000000,0x00003FF2 // L_hi = hi part log(2)/2^12
data8 0xF473DE6AF278ECE6,0x00003FD4 // L_lo = lo part log(2)/2^12
LOCAL_OBJECT_END(Constants_exp_64_Arg)

LOCAL_OBJECT_START(Constants_exp_64_A)
// Reversed
data8 0xAAAAAAABB1B736A0,0x00003FFA
data8 0xAAAAAAAB90CD6327,0x00003FFC
data8 0xFFFFFFFFFFFFFFFF,0x00003FFD
LOCAL_OBJECT_END(Constants_exp_64_A)

LOCAL_OBJECT_START(Constants_exp_64_P)
// Reversed
data8 0xD00D6C8143914A8A,0x00003FF2
data8 0xB60BC4AC30304B30,0x00003FF5
data8 0x888888887474C518,0x00003FF8
data8 0xAAAAAAAA8DAE729D,0x00003FFA
data8 0xAAAAAAAAAAAAAF61,0x00003FFC
data8 0x80000000000004C7,0x00003FFE
LOCAL_OBJECT_END(Constants_exp_64_P)

LOCAL_OBJECT_START(Constants_exp_64_T1)
data4 0x3F800000,0x3F8164D2,0x3F82CD87,0x3F843A29
data4 0x3F85AAC3,0x3F871F62,0x3F88980F,0x3F8A14D5
data4 0x3F8B95C2,0x3F8D1ADF,0x3F8EA43A,0x3F9031DC
data4 0x3F91C3D3,0x3F935A2B,0x3F94F4F0,0x3F96942D
data4 0x3F9837F0,0x3F99E046,0x3F9B8D3A,0x3F9D3EDA
data4 0x3F9EF532,0x3FA0B051,0x3FA27043,0x3FA43516
data4 0x3FA5FED7,0x3FA7CD94,0x3FA9A15B,0x3FAB7A3A
data4 0x3FAD583F,0x3FAF3B79,0x3FB123F6,0x3FB311C4
data4 0x3FB504F3,0x3FB6FD92,0x3FB8FBAF,0x3FBAFF5B
data4 0x3FBD08A4,0x3FBF179A,0x3FC12C4D,0x3FC346CD
data4 0x3FC5672A,0x3FC78D75,0x3FC9B9BE,0x3FCBEC15
data4 0x3FCE248C,0x3FD06334,0x3FD2A81E,0x3FD4F35B
data4 0x3FD744FD,0x3FD99D16,0x3FDBFBB8,0x3FDE60F5
data4 0x3FE0CCDF,0x3FE33F89,0x3FE5B907,0x3FE8396A
data4 0x3FEAC0C7,0x3FED4F30,0x3FEFE4BA,0x3FF28177
data4 0x3FF5257D,0x3FF7D0DF,0x3FFA83B3,0x3FFD3E0C
LOCAL_OBJECT_END(Constants_exp_64_T1)

LOCAL_OBJECT_START(Constants_exp_64_T2)
data4 0x3F800000,0x3F80058C,0x3F800B18,0x3F8010A4
data4 0x3F801630,0x3F801BBD,0x3F80214A,0x3F8026D7
data4 0x3F802C64,0x3F8031F2,0x3F803780,0x3F803D0E
data4 0x3F80429C,0x3F80482B,0x3F804DB9,0x3F805349
data4 0x3F8058D8,0x3F805E67,0x3F8063F7,0x3F806987
data4 0x3F806F17,0x3F8074A8,0x3F807A39,0x3F807FCA
data4 0x3F80855B,0x3F808AEC,0x3F80907E,0x3F809610
data4 0x3F809BA2,0x3F80A135,0x3F80A6C7,0x3F80AC5A
data4 0x3F80B1ED,0x3F80B781,0x3F80BD14,0x3F80C2A8
data4 0x3F80C83C,0x3F80CDD1,0x3F80D365,0x3F80D8FA
data4 0x3F80DE8F,0x3F80E425,0x3F80E9BA,0x3F80EF50
data4 0x3F80F4E6,0x3F80FA7C,0x3F810013,0x3F8105AA
data4 0x3F810B41,0x3F8110D8,0x3F81166F,0x3F811C07
data4 0x3F81219F,0x3F812737,0x3F812CD0,0x3F813269
data4 0x3F813802,0x3F813D9B,0x3F814334,0x3F8148CE
data4 0x3F814E68,0x3F815402,0x3F81599C,0x3F815F37
LOCAL_OBJECT_END(Constants_exp_64_T2)

LOCAL_OBJECT_START(Constants_exp_64_W1)
data8 0x0000000000000000, 0xBE384454171EC4B4
data8 0xBE6947414AA72766, 0xBE5D32B6D42518F8
data8 0x3E68D96D3A319149, 0xBE68F4DA62415F36
data8 0xBE6DDA2FC9C86A3B, 0x3E6B2E50F49228FE
data8 0xBE49C0C21188B886, 0x3E64BFC21A4C2F1F
data8 0xBE6A2FBB2CB98B54, 0x3E5DC5DE9A55D329
data8 0x3E69649039A7AACE, 0x3E54728B5C66DBA5
data8 0xBE62B0DBBA1C7D7D, 0x3E576E0409F1AF5F
data8 0x3E6125001A0DD6A1, 0xBE66A419795FBDEF
data8 0xBE5CDE8CE1BD41FC, 0xBE621376EA54964F
data8 0x3E6370BE476E76EE, 0x3E390D1A3427EB92
data8 0x3E1336DE2BF82BF8, 0xBE5FF1CBD0F7BD9E
data8 0xBE60A3550CEB09DD, 0xBE5CA37E0980F30D
data8 0xBE5C541B4C082D25, 0xBE5BBECA3B467D29
data8 0xBE400D8AB9D946C5, 0xBE5E2A0807ED374A
data8 0xBE66CB28365C8B0A, 0x3E3AAD5BD3403BCA
data8 0x3E526055C7EA21E0, 0xBE442C75E72880D6
data8 0x3E58B2BB85222A43, 0xBE5AAB79522C42BF
data8 0xBE605CB4469DC2BC, 0xBE589FA7A48C40DC
data8 0xBE51C2141AA42614, 0xBE48D087C37293F4
data8 0x3E367A1CA2D673E0, 0xBE51BEBB114F7A38
data8 0xBE6348E5661A4B48, 0xBDF526431D3B9962
data8 0x3E3A3B5E35A78A53, 0xBE46C46C1CECD788
data8 0xBE60B7EC7857D689, 0xBE594D3DD14F1AD7
data8 0xBE4F9C304C9A8F60, 0xBE52187302DFF9D2
data8 0xBE5E4C8855E6D68F, 0xBE62140F667F3DC4
data8 0xBE36961B3BF88747, 0x3E602861C96EC6AA
data8 0xBE3B5151D57FD718, 0x3E561CD0FC4A627B
data8 0xBE3A5217CA913FEA, 0x3E40A3CC9A5D193A
data8 0xBE5AB71310A9C312, 0x3E4FDADBC5F57719
data8 0x3E361428DBDF59D5, 0x3E5DB5DB61B4180D
data8 0xBE42AD5F7408D856, 0x3E2A314831B2B707
LOCAL_OBJECT_END(Constants_exp_64_W1)

LOCAL_OBJECT_START(Constants_exp_64_W2)
data8 0x0000000000000000, 0xBE641F2537A3D7A2
data8 0xBE68DD57AD028C40, 0xBE5C77D8F212B1B6
data8 0x3E57878F1BA5B070, 0xBE55A36A2ECAE6FE
data8 0xBE620608569DFA3B, 0xBE53B50EA6D300A3
data8 0x3E5B5EF2223F8F2C, 0xBE56A0D9D6DE0DF4
data8 0xBE64EEF3EAE28F51, 0xBE5E5AE2367EA80B
data8 0x3E47CB1A5FCBC02D, 0xBE656BA09BDAFEB7
data8 0x3E6E70C6805AFEE7, 0xBE6E0509A3415EBA
data8 0xBE56856B49BFF529, 0x3E66DD3300508651
data8 0x3E51165FC114BC13, 0x3E53333DC453290F
data8 0x3E6A072B05539FDA, 0xBE47CD877C0A7696
data8 0xBE668BF4EB05C6D9, 0xBE67C3E36AE86C93
data8 0xBE533904D0B3E84B, 0x3E63E8D9556B53CE
data8 0x3E212C8963A98DC8, 0xBE33138F032A7A22
data8 0x3E530FA9BC584008, 0xBE6ADF82CCB93C97
data8 0x3E5F91138370EA39, 0x3E5443A4FB6A05D8
data8 0x3E63DACD181FEE7A, 0xBE62B29DF0F67DEC
data8 0x3E65C4833DDE6307, 0x3E5BF030D40A24C1
data8 0x3E658B8F14E437BE, 0xBE631C29ED98B6C7
data8 0x3E6335D204CF7C71, 0x3E529EEDE954A79D
data8 0x3E5D9257F64A2FB8, 0xBE6BED1B854ED06C
data8 0x3E5096F6D71405CB, 0xBE3D4893ACB9FDF5
data8 0xBDFEB15801B68349, 0x3E628D35C6A463B9
data8 0xBE559725ADE45917, 0xBE68C29C042FC476
data8 0xBE67593B01E511FA, 0xBE4A4313398801ED
data8 0x3E699571DA7C3300, 0x3E5349BE08062A9E
data8 0x3E5229C4755BB28E, 0x3E67E42677A1F80D
data8 0xBE52B33F6B69C352, 0xBE6B3550084DA57F
data8 0xBE6DB03FD1D09A20, 0xBE60CBC42161B2C1
data8 0x3E56ED9C78A2B771, 0xBE508E319D0FA795
data8 0xBE59482AFD1A54E9, 0xBE2A17CEB07FD23E
data8 0x3E68BF5C17365712, 0x3E3956F9B3785569
LOCAL_OBJECT_END(Constants_exp_64_W2)

LOCAL_OBJECT_START(Constants_log_80_P)
// P_8, P_7, ..., P_1
data8 0xCCCE8B883B1042BC, 0x0000BFFB // P_8
data8 0xE38997B7CADC2149, 0x00003FFB // P_7
data8 0xFFFFFFFEB1ACB090, 0x0000BFFB // P_6
data8 0x9249249806481C81, 0x00003FFC // P_5
data8 0x0000000000000000, 0x00000000 // Pad for bank conflicts
data8 0xAAAAAAAAAAAAB0EF, 0x0000BFFC // P_4
data8 0xCCCCCCCCCCC91416, 0x00003FFC // P_3
data8 0x8000000000000000, 0x0000BFFD // P_2
data8 0xAAAAAAAAAAAAAAAB, 0x00003FFD // P_1
LOCAL_OBJECT_END(Constants_log_80_P)

LOCAL_OBJECT_START(Constants_log_80_Q)
// log2_hi, log2_lo, Q_6, Q_5, Q_4, Q_3, Q_2, Q_1
data8 0xB172180000000000,0x00003FFE
data8 0x82E308654361C4C6,0x0000BFE2
data8 0x92492453A51BE0AF,0x00003FFC
data8 0xAAAAAB73A0CFD29F,0x0000BFFC
data8 0xCCCCCCCCCCCE3872,0x00003FFC
data8 0xFFFFFFFFFFFFB4FB,0x0000BFFC
data8 0xAAAAAAAAAAAAAAAB,0x00003FFD
data8 0x8000000000000000,0x0000BFFE
LOCAL_OBJECT_END(Constants_log_80_Q)

LOCAL_OBJECT_START(Constants_log_80_Z_G_H_h1)
// Z1 - 16 bit fixed, G1 and H1 IEEE single, h1 IEEE double
data4 0x00008000,0x3F800000,0x00000000,0x00000000
data4 0x00000000,0x00000000,0x00000000,0x00000000
data4 0x00007879,0x3F70F0F0,0x3D785196,0x00000000
data4 0xEBA0E0D1,0x8B1D330B,0x00003FDA,0x00000000
data4 0x000071C8,0x3F638E38,0x3DF13843,0x00000000
data4 0x9EADD553,0xE2AF365E,0x00003FE2,0x00000000
data4 0x00006BCB,0x3F579430,0x3E2FF9A0,0x00000000
data4 0x752F34A2,0xF585FEC3,0x0000BFE3,0x00000000
data4 0x00006667,0x3F4CCCC8,0x3E647FD6,0x00000000
data4 0x893B03F3,0xF3546435,0x00003FE2,0x00000000
data4 0x00006187,0x3F430C30,0x3E8B3AE7,0x00000000
data4 0x39CDD2AC,0xBABA62E0,0x00003FE4,0x00000000
data4 0x00005D18,0x3F3A2E88,0x3EA30C68,0x00000000
data4 0x457978A1,0x8718789F,0x00003FE2,0x00000000
data4 0x0000590C,0x3F321640,0x3EB9CEC8,0x00000000
data4 0x3185E56A,0x9442DF96,0x0000BFE4,0x00000000
data4 0x00005556,0x3F2AAAA8,0x3ECF9927,0x00000000
data4 0x2BBE2CBD,0xCBF9A4BF,0x00003FE4,0x00000000
data4 0x000051EC,0x3F23D708,0x3EE47FC5,0x00000000
data4 0x852D5935,0xF3537535,0x00003FE3,0x00000000
data4 0x00004EC5,0x3F1D89D8,0x3EF8947D,0x00000000
data4 0x46CDF32F,0xA1F1E699,0x0000BFDF,0x00000000
data4 0x00004BDB,0x3F17B420,0x3F05F3A1,0x00000000
data4 0xD8484CE3,0x84A61856,0x00003FE4,0x00000000
data4 0x00004925,0x3F124920,0x3F0F4303,0x00000000
data4 0xFF28821B,0xC7DD97E0,0x0000BFE2,0x00000000
data4 0x0000469F,0x3F0D3DC8,0x3F183EBF,0x00000000
data4 0xEF1FD32F,0xD3C4A887,0x00003FE3,0x00000000
data4 0x00004445,0x3F088888,0x3F20EC80,0x00000000
data4 0x464C76DA,0x84672BE6,0x00003FE5,0x00000000
data4 0x00004211,0x3F042108,0x3F29516A,0x00000000
data4 0x18835FB9,0x9A43A511,0x0000BFE5,0x00000000
LOCAL_OBJECT_END(Constants_log_80_Z_G_H_h1)

LOCAL_OBJECT_START(Constants_log_80_Z_G_H_h2)
// Z2 - 16 bit fixed, G2 and H2 IEEE single, h2 IEEE double
data4 0x00008000,0x3F800000,0x00000000,0x00000000
data4 0x00000000,0x00000000,0x00000000,0x00000000
data4 0x00007F81,0x3F7F00F8,0x3B7F875D,0x00000000
data4 0x211398BF,0xAD08B116,0x00003FDB,0x00000000
data4 0x00007F02,0x3F7E03F8,0x3BFF015B,0x00000000
data4 0xC376958E,0xB106790F,0x00003FDE,0x00000000
data4 0x00007E85,0x3F7D08E0,0x3C3EE393,0x00000000
data4 0x79A7679A,0xFD03F242,0x0000BFDA,0x00000000
data4 0x00007E08,0x3F7C0FC0,0x3C7E0586,0x00000000
data4 0x05E7AE08,0xF03F81C3,0x0000BFDF,0x00000000
data4 0x00007D8D,0x3F7B1880,0x3C9E75D2,0x00000000
data4 0x049EB22F,0xD1B87D3C,0x00003FDE,0x00000000
data4 0x00007D12,0x3F7A2328,0x3CBDC97A,0x00000000
data4 0x3A9E81E0,0xFABC8B95,0x00003FDF,0x00000000
data4 0x00007C98,0x3F792FB0,0x3CDCFE47,0x00000000
data4 0x7C4B5443,0xF5F3653F,0x00003FDF,0x00000000
data4 0x00007C20,0x3F783E08,0x3CFC15D0,0x00000000
data4 0xF65A1773,0xE78AB204,0x00003FE0,0x00000000
data4 0x00007BA8,0x3F774E38,0x3D0D874D,0x00000000
data4 0x7B8EF695,0xDB7CBFFF,0x0000BFE0,0x00000000
data4 0x00007B31,0x3F766038,0x3D1CF49B,0x00000000
data4 0xCF773FB3,0xC0241AEA,0x0000BFE0,0x00000000
data4 0x00007ABB,0x3F757400,0x3D2C531D,0x00000000
data4 0xC9539FDF,0xFC8F4D48,0x00003FE1,0x00000000
data4 0x00007A45,0x3F748988,0x3D3BA322,0x00000000
data4 0x954665C2,0x9CD035FB,0x0000BFE1,0x00000000
data4 0x000079D1,0x3F73A0D0,0x3D4AE46F,0x00000000
data4 0xDD367A30,0xEC9017C7,0x00003FE1,0x00000000
data4 0x0000795D,0x3F72B9D0,0x3D5A1756,0x00000000
data4 0xCB11189C,0xEE6625D3,0x0000BFE1,0x00000000
data4 0x000078EB,0x3F71D488,0x3D693B9D,0x00000000
data4 0xBE11C424,0xA49C8DB5,0x0000BFE0,0x00000000
LOCAL_OBJECT_END(Constants_log_80_Z_G_H_h2)

LOCAL_OBJECT_START(Constants_log_80_h3_G_H)
// h3 IEEE double extended, H3 and G3 IEEE single
data4 0x112666B0,0xAAACAAB1,0x00003FD3,0x3F7FFC00
data4 0x9B7FAD21,0x90051030,0x00003FD8,0x3F7FF400
data4 0xF4D783C4,0xA6B46F46,0x00003FDA,0x3F7FEC00
data4 0x11C6DDCA,0xDA148D88,0x0000BFD8,0x3F7FE400
data4 0xCA964D95,0xCE65C1D8,0x0000BFD8,0x3F7FDC00
data4 0x23412D13,0x883838EE,0x0000BFDB,0x3F7FD400
data4 0x983ED687,0xB7E5CFA1,0x00003FDB,0x3F7FCC08
data4 0xE3C3930B,0xDBE23B16,0x0000BFD9,0x3F7FC408
data4 0x48AA4DFC,0x9B92F1FC,0x0000BFDC,0x3F7FBC10
data4 0xCE9C8F7E,0x9A8CEB15,0x0000BFD9,0x3F7FB410
data4 0x0DECE74A,0x8C220879,0x00003FDC,0x3F7FAC18
data4 0x2F053150,0xB25CA912,0x0000BFDA,0x3F7FA420
data4 0xD9A5BE20,0xA5876555,0x00003FDB,0x3F7F9C20
data4 0x2053F087,0xC919BB6E,0x00003FD9,0x3F7F9428
data4 0x041E9A77,0xB70BDA79,0x00003FDC,0x3F7F8C30
data4 0xEA1C9C30,0xF18A5C08,0x00003FDA,0x3F7F8438
data4 0x796D89E5,0xA3790D84,0x0000BFDD,0x3F7F7C40
data4 0xA2915A3A,0xE1852369,0x0000BFDD,0x3F7F7448
data4 0xA39ED868,0xD803858F,0x00003FDC,0x3F7F6C50
data4 0x9417EBB7,0xB2EEE356,0x0000BFDD,0x3F7F6458
data4 0x9BB0D07F,0xED5C1F8A,0x0000BFDC,0x3F7F5C68
data4 0xE87C740A,0xD6D201A0,0x0000BFDD,0x3F7F5470
data4 0x1CA74025,0xE8DEBF5E,0x00003FDC,0x3F7F4C78
data4 0x1F34A7EB,0x9A995A97,0x0000BFDC,0x3F7F4488
data4 0x359EED97,0x9CB0F742,0x0000BFDA,0x3F7F3C90
data4 0xBBC6A1C8,0xD6F833C2,0x0000BFDD,0x3F7F34A0
data4 0xE71090EC,0xE1F68F2A,0x00003FDC,0x3F7F2CA8
data4 0xC160A74F,0xD1881CF1,0x0000BFDB,0x3F7F24B8
data4 0xD78CB5A4,0x9AD05AE2,0x00003FD6,0x3F7F1CC8
data4 0x9A77DC4B,0xE658CB8E,0x0000BFDD,0x3F7F14D8
data4 0x6BD6D312,0xBA281296,0x00003FDC,0x3F7F0CE0
data4 0xF95210D0,0xB478BBEB,0x0000BFDB,0x3F7F04F0
data4 0x38800100,0x39400480,0x39A00640,0x39E00C41 // H's start here
data4 0x3A100A21,0x3A300F22,0x3A4FF51C,0x3A6FFC1D
data4 0x3A87F20B,0x3A97F68B,0x3AA7EB86,0x3AB7E101
data4 0x3AC7E701,0x3AD7DD7B,0x3AE7D474,0x3AF7CBED
data4 0x3B03E1F3,0x3B0BDE2F,0x3B13DAAA,0x3B1BD766
data4 0x3B23CC5C,0x3B2BC997,0x3B33C711,0x3B3BBCC6
data4 0x3B43BAC0,0x3B4BB0F4,0x3B53AF6D,0x3B5BA620
data4 0x3B639D12,0x3B6B9444,0x3B7393BC,0x3B7B8B6D
LOCAL_OBJECT_END(Constants_log_80_h3_G_H)

GR_sig_inv_ln2      = r14
GR_rshf_2to51       = r15
GR_exp_2tom51       = r16
GR_rshf             = r17
GR_exp_half         = r18
GR_sign_mask        = r19
GR_exp_square_oflow = r20
GR_exp_square_uflow = r21
GR_exp_ynear1_oflow = r22
GR_exp_ynear1_uflow = r23
GR_signif_Z         = r24

GR_signexp_x        = r32

GR_exp_x            = r33

GR_Table_Ptr        = r34

GR_Table_Ptr1       = r35

GR_Index1           = r36

GR_Index2           = r37
GR_Expo_X           = r37

GR_M                = r38

GR_X_0              = r39
GR_Mask             = r39

GR_X_1              = r40
GR_W1_ptr           = r40

GR_W2_ptr           = r41
GR_X_2              = r41

GR_Z_1              = r42
GR_M2               = r42

GR_M1               = r43
GR_Z_2              = r43

GR_N                = r44
GR_k                = r44

GR_Big_Pos_Exp      = r45

GR_exp_pos_max      = r46

GR_exp_bias_p_k     = r47

GR_Index3           = r48
GR_temp             = r48

GR_vsm_expo         = r49

GR_T1_ptr           = r50
GR_P_ptr1           = r50
GR_T2_ptr           = r51
GR_P_ptr2           = r51
GR_N_fix            = r52
GR_exp_y            = r53
GR_signif_y         = r54
GR_signexp_y        = r55
GR_fraction_y       = r55
GR_low_order_bit    = r56
GR_exp_mask         = r57
GR_exp_bias         = r58
GR_y_sign           = r59
GR_table_base       = r60
GR_ptr_exp_Arg      = r61
GR_Delta_Exp        = r62
GR_Special_Exp      = r63
GR_exp_neg_max      = r64
GR_Big_Neg_Exp      = r65

//** Registers for unwind support

GR_SAVE_PFS         = r59
GR_SAVE_B0          = r60
GR_SAVE_GP          = r61
GR_Parameter_X      = r62
GR_Parameter_Y      = r63
GR_Parameter_RESULT = r64
GR_Parameter_TAG    = r65

//**

FR_Input_X          = f8
FR_Result           = f8
FR_Input_Y          = f9

FR_Neg              = f10
FR_P_hi             = f10
FR_X                = f10

FR_Half             = f11
FR_h_3              = f11
FR_poly_hi          = f11

FR_Sgn              = f12

FR_half_W           = f13

FR_X_cor            = f14
FR_P_lo             = f14

FR_W                = f15

FR_X_lo             = f32

FR_S                = f33
FR_W3               = f33

FR_Y_hi             = f34
FR_logx_hi          = f34

FR_Z                = f35
FR_logx_lo          = f35
FR_GS_hi            = f35
FR_Y_lo             = f35

FR_r_cor            = f36
FR_Scale            = f36

FR_G_1              = f37
FR_G                = f37
FR_Wsq              = f37
FR_temp             = f37

FR_H_1              = f38
FR_H                = f38
FR_W4               = f38

FR_h                = f39
FR_h_1              = f39
FR_N                = f39
FR_P_7              = f39

FR_G_2              = f40
FR_P_8              = f40
FR_L_hi             = f40

FR_H_2              = f41
FR_L_lo             = f41
FR_A_1              = f41

FR_h_2              = f42

FR_W1               = f43

FR_G_3              = f44
FR_P_8              = f44
FR_T1               = f44

FR_log2_hi          = f45
FR_W2               = f45

FR_GS_lo            = f46
FR_T2               = f46

FR_W_1_p1           = f47
FR_H_3              = f47

FR_float_N          = f48

FR_A_2              = f49

FR_Q_4              = f50
FR_r4               = f50

FR_Q_3              = f51
FR_A_3              = f51

FR_Q_2              = f52
FR_P_2              = f52

FR_Q_1              = f53
FR_P_1              = f53
FR_T                = f53

FR_Wp1              = f54
FR_Q_5              = f54
FR_P_3              = f54

FR_Q_6              = f55

FR_log2_lo          = f56
FR_Two              = f56

FR_Big              = f57

FR_neg_2_mK         = f58

FR_r                = f59

FR_poly_lo          = f60

FR_poly             = f61

FR_P_5              = f62
FR_Result_small     = f62

FR_rsq              = f63

FR_Delta            = f64

FR_save_Input_X     = f65
FR_norm_X           = f66
FR_norm_Y           = f67
FR_Y_lo_2           = f68

FR_P_6              = f69
FR_Result_big       = f69

FR_RSHF_2TO51       = f70
FR_INV_LN2_2TO63    = f71
FR_2TOM51           = f72
FR_RSHF             = f73
FR_TMP1             = f74
FR_TMP2             = f75
FR_TMP3             = f76
FR_Tscale           = f77
FR_P_4              = f78
FR_NBig             = f79


.section .text
GLOBAL_LIBM_ENTRY(powl)
//
//     Get significand of x.  It is the critical path.
//
{ .mfi
      getf.sig GR_signif_Z = FR_Input_X    // Get significand of x
      fclass.m p11, p12 = FR_Input_X, 0x0b // Test x unorm
      nop.i 999
}
{ .mfi
      nop.m 999
      fnorm.s1 FR_norm_X = FR_Input_X      // Normalize x
      mov GR_exp_half = 0xffff - 1         // Exponent for 0.5
}
;;

{ .mfi
      alloc  r32 = ar.pfs,0,30,4,0
      fclass.m p7, p0 =  FR_Input_Y, 0x1E7 // Test y natval, nan, inf, zero
      mov GR_exp_pos_max = 0x13fff         // Max exponent for pos oflow test
}
{ .mfi
      addl GR_table_base = @ltoff(Constants_exp_64_Arg#), gp // Ptr to tables
      fnorm.s1 FR_norm_Y = FR_Input_Y      // Normalize y
      mov GR_exp_neg_max = 0x33fff         // Max exponent for neg oflow test
}
;;

{ .mfi
      getf.exp GR_signexp_y = FR_Input_Y   // Get sign and exp of y
(p12) fclass.m p11, p0 =  FR_Input_Y, 0x0b // Test y unorm
      mov GR_sign_mask = 0x20000           // Sign mask
}
{ .mfi
      ld8 GR_table_base = [GR_table_base]  // Get base address for tables
      fadd.s1 FR_Two = f1, f1              // Form 2.0 for square test
      mov GR_exp_mask = 0x1FFFF            // Exponent mask
}
;;

{ .mfi
      getf.sig GR_signif_y = FR_Input_Y    // Get significand of y
      fclass.m p6, p0 =  FR_Input_X, 0x1E7 // Test x natval, nan, inf, zero
      nop.i 999
}
;;

{ .mfi
      getf.exp GR_signexp_x = FR_Input_X   // Get signexp of x
      fmerge.s FR_save_Input_X = FR_Input_X, FR_Input_X
      extr.u GR_Index1 = GR_signif_Z, 59, 4  // Extract upper 4 signif bits of x
}
{ .mfb
      setf.exp FR_Half = GR_exp_half       // Load half
      nop.f 999
(p11) br.cond.spnt  POWL_DENORM            // Branch if x or y denorm/unorm
}
;;

// Return here from POWL_DENORM
POWL_COMMON:
{ .mfi
      setf.exp FR_Big = GR_exp_pos_max     // Form big pos value for oflow test
      fclass.nm p11, p0 = FR_Input_Y, 0x1FF // Test Y unsupported
      shl GR_Index1 = GR_Index1,5          // Adjust index1 pointer x 32
}
{ .mfi
      add GR_Table_Ptr = 0x7c0, GR_table_base // Constants_log_80_Z_G_H_h1
      fma.s1 FR_Sgn = f1,f1,f0             // Assume result positive
      mov GR_exp_bias = 0xFFFF             // Form exponent bias
}
;;

//
//     Identify NatVals, NaNs, Infs, and Zeros.
//
//
//     Remove sign bit from exponent of y.
//     Check for x = 1
//     Branch on Infs, Nans, Zeros, and Natvals
//     Check to see that exponent < 0
//
{ .mfi
      setf.exp FR_NBig = GR_exp_neg_max    // Form big neg value for oflow test
      fclass.nm p8, p0 =  FR_Input_X, 0x1FF  // Test X unsupported
      and GR_exp_y = GR_exp_mask,GR_signexp_y // Get biased exponent of y
}
{ .mfb
      add GR_Index1 = GR_Index1,GR_Table_Ptr
      nop.f 999
(p6)  br.cond.spnt POWL_64_SPECIAL         // Branch if x natval, nan, inf, zero
}
;;

//     load Z_1 from Index1

// There is logic starting here to determine if y is an integer when x < 0.
// If 0 < |y| < 1 then clearly y is not an integer.
// If |y| > 1, then the significand of y is shifted left by the size of
//    the exponent of y.  This preserves the lsb of the integer part + the
//    fractional bits.  The lsb of the integer can be tested to determine if
//    the integer is even or odd.  The fractional bits can be tested.  If zero,
//    then y is an integer.
//
{ .mfi
      ld2 GR_Z_1 =[GR_Index1],4            // Load Z_1
      fmerge.s FR_Z = f0, FR_norm_X        // Z = |x|
      extr.u GR_X_0 = GR_signif_Z, 49, 15  // Extract X_0 from significand
}
{ .mfb
      cmp.lt p9, p0 = GR_exp_y,GR_exp_bias // Test 0 < |y| < 1
      nop.f 999
(p7)  br.cond.spnt POWL_64_SPECIAL         // Branch if y natval, nan, inf, zero
}
;;

{ .mfb
      ldfs  FR_G_1 = [GR_Index1],4         // Load G_1
      fcmp.eq.s1 p10, p0 =  FR_Input_Y, f1 // Test Y = +1.0
(p8)  br.cond.spnt POWL_64_UNSUPPORT       // Branch if x unsupported
}
;;

//
//     X_0  = High order 15 bit of Z
//
{ .mfb
      ldfs  FR_H_1 = [GR_Index1],8             // Load H_1
(p9)  fcmp.lt.unc.s1 p9, p0 = FR_Input_X, f0   // Test x<0, 0 <|y|<1
(p11) br.cond.spnt POWL_64_UNSUPPORT           // Branch if y unsupported
}
;;

{ .mfi
      ldfe FR_h_1 = [GR_Index1]                // Load h_1
      fcmp.eq.s1 p7, p0 =  FR_Input_Y, FR_Two  // Test y = 2.0
      pmpyshr2.u GR_X_1 = GR_X_0,GR_Z_1,15     // X_1 = X_0 * Z_1 (bits 15-30)
                                               // Wait 4 cycles to use result
}
{ .mfi
      add GR_Table_Ptr = 0x9c0, GR_table_base  // Constants_log_80_Z_G_H_h2
      nop.f 999
      sub GR_exp_y = GR_exp_y,GR_exp_bias      // Get true exponent of y
}
;;

//
//      Branch for (x < 0) and Y not an integer.
//
{ .mfb
      nop.m 999
      fcmp.lt.s1 p6, p0  =  FR_Input_X, f0     // Test x < 0
(p9)  br.cond.spnt POWL_64_XNEG                // Branch if x < 0, 0 < |y| < 1
}
;;

{ .mfi
      nop.m 999
      fcmp.eq.s1 p12, p0 =  FR_Input_X, f1     // Test x=+1.0
      nop.i 999
}
{ .mfb
      nop.m 999
      fsub.s1 FR_W = FR_Z, f1                  // W = Z - 1
(p7)  br.cond.spnt POWL_64_SQUARE              // Branch if y=2
}
;;

{ .mfi
      nop.m 999
(p10) fmpy.s0 FR_Result = FR_Input_X, f1       // If y=+1.0, result=x
(p6)  shl GR_fraction_y=  GR_signif_y,GR_exp_y // Get lsb of int + fraction
                                               // Wait 4 cycles to use result
}
;;

{ .mfi
      nop.m 999
(p12) fma.s0 FR_Result = FR_Input_Y, f0, f1    // If x=1.0, result=1, chk denorm
      extr.u GR_Index2 = GR_X_1, 6, 4          // Extract index2
}
;;

//
//     N = exponent of Z
//
{ .mib
      getf.exp GR_N =  FR_Z                    // Get exponent of Z (also x)
      shl GR_Index2=GR_Index2,5                // Index2  x 32 bytes
(p10) br.ret.spnt  b0                          // Exit if y=+1.0
}
;;

{ .mib
      add GR_Index2 = GR_Index2, GR_Table_Ptr  // Pointer to table 2
      nop.i 999
(p12) br.ret.spnt  b0                          // Exit if x=+1.0
}
;;

{ .mmi
      ld2 GR_Z_2 =[GR_Index2],4                // Load Z_2
;;
      ldfs  FR_G_2 = [GR_Index2],4             // Load G_2
      nop.i 999
}
;;

{ .mii
      ldfs  FR_H_2 = [GR_Index2],8             // Load H_2
(p6)  tbit.nz.unc p9, p0 = GR_fraction_y, 63   // Test x<0 and y odd integer
      add GR_Table_Ptr = 0xbcc, GR_table_base  // Constants_log_80_h3_G_H, G_3
}
;;

//
//      For x < 0 and y odd integer,, set sign = -1.
//
{ .mfi
      getf.exp GR_M = FR_W                      // Get signexp of W
      nop.f 999
      pmpyshr2.u GR_X_2 = GR_X_1,GR_Z_2,15      // X_2 = X_1 * Z_2 (bits 15-30)
}
{ .mfi
      ldfe FR_h_2 = [GR_Index2]                // Load h_2
(p9)  fnma.s1 FR_Sgn = f1, f1, f0          // If x<0, y odd int, result negative
      sub GR_N = GR_N, GR_exp_bias             // Get true exponent of x = N
}
;;

{ .mfi
      add GR_Table_Ptr1 = 0xdc0, GR_table_base // Ptr to H_3
      fcmp.eq.s0 p11, p0 = FR_Input_Y, FR_Half // Test y=0.5, also set denorm
(p6)  shl GR_fraction_y=  GR_fraction_y, 1     // Shift left 1 to get fraction
}
;;

{ .mmb
      setf.sig FR_float_N = GR_N
(p6)  cmp.ne.unc p8, p0 = GR_fraction_y, r0    // Test x<0 and y not integer
(p8)  br.cond.spnt POWL_64_XNEG                // Branch if x<0 and y not int
}
;;

//
//      Raise possible denormal operand exception for both X and Y.
//      Set pointers in case |x| near 1
//      Branch to embedded sqrt(x) if y=0.5
//
{ .mfi
      add GR_P_ptr1 = 0x6b0, GR_table_base // Constants_log_80_P, P8, NEAR path
      fcmp.eq.s0 p12, p0 =  FR_Input_X, FR_Input_Y // Dummy to set denormal
      add GR_P_ptr2 = 0x700, GR_table_base // Constants_log_80_P, P4, NEAR path
}
{ .mfb
      cmp.eq p15, p14 =  r0, r0            // Assume result safe (no over/under)
      fsub.s1  FR_Delta = FR_Input_Y,f1    // Delta = y - 1.0
(p11) br.cond.spnt POWL_64_SQRT            // Branch if y=0.5
}
;;

//
//     Computes ln( x ) to extra precision
//     Input  FR 1: FR_X
//     Output FR 2: FR_Y_hi
//     Output FR 3: FR_Y_lo
//     Output PR 1: PR_Safe
//
{ .mfi
      and GR_M = GR_exp_mask, GR_M            // Mask to get exponent of W
      nop.f 999
      extr.u GR_Index3 = GR_X_2, 1, 5         // Get index3
}
;;

{ .mmi
      shladd GR_Table_Ptr1 = GR_Index3,2,GR_Table_Ptr1 // Ptr to H_3
      shladd GR_Index3 = GR_Index3,4,GR_Table_Ptr      // Ptr to G_3
      sub GR_M = GR_M, GR_exp_bias            // Get true exponent of W
}
;;

{ .mib
      ldfs FR_G_3 = [GR_Index3],-12           // Load G_3
      cmp.gt  p7, p14 =  -8, GR_M             // Test if |x-1| < 2^-8
(p7)  br.cond.spnt LOGL80_NEAR                // Branch if |x-1| < 2^-8
}
;;

// Here if |x-1| >= 2^-8
{ .mmf
      ldfs FR_H_3 = [GR_Table_Ptr1]           // Load H_3
      nop.m 999
      nop.f 999
}
;;

{ .mfi
      ldfe FR_h_3 = [GR_Index3]               // Load h_3
      fmerge.se FR_S =  f1,FR_Z               // S = merge of 1.0 and signif(Z)
      nop.i 999
}
{ .mfi
      add GR_Table_Ptr = 0x740, GR_table_base // Constants_log_80_Q
      fmpy.s1 FR_G = FR_G_1, FR_G_2           // G = G_1 * G_2
      nop.i 999
}
;;

//
//     Begin Loading Q's -  load log2_hi part
//
{ .mfi
      ldfe FR_log2_hi = [GR_Table_Ptr],16     // Load log2_hi
      fadd.s1 FR_H = FR_H_1, FR_H_2           // H = H_1 + H_2
      nop.i 999
};;

//
//     h = h_1 + h_2
//
{ .mfi
      ldfe FR_log2_lo = [GR_Table_Ptr],16     // Load log2_lo
      fadd.s1 FR_h = FR_h_1, FR_h_2           // h = h_1 + h_2
      nop.i 999
}
;;

{ .mfi
      ldfe FR_Q_6 = [GR_Table_Ptr],16         // Load Q_6
      fcvt.xf FR_float_N = FR_float_N
      nop.i 999
}
;;

{ .mfi
      ldfe FR_Q_5 = [GR_Table_Ptr],16         // Load Q_5
      nop.f 999
      nop.i 999
}
;;

//
//     G = G_1 * G_2 * G_3
//
{ .mfi
      ldfe FR_Q_4 = [GR_Table_Ptr],16         // Load Q_4
      fmpy.s1 FR_G = FR_G, FR_G_3
      nop.i 999
}
;;

//
//     H = H_1 + H_2 + H_3
//
{ .mfi
      ldfe FR_Q_3 = [GR_Table_Ptr],16         // Load Q_3
      fadd.s1 FR_H = FR_H, FR_H_3
      nop.i 999
}
;;

//
//     Y_lo = poly + Y_lo
//
//     h = h_1 + h_2 + h_3
//
{ .mfi
      ldfe FR_Q_2 = [GR_Table_Ptr],16         // Load Q_2
      fadd.s1 FR_h = FR_h, FR_h_3
      nop.i 999
}
;;

//
//     GS_hi = G*S
//     r = G*S -1
//
{ .mfi
      ldfe FR_Q_1 = [GR_Table_Ptr],16         // Load Q_1
      fmpy.s1 FR_GS_hi = FR_G, FR_S
      nop.i 999
}
{ .mfi
      nop.m 999
      fms.s1 FR_r = FR_G, FR_S, f1
      nop.i 999
}
;;

//
//     poly_lo = Q_5 + r * Q_6
//
{ .mfi
      getf.exp GR_Delta_Exp =  FR_Delta     // Get signexp of y-1 for exp calc
      fma.s1 FR_poly_lo = FR_r, FR_Q_6, FR_Q_5
      nop.i 999
}
//
//     r_cor = GS_hi -1
//
{ .mfi
      nop.m 999
      fsub.s1 FR_r_cor = FR_GS_hi, f1
      nop.i 999
}
;;

//
//     GS_lo  = G*S - GS_hi
//
{ .mfi
      nop.m 999
      fms.s1 FR_GS_lo = FR_G, FR_S, FR_GS_hi
      nop.i 999
}
;;

//
//     rsq = r * r
//
{ .mfi
      nop.m 999
      fmpy.s1 FR_rsq = FR_r, FR_r
      nop.i 999
}
//
//     G = float_N*log2_hi + H
//
{ .mfi
      nop.m 999
      fma.s1 FR_G = FR_float_N, FR_log2_hi, FR_H
      nop.i 999
}
;;

//
//     Y_lo = float_N*log2_lo + h
//
{ .mfi
      nop.m 999
      fma.s1 FR_Y_lo = FR_float_N, FR_log2_lo, FR_h
      nop.i 999
}
;;

//
//      poly_lo = Q_4 + r * poly_lo
//      r_cor = r_cor - r
//
{ .mfi
      nop.m 999
      fma.s1 FR_poly_lo = FR_r, FR_poly_lo, FR_Q_4
      nop.i 999
}
{ .mfi
      nop.m 999
      fsub.s1 FR_r_cor = FR_r_cor, FR_r
      nop.i 999
}
;;

//
//      poly_hi = r * Q_2 + Q_1
//      Y_hi = G + r
//
{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_r, FR_Q_2, FR_Q_1
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_Y_hi = FR_G, FR_r
      nop.i 999
}
;;

//
//      poly_lo = Q_3 + r * poly_lo
//      r_cor = r_cor + GS_lo
//
{ .mfi
      nop.m 999
      fma.s1 FR_poly_lo = FR_r, FR_poly_lo, FR_Q_3
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_r_cor = FR_r_cor, FR_GS_lo
      nop.i 999
}
;;

//
//      Y_lo = G - Y_hi
//
{ .mfi
      nop.m 999
      fsub.s1 FR_Y_lo_2 = FR_G, FR_Y_hi
      nop.i 999
}
;;

//
//      r_cor = r_cor + Y_lo
//      poly = poly_hi + rsq * poly_lo
//
{ .mfi
      add  GR_Table_Ptr   = 0x0, GR_table_base   // Constants_exp_64_Arg
      fadd.s1 FR_r_cor = FR_r_cor, FR_Y_lo
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_rsq, FR_poly_lo, FR_poly
      nop.i 999
}
;;

//
//      Load L_hi
//      Load L_lo
//      all long before they are needed.
//      They are used in LOGL_RETURN PATH
//
//      Y_lo =  Y_lo + r
//      poly = rsq * poly + r_cor
//
{ .mfi
      ldfe FR_L_hi = [GR_Table_Ptr],16           // Load L_hi
      fadd.s1 FR_Y_lo = FR_Y_lo_2, FR_r
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_rsq, FR_poly, FR_r_cor
      nop.i 999
}
;;

{ .mfb
      ldfe FR_L_lo = [GR_Table_Ptr],16           // Load L_lo
      fadd.s1 FR_Y_lo = FR_Y_lo, FR_poly
      br.cond.sptk LOGL_RETURN                   // Branch to common code
}
;;


LOGL80_NEAR:
// Here if |x-1| < 2^-8
//
//     Branch LOGL80_NEAR
//

{ .mmf
      ldfe FR_P_8 = [GR_P_ptr1],16           // Load P_8
      ldfe FR_P_4 = [GR_P_ptr2],16           // Load P_4
      fmpy.s1 FR_Wsq = FR_W, FR_W
}
;;

{ .mmi
      ldfe FR_P_7 = [GR_P_ptr1],16           // Load P_7
      ldfe FR_P_3 = [GR_P_ptr2],16           // Load P_3
      nop.i 999
}
;;

{ .mmi
      ldfe FR_P_6 = [GR_P_ptr1],16           // Load P_6
      ldfe FR_P_2 = [GR_P_ptr2],16           // Load P_2
      nop.i 999
}
;;

{ .mmi
      ldfe FR_P_5 = [GR_P_ptr1],16           // Load P_5
      ldfe FR_P_1 = [GR_P_ptr2],16           // Load P_1
      nop.i 999
}
;;

{ .mfi
      getf.exp GR_Delta_Exp =  FR_Delta      // Get signexp of y-1 for exp calc
      fmpy.s1 FR_W4 = FR_Wsq, FR_Wsq
      nop.i 999
}
{ .mfi
      add  GR_Table_Ptr = 0x0, GR_table_base // Constants_exp_64_Arg
      fmpy.s1 FR_W3 = FR_Wsq, FR_W
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_half_W = FR_Half, FR_W
      nop.i 999
}
;;

{ .mfi
      ldfe FR_L_hi = [GR_Table_Ptr],16
      fma.s1 FR_poly_lo = FR_W, FR_P_8,FR_P_7
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_W, FR_P_4, FR_P_3
      nop.i 999
}
;;

{ .mfi
      ldfe FR_L_lo = [GR_Table_Ptr],16
      fnma.s1 FR_Y_hi = FR_W, FR_half_W, FR_W
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly_lo = FR_W, FR_poly_lo, FR_P_6
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_W, FR_poly, FR_P_2
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1 FR_Y_lo = FR_W, FR_Y_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly_lo = FR_W, FR_poly_lo, FR_P_5
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_W, FR_poly, FR_P_1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fnma.s1 FR_Y_lo = FR_W, FR_half_W, FR_Y_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_poly_lo, FR_W4, FR_poly
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_Y_lo = FR_poly, FR_W3, FR_Y_lo
      nop.i 999
}
;;


LOGL_RETURN:
// Common code for completion of both logx paths

//
//     L_hi, L_lo already loaded.
//
//
//     kernel_log_80 computed ln(X)
//     and return logX_hi and logX_lo as results.
//     PR_pow_Safe set as well.
//
//
//     Compute Y * (logX_hi + logX_lo)
//     P_hi -> X
//     P_lo -> X_cor
//     (Manipulate names so that inputs are in
//     the place kernel_exp expects them)
//
//     This function computes exp( x  + x_cor)
//     Input  FR 1: FR_X
//     Input  FR 2: FR_X_cor
//     Output FR 3: FR_Y_hi
//     Output FR 4: FR_Y_lo
//     Output FR 5: FR_Scale
//     Output PR 1: PR_Safe
//
//     P15 is True
//
// Load constants used in computing N using right-shift technique
{ .mlx
      mov GR_exp_2tom51 = 0xffff-51
      movl GR_sig_inv_ln2 = 0xb8aa3b295c17f0bc  // significand of 1/ln2
}
{ .mlx
      add  GR_Special_Exp = -50,GR_exp_bias
      movl GR_rshf_2to51 = 0x4718000000000000   // 1.10000 2^(63+51)
}
;;

//
//     Point to Table of W1s
//     Point to Table of W2s
//
{ .mmi
      add GR_W1_ptr   = 0x2b0, GR_table_base    // Constants_exp_64_W1
      add GR_W2_ptr   = 0x4b0, GR_table_base    // Constants_exp_64_W2
      cmp.le p6,p0= GR_Delta_Exp,GR_Special_Exp
};;

// Form two constants we need
//  1/ln2 * 2^63  to compute  w = x * 1/ln2 * 128
//  1.1000..000 * 2^(63+63-12) to right shift int(N) into the significand

{ .mfi
      setf.sig  FR_INV_LN2_2TO63 = GR_sig_inv_ln2 // form 1/ln2 * 2^63
      nop.f 999
      and GR_Delta_Exp=GR_Delta_Exp,GR_exp_mask  // Get exponent of y-1
}
{ .mlx
      setf.d  FR_RSHF_2TO51 = GR_rshf_2to51    // Form const 1.1000 * 2^(63+51)
      movl GR_rshf = 0x43e8000000000000        // 1.10000 2^63 for right shift
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_X_lo = FR_Input_Y, FR_logx_lo // logx_lo is Y_lo
      cmp.eq  p15, p0=  r0, r0                 // Set p15, assume safe
};;

{ .mmi
      setf.exp FR_2TOM51 = GR_exp_2tom51 // Form 2^-51 for scaling float_N
      setf.d  FR_RSHF = GR_rshf          // Form right shift const 1.1000 * 2^63
      add GR_Table_Ptr1   = 0x50, GR_table_base // Constants_exp_64_P for
                                                // EXPL_SMALL path
}
;;

{ .mmi
      ldfe FR_P_6 = [GR_Table_Ptr1],16          // Load P_6 for EXPL_SMALL path
;;
      ldfe FR_P_5 = [GR_Table_Ptr1],16          // Load P_5 for EXPL_SMALL path
      nop.i 999
}
;;

{ .mfi
      ldfe FR_P_4 = [GR_Table_Ptr1],16          // Load P_4 for EXPL_SMALL path
      fma.s1 FR_P_hi = FR_Input_Y, FR_logx_hi,FR_X_lo  // logx_hi ix Y_hi
      nop.i 999
}
;;

{ .mmi
      ldfe FR_P_3 = [GR_Table_Ptr1],16          // Load P_3 for EXPL_SMALL path
;;
      ldfe FR_P_2 = [GR_Table_Ptr1],16          // Load P_2 for EXPL_SMALL path
      nop.i 999
}
;;

// N = X * Inv_log2_by_2^12
// By adding 1.10...0*2^63 we shift and get round_int(N_signif) in significand.
// We actually add 1.10...0*2^51 to X * Inv_log2 to do the same thing.
{ .mfi
      ldfe FR_P_1 = [GR_Table_Ptr1]             // Load P_1 for EXPL_SMALL path
      fma.s1 FR_N = FR_X, FR_INV_LN2_2TO63, FR_RSHF_2TO51
      nop.i 999
}
{ .mfb
      nop.m 999
      fms.s1 FR_P_lo= FR_Input_Y, FR_logx_hi, FR_P_hi  // P_hi is X
(p6)  br.cond.spnt POWL_Y_ALMOST_1              // Branch if |y-1| < 2^-50
}
;;

{ .mmi
      getf.exp GR_Expo_X = FR_X
      add GR_T1_ptr   = 0x0b0, GR_table_base    // Constants_exp_64_T1
      add GR_T2_ptr   = 0x1b0, GR_table_base    // Constants_exp_64_T2
}
;;

// float_N = round_int(N)
// The signficand of N contains the rounded integer part of X * 2^12/ln2,
// as a twos complement number in the lower bits (that is, it may be negative).
// That twos complement number (called N) is put into GR_N_fix.

// Since N is scaled by 2^51, it must be multiplied by 2^-51
// before the shift constant 1.10000 * 2^63 is subtracted to yield float_N.
// Thus, float_N contains the floating point version of N


{ .mfi
      add  GR_Table_Ptr   = 0x20, GR_table_base    // Constants_exp_64_A
      fms.s1 FR_float_N = FR_N, FR_2TOM51, FR_RSHF // Form float_N
      nop.i 999
}
//     Create low part of Y(ln(x)_hi + ln(x)_lo) as P_lo
{ .mfi
      mov GR_Big_Pos_Exp = 0x3ffe               // 16382, largest safe exponent
      fadd.s1 FR_P_lo = FR_P_lo, FR_X_lo
      mov GR_Big_Neg_Exp = -0x3ffd              // -16381 smallest safe exponent
};;

{ .mfi
      nop.m 999
      fmpy.s1 FR_rsq = FR_X, FR_X               // rsq = X*X for EXPL_SMALL path
      mov GR_vsm_expo = -70                     // Exponent for very small path
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly_lo = FR_P_6, FR_X, FR_P_5  // poly_lo for EXPL_SMALL path
      add GR_temp = 0x1,r0                      // For tiny signif if small path
}
;;

//
//      If expo_X < -6 goto exp_small
//
{ .mmi
      getf.sig GR_N_fix = FR_N
      ldfe FR_A_3 = [GR_Table_Ptr],16         // Load A_3
      and GR_Expo_X = GR_Expo_X, GR_exp_mask  // Get exponent of X
}
;;

{ .mfi
      ldfe FR_A_2 = [GR_Table_Ptr],16         // Load A_2
      nop.f 999
      sub GR_Expo_X = GR_Expo_X, GR_exp_bias  // Get true exponent of X
}
;;

//
//     If -6 > Expo_X, set P9 and branch
//
{ .mfb
      cmp.gt  p9, p0  =  -6, GR_Expo_X
      fnma.s1 FR_r = FR_L_hi, FR_float_N, FR_X // r = X - L_hi * float_N
(p9)  br.cond.spnt EXPL_SMALL                  // Branch if |X| < 2^-6
}
;;

//
//     If 14 <= Expo_X, set P10
//
{ .mib
      cmp.le  p10, p0 =  14, GR_Expo_X
      nop.i 999
(p10) br.cond.spnt EXPL_HUGE                   // Branch if |X| >= 2^14
}
;;

//
//      Load single T1
//      Load single T2
//      W_1_p1 = W_1 + 1
//
{ .mmi
      nop.m 999
      nop.m 999
      extr.u GR_M1 = GR_N_fix, 6, 6            // Extract index M_1
}
;;

//
//      k = extr.u(N_fix,0,6)
//
{ .mmi
      shladd GR_W1_ptr = GR_M1,3,GR_W1_ptr     // Point to W1
      shladd GR_T1_ptr = GR_M1,2,GR_T1_ptr     // Point to T1
      extr.u GR_M2 = GR_N_fix, 0, 6            // Extract index M_2
}
;;

// N_fix is only correct up to 50 bits because of our right shift technique.
// Actually in the normal path we will have restricted K to about 14 bits.
// Somewhat arbitrarily we extract 32 bits.
{ .mmi
      ldfd  FR_W1 = [GR_W1_ptr]
      shladd GR_W2_ptr = GR_M2,3,GR_W2_ptr     // Point to W2
      extr GR_k = GR_N_fix, 12, 32             // Extract k
}
;;

{ .mfi
      ldfs  FR_T1 = [GR_T1_ptr]
      fnma.s1 FR_r = FR_L_lo, FR_float_N, FR_r
      shladd GR_T2_ptr = GR_M2,2,GR_T2_ptr     // Point to T2
}
{ .mfi
      add GR_exp_bias_p_k = GR_exp_bias, GR_k
      nop.f 999
      cmp.gt  p14,p15 = GR_k,GR_Big_Pos_Exp
}
;;

//
//      if k < big_neg_exp, set p14 and Safe=False
//
{ .mmi
      ldfs  FR_T2 = [GR_T2_ptr]
(p15) cmp.lt p14,p15 = GR_k,GR_Big_Neg_Exp
      nop.i 999
}
;;

{ .mmi
      setf.exp FR_Scale = GR_exp_bias_p_k
      ldfd  FR_W2 = [GR_W2_ptr]
      nop.i 999
}
;;

{ .mfi
      ldfe FR_A_1 = [GR_Table_Ptr],16
      fadd.s1 FR_r = FR_r, FR_X_cor
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 FR_W_1_p1 = FR_W1, f1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_r, FR_A_3, FR_A_2
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 FR_rsq = FR_r, FR_r
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_T = FR_T1, FR_T2
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_W = FR_W2, FR_W_1_p1, FR_W1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_TMP1 = FR_Scale, FR_Sgn, f0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_r, FR_poly, FR_A_1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_TMP2 = FR_T, f1, f0            // TMP2 = Y_hi = T
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 FR_Wp1 = FR_W, f1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_rsq, FR_poly,FR_r
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_Tscale = FR_T, FR_TMP1, f0    // Scale * Sgn * T
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_Y_lo = FR_Wp1, FR_poly, FR_W
      nop.i 999
}
;;

{ .mfb
      nop.m 999
      fmpy.s1 FR_TMP3 = FR_Y_lo, FR_Tscale
      br.cond.sptk POWL_64_SHARED
}
;;


EXPL_SMALL:
// Here if |ylogx| < 2^-6
//
//     Begin creating lsb to perturb final result
//
{ .mfi
      setf.sig FR_temp = GR_temp
      fma.s1 FR_poly_lo = FR_poly_lo, FR_X, FR_P_4
      cmp.lt  p12, p0 =  GR_Expo_X, GR_vsm_expo   // Test |ylogx| < 2^-70
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly_hi = FR_P_2, FR_X, FR_P_1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_TMP2 = f1, f1
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 FR_TMP1 = FR_Sgn, f1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_r4 = FR_rsq, FR_rsq
(p12) cmp.eq  p15, p0 =  r0, r0                   // Set safe if |ylogx| < 2^-70
}
{ .mfb
      nop.m 999
(p12) fmpy.s1 FR_TMP3 = FR_Sgn, FR_X
(p12) br.cond.spnt POWL_64_SHARED                 // Branch if |ylogx| < 2^-70
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly_lo = FR_poly_lo, FR_X, FR_P_3
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_poly_hi = FR_poly_hi, FR_rsq, FR_X
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_Y_lo = FR_poly_lo, FR_r4, FR_poly_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_TMP3 = FR_Y_lo, FR_TMP1      // Add sign info
      nop.i 999
}
;;

//
//     Toggle on last bit of Y_lo
//     Set lsb of Y_lo to 1
//
{ .mfi
      nop.m 999
      for FR_temp = FR_Y_lo,FR_temp
      nop.i 999
}
;;

{ .mfb
      nop.m 999
      fmerge.se FR_TMP3 = FR_TMP3,FR_temp
      br.cond.sptk POWL_64_SHARED
}
;;


EXPL_HUGE:
// Here if |ylogx| >= 2^14
{ .mfi
      mov GR_temp = 0x0A1DC               // If X < 0, exponent -24100
      fcmp.gt.s1 p12, p13 =  FR_X, f0     // Test X > 0
      cmp.eq  p14, p15 =  r0, r0          // Set Safe to false
}
;;

{ .mmi
(p12) mov GR_Mask = 0x15DC0               // If X > 0, exponent +24000
(p13) mov GR_Mask = 0x0A240               // If X < 0, exponent -24000
      nop.i 999
}
;;

{ .mmf
      setf.exp FR_TMP2 = GR_Mask          // Form Y_hi = TMP2
(p13) setf.exp FR_Y_lo = GR_temp          // If X < 0, Y_lo = 2^-24100
(p12) mov FR_Y_lo = f1                    // IF X > 0, Y_lo = 1.0
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_TMP1 = FR_TMP2, FR_Sgn   // TMP1 = Y_hi * Sgn
      nop.i 999
}
;;

{ .mfb
      nop.m 999
      fmpy.s1 FR_TMP3 = FR_Y_lo,FR_TMP1   // TMP3 = Y_lo * (Y_hi * Sgn)
      br.cond.sptk POWL_64_SHARED
}
;;

POWL_Y_ALMOST_1:
// Here if delta = |y-1| < 2^-50
//
//  x**(1 + delta) = x * e (ln(x)*delta) = x ( 1 + ln(x) * delta)
//
// Computation will be safe for 2^-16381 <= x < 2^16383

{ .mfi
       mov GR_exp_ynear1_oflow = 0xffff + 16383
       fma.s1 FR_TMP1 = FR_Input_X,FR_Delta,f0
       and GR_exp_x = GR_exp_mask, GR_signexp_x
}
;;

{ .mfi
       cmp.lt  p15, p14 =  GR_exp_x, GR_exp_ynear1_oflow
       fma.s1 FR_TMP2 = FR_logx_hi,f1,FR_X_lo
       mov GR_exp_ynear1_uflow = 0xffff - 16381
}
;;

{ .mfb
(p15)  cmp.ge  p15, p14 =  GR_exp_x, GR_exp_ynear1_uflow
       fma.s1 FR_TMP3 = FR_Input_X,f1,f0
       br.cond.sptk POWL_64_SHARED
};;

POWL_64_SQUARE:
//
//      Here if x not zero and y=2.
//
//      Setup for multipath code
//
{ .mfi
      mov GR_exp_square_oflow = 0xffff + 8192   // Exponent where x*x overflows
      fmerge.se FR_TMP1 = FR_Input_X, FR_Input_X
      and GR_exp_x = GR_exp_mask, GR_signexp_x  // Get exponent of x
}
;;

{ .mfi
      cmp.lt  p15, p14 =  GR_exp_x, GR_exp_square_oflow // Decide safe/unsafe
      fmerge.se FR_TMP2 = FR_Input_X, FR_Input_X
      mov GR_exp_square_uflow = 0xffff - 8191   // Exponent where x*x underflows
}
;;

{ .mfi
(p15) cmp.ge  p15, p14 =  GR_exp_x, GR_exp_square_uflow // Decide safe/unsafe
      fma.s1 FR_TMP3 = f0,f0,f0
      nop.i 999
}
;;

//
//      This is the shared path that will set overflow and underflow.
//
POWL_64_SHARED:

//
//      Return if no danger of over or underflow.
//
{ .mfb
      nop.m 999
      fma.s0 FR_Result = FR_TMP1, FR_TMP2, FR_TMP3
(p15) br.ret.sptk  b0      // Main path return if certain no over/underflow
}
;;

//
//      S0 user supplied status
//      S2 user supplied status + WRE + TD  (Overflows)
//      S2 user supplied status + FZ + TD   (Underflows)
//
//
//     If (Safe) is true, then
//        Compute result using user supplied status field.
//        No overflow or underflow here, but perhaps inexact.
//        Return
//     Else
//       Determine if overflow or underflow was raised.
//       Fetch +/- overflow threshold for IEEE double extended

{ .mfi
      nop.m 999
      fsetc.s2 0x7F,0x41       // For underflow test, set S2=User+TD+FTZ
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s2 FR_Result_small = FR_TMP1, FR_TMP2, FR_TMP3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsetc.s2 0x7F,0x42       // For overflow test, set S2=User+TD+WRE
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s2 FR_Result_big = FR_TMP1, FR_TMP2,FR_TMP3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsetc.s2 0x7F,0x40       // Reset S2=User
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p11, p0 = FR_Result_small, 0x00F // Test small result unorm/zero
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcmp.ge.s1 p8, p0 = FR_Result_big , FR_Big // Test >= + oflow threshold
      nop.i 999
}
;;

{ .mfb
(p11) mov   GR_Parameter_TAG = 19                // Set tag for underflow
      fcmp.le.s1 p9, p0 = FR_Result_big, FR_NBig // Test <= - oflow threshold
(p11) br.cond.spnt __libm_error_region           // Branch if pow underflowed
}
;;

{ .mfb
(p8)  mov   GR_Parameter_TAG = 18                // Set tag for overflow
      nop.f 999
(p8)  br.cond.spnt __libm_error_region           // Branch if pow +overflow
}
;;

{ .mbb
(p9)  mov   GR_Parameter_TAG = 18                // Set tag for overflow
(p9)  br.cond.spnt __libm_error_region           // Branch if pow -overflow
      br.ret.sptk  b0                            // Branch if result really ok
}
;;


POWL_64_SPECIAL:
// Here if x or y is NatVal, nan, inf, or zero
{ .mfi
      nop.m 999
      fcmp.eq.s1 p15, p0 =  FR_Input_X, f1  // Test x=+1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p8, p0 =  FR_Input_X, 0x143  // Test x natval, snan
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p15) fcmp.eq.unc.s0 p6,p0 = FR_Input_Y, f0 // If x=1, flag invalid if y=SNaN
      nop.i 999
}
{ .mfb
      nop.m 999
(p15) fmpy.s0 FR_Result = f1,f1             // If x=1, result=1
(p15) br.ret.spnt b0                        // Exit if x=1
}
;;

{ .mfi
      nop.m 999
      fclass.m p6, p0 =  FR_Input_Y, 0x007  // Test y zero
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p9, p0 =  FR_Input_Y, 0x143  // Test y natval, snan
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p10, p0 =  FR_Input_X, 0x083 // Test x qnan
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fmpy.s0 FR_Result = FR_Input_Y, FR_Input_X // If x=snan, result=qnan
(p6)  cmp.ne p8,p0 = r0,r0     // Don't exit if x=snan, y=0 ==> result=+1
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc p15, p0 =  FR_Input_X,0x007   // Test x=0, y=0
      nop.i 999
}
{ .mfb
      nop.m 999
(p9)  fmpy.s0 FR_Result = FR_Input_Y, FR_Input_X // If y=snan, result=qnan
(p8)  br.ret.spnt b0                             // Exit if x=snan, y not 0,
                                                 //   result=qnan
}
;;

{ .mfi
      nop.m 999
      fcmp.eq.s1 p7, p0 =  FR_Input_Y, f1        // Test y +1.0
      nop.i 999
}
{ .mfb
      nop.m 999
(p10) fmpy.s0 FR_Result = FR_Input_X, f0         // If x=qnan, result=qnan
(p9)  br.ret.spnt b0                             // Exit if y=snan, result=qnan
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc p8, p0 =  FR_Input_X,0x0C3    // Test x=nan, y=0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fcmp.eq.s0 p9,p0 = FR_Input_X, f0          // If y=0, flag if x denormal
      nop.i 999
}
{ .mfi
      nop.m 999
(p6)  fadd.s0 FR_Result = f1, f0                 // If y=0, result=1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p11, p0 =  FR_Input_Y, 0x083      // Test y qnan
      nop.i 999
}
{ .mfb
(p15) mov GR_Parameter_TAG = 20                  // Error tag for x=0, y=0
(p7)  fmpy.s0 FR_Result = FR_Input_X,f1          // If y=1, result=x
(p15) br.cond.spnt __libm_error_region           // Branch if x=0, y=0, result=1
}
;;

{ .mfb
(p8)  mov GR_Parameter_TAG = 23                  // Error tag for x=nan, y=0
      fclass.m p14, p0 =  FR_Input_Y, 0x023      // Test y inf
(p8)  br.cond.spnt __libm_error_region           // Branch if x=snan, y=0,
                                                 //   result=1
}
;;

{ .mfb
      nop.m 999
      fclass.m p13, p0 =  FR_Input_X, 0x023      // Test x inf
(p6)  br.ret.spnt b0                             // Exit y=0, x not nan or 0,
                                                 //   result=1
}
;;

{ .mfb
      nop.m 999
(p14) fcmp.eq.unc.s1 p0,p14 = FR_Input_X,f0      // Test x not 0, y=inf
(p7)  br.ret.spnt b0                             // Exit y=1, x not snan,
                                                 //   result=x
}
;;

{ .mfb
      nop.m 999
(p10) fmpy.s0 FR_Result = FR_Input_Y,FR_Input_X  // If x=qnan, y not snan,
                                                 //   result=qnan
(p10) br.ret.spnt b0                             // Exit x=qnan, y not snan,
                                                 //   result=qnan
}
;;

{ .mfb
      nop.m 999
(p11) fmpy.s0 FR_Result = FR_Input_Y,FR_Input_X  // If y=qnan, x not nan or 1,
                                                 //   result=qnan
(p11) br.ret.spnt b0                             // Exit y=qnan, x not nan or 1,
                                                 //   result=qnan
}
;;

{ .mbb
      nop.m 999
(p14) br.cond.spnt POWL_64_Y_IS_INF           // Branch if y=inf, x not 1 or nan
(p13) br.cond.spnt POWL_64_X_IS_INF           // Branch if x=inf, y not 1 or nan
}
;;


POWL_64_X_IS_ZERO:
// Here if x=0, y not nan or 1 or inf or 0

// There is logic starting here to determine if y is an integer when x = 0.
// If 0 < |y| < 1 then clearly y is not an integer.
// If |y| > 1, then the significand of y is shifted left by the size of
//    the exponent of y.  This preserves the lsb of the integer part + the
//    fractional bits.  The lsb of the integer can be tested to determine if
//    the integer is even or odd.  The fractional bits can be tested.  If zero,
//    then y is an integer.
//
{ .mfi
      and GR_exp_y = GR_exp_mask,GR_signexp_y   // Get biased exponent of y
      nop.f 999
      and GR_y_sign = GR_sign_mask,GR_signexp_y // Get sign of y
}
;;

//
//     Maybe y is < 1 already, so
//     can never be an integer.
//
{ .mfi
      cmp.lt  p9, p8 = GR_exp_y,GR_exp_bias     // Test 0 < |y| < 1
      nop.f 999
      sub GR_exp_y = GR_exp_y,GR_exp_bias       // Get true exponent of y
}
;;

//
//     Shift significand of y looking for nonzero bits
//     For y > 1, shift signif_y exp_y bits to the left
//     For y < 1, turn on 4 low order bits of significand of y
//     so that the fraction will always be non-zero
//
{ .mmi
(p9)  or  GR_exp_y=  0xF,GR_signif_y            // Force nonzero fraction if y<1
;;
      nop.m 999
(p8)  shl GR_exp_y=  GR_signif_y,GR_exp_y       // Get lsb of int + fraction
                                                // Wait 4 cycles to use result
}
;;

{ .mmi
      nop.m 999
;;
      nop.m 999
      nop.i 999
}
;;

{ .mmi
      nop.m 999
;;
      nop.m 999
      shl GR_fraction_y=  GR_exp_y,1            // Shift left 1 to get fraction
}
;;

//
//     Integer part of y  shifted off.
//     Get y's low even or odd bit - y might not be an int.
//
{ .mii
      cmp.eq  p13,p0  =  GR_fraction_y, r0      // Test for y integer
      cmp.eq  p8,p0 =  GR_y_sign, r0            // Test for y > 0
;;
(p13) tbit.nz.unc p13,p0 = GR_exp_y, 63         // Test if y an odd integer
}
;;

{ .mfi
(p13) cmp.eq.unc p13,p14 =  GR_y_sign, r0   // Test y pos odd integer
(p8)  fcmp.eq.s0 p12,p0 = FR_Input_Y, f0    // If x=0 and y>0 flag if y denormal
      nop.i 999
}
;;

//
//     Return +/-0 when x=+/-0 and y is positive odd integer
//
{ .mfb
      nop.m 999
(p13) mov FR_Result = FR_Input_X            // If x=0,  y pos odd int, result=x
(p13) br.ret.spnt b0                        // Exit x=0, y pos odd int, result=x
}
;;

//
//     Return +/-inf when x=+/-0 and y is negative odd int
//
{ .mfb
(p14) mov GR_Parameter_TAG = 21
(p14) frcpa.s0 FR_Result, p0 = f1, FR_Input_X  // Result +-inf, set Z flag
(p14) br.cond.spnt __libm_error_region
}
;;

//
//     Return +0 when x=+/-0 and y positive and not an odd integer
//
{ .mfb
      nop.m 999
(p8)  mov FR_Result = f0      // If x=0, y>0 and not odd integer, result=+0
(p8)  br.ret.sptk b0          // Exit x=0, y>0 and not odd integer, result=+0
}
;;

//
//     Return +inf when x=+/-0 and y is negative and not odd int
//
{ .mfb
      mov GR_Parameter_TAG = 21
      frcpa.s0 FR_Result, p10 = f1,f0   // Result +inf, raise Z flag
      br.cond.sptk __libm_error_region
}
;;


POWL_64_X_IS_INF:
//
// Here if x=inf, y not 1 or nan
//
{ .mfi
      and GR_exp_y = GR_exp_mask,GR_signexp_y   // Get biased exponent y
      fclass.m p13, p0 =  FR_Input_X,0x022      // Test x=-inf
      nop.i 999
}
;;

{ .mfi
      and GR_y_sign = GR_sign_mask,GR_signexp_y // Get sign of y
      fcmp.eq.s0 p9,p0 = FR_Input_Y, f0         // Dummy to set flag if y denorm
      nop.i 999
}
;;

//
//     Maybe y is < 1 already, so
//     isn't an int.
//
{ .mfi
(p13) cmp.lt.unc  p9, p8 = GR_exp_y,GR_exp_bias // Test 0 < |y| < 1 if x=-inf
      fclass.m p11, p0 =  FR_Input_X,0x021      // Test x=+inf
      sub GR_exp_y = GR_exp_y,GR_exp_bias       // Get true exponent y
}
;;

//
//     Shift significand of y looking for nonzero bits
//     For y > 1, shift signif_y exp_y bits to the left
//     For y < 1, turn on 4 low order bits of significand of y
//     so that the fraction will always be non-zero
//
{ .mmi
(p9)  or  GR_exp_y=  0xF,GR_signif_y          // Force nonzero fraction if y<1
;;
(p11) cmp.eq.unc  p14,p12 = GR_y_sign, r0     // Test x=+inf, y>0
(p8)  shl GR_exp_y=  GR_signif_y,GR_exp_y     // Get lsb of int + fraction
                                              // Wait 4 cycles to use result
}
;;

//
//     Return +inf for x=+inf, y > 0
//     Return +0   for x=+inf, y < 0
//
{ .mfi
      nop.m 999
(p12) mov FR_Result = f0                      // If x=+inf, y<0, result=+0
      nop.i 999
}
{ .mfb
      nop.m 999
(p14) fma.s0 FR_Result = FR_Input_X,f1,f0     // If x=+inf, y>0, result=+inf
(p11) br.ret.sptk b0                          // Exit x=+inf
}
;;

//
// Here only if x=-inf.  Wait until can use result of shl...
//
{ .mmi
      nop.m 999
;;
      nop.m 999
      nop.i 999
}
;;

{ .mfi
      cmp.eq  p8,p9 = GR_y_sign, r0           // Test y pos
      nop.f 999
      shl GR_fraction_y = GR_exp_y,1          // Shift left 1 to get fraction
}
;;

{ .mmi
      cmp.eq  p13,p0 = GR_fraction_y, r0      // Test y integer
;;
      nop.m 999
(p13) tbit.nz.unc  p13,p0 = GR_exp_y, 63      // Test y odd integer
}
;;

//
//     Is y even or odd?
//
{ .mii
(p13) cmp.eq.unc  p14,p10 = GR_y_sign, r0     // Test x=-inf, y pos odd int
(p13) cmp.ne.and  p8,p9 = r0,r0               // If y odd int, turn off p8,p9
      nop.i 999
}
;;

//
//     Return -0   for x = -inf and y < 0 and odd int.
//     Return -Inf for x = -inf and y > 0 and odd int.
//
{ .mfi
      nop.m 999
(p10) fmerge.ns FR_Result = f0, f0      // If x=-inf, y neg odd int, result=-0
      nop.i 999
}
{ .mfi
      nop.m 999
(p14) fmpy.s0 FR_Result = FR_Input_X,f1 // If x=-inf, y pos odd int, result=-inf
      nop.i 999
}
;;

//
//     Return Inf for x = -inf and y > 0 not an odd int.
//     Return +0  for x = -inf and y < 0 not an odd int.
//
.pred.rel "mutex",p8,p9
{ .mfi
      nop.m 999
(p8)  fmerge.ns FR_Result = FR_Input_X, FR_Input_X // If x=-inf, y>0 not odd int
                                                   //   result=+inf
      nop.i 999
}
{ .mfb
      nop.m 999
(p9)  fmpy.s0 FR_Result = f0,f0                    // If x=-inf, y<0 not odd int
                                                   //   result=+0
      br.ret.sptk b0                               // Exit for x=-inf
}
;;


POWL_64_Y_IS_INF:
// Here if y=inf, x not 1 or nan
//
//     For y = +Inf and |x| < 1  returns 0
//     For y = +Inf and |x| > 1  returns Inf
//     For y = -Inf and |x| < 1  returns Inf
//     For y = -Inf and |x| > 1  returns 0
//     For y =  Inf and |x| = 1  returns 1
//
{ .mfi
      nop.m 999
      fclass.m p8, p0 =  FR_Input_Y, 0x021    // Test y=+inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p9, p0 =  FR_Input_Y, 0x022    // Test y=-inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fabs FR_X = FR_Input_X                  // Form |x|
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcmp.eq.s0 p10,p0 = FR_Input_X, f0      // flag if x denormal
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fcmp.lt.unc.s1 p6, p0  =  FR_X, f1      // Test y=+inf, |x|<1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fcmp.gt.unc.s1 p7, p0  =  FR_X, f1      // Test y=+inf, |x|>1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fcmp.lt.unc.s1 p12, p0 =  FR_X, f1      // Test y=-inf, |x|<1
      nop.i 999
}
{ .mfi
      nop.m 999
(p6)  fmpy.s0 FR_Result = f0,f0               // If y=+inf, |x|<1, result=+0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fcmp.gt.unc.s1 p13, p0 =  FR_X, f1      // Test y=-inf, |x|>1
      nop.i 999
}
{ .mfi
      nop.m 999
(p7)  fmpy.s0 FR_Result = FR_Input_Y, f1      // If y=+inf, |x|>1, result=+inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcmp.eq.s1 p14, p0 =  FR_X, f1          // Test y=inf, |x|=1
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fnma.s0 FR_Result = FR_Input_Y, f1, f0  // If y=-inf, |x|<1, result=+inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p13) mov FR_Result = f0                      // If y=-inf, |x|>1, result=+0
      nop.i 999
}
;;

{ .mfb
      nop.m 999
(p14) fmpy.s0 FR_Result = f1,f1               // If y=inf, |x|=1, result=+1
      br.ret.sptk b0                          // Common return for y=inf
}
;;


// Here if x or y denorm/unorm
POWL_DENORM:
{ .mmi
      getf.sig GR_signif_Z = FR_norm_X   // Get significand of x
;;
      getf.exp GR_signexp_y = FR_norm_Y  // Get sign and exp of y
      nop.i 999
}
;;

{ .mfi
      getf.sig GR_signif_y = FR_norm_Y   // Get significand of y
      nop.f 999
      nop.i 999
}
;;

{ .mib
      getf.exp GR_signexp_x = FR_norm_X  // Get sign and exp of x
      extr.u GR_Index1 = GR_signif_Z, 59, 4  // Extract upper 4 signif bits of x
      br.cond.sptk  POWL_COMMON          // Branch back to main path
}
;;


POWL_64_UNSUPPORT:
//
//     Raise exceptions for specific
//     values - pseudo NaN and
//     infinities.
//     Return NaN and raise invalid
//
{ .mfb
      nop.m 999
      fmpy.s0 FR_Result = FR_Input_X,f0
      br.ret.sptk b0
}
;;

POWL_64_XNEG:
//
//     Raise invalid for x < 0  and
//     y not an integer
//
{ .mfi
      nop.m 999
      frcpa.s0 FR_Result, p8 =  f0, f0
      mov GR_Parameter_TAG = 22
}
{ .mib
      nop.m 999
      nop.i 999
      br.cond.sptk __libm_error_region
}
;;

POWL_64_SQRT:
{ .mfi
      nop.m 999
      frsqrta.s0 FR_Result,p10 = FR_save_Input_X
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f62=FR_Half,FR_save_Input_X,f0
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f63=FR_Result,FR_Result,f0
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fnma.s1  f32=f63,f62,FR_Half
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f33=f32,FR_Result,FR_Result
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f34=f33,f62,f0
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fnma.s1  f35=f34,f33,FR_Half
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f63=f35,f33,f33
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f32=FR_save_Input_X,f63,f0
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fma.s1   FR_Result=f63,f62,f0
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f33=f11,f63,f0
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fnma.s1  f34=f32,f32,FR_save_Input_X
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fnma.s1  f35=FR_Result,f63,FR_Half
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fma.s1   f62=f33,f34,f32
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fma.s1   f63=f33,f35,f33
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10) fnma.s1  f32=f62,f62,FR_save_Input_X
      nop.i 999 ;;
}
{ .mfb
      nop.m 999
(p10) fma.s0 FR_Result=f32,f63,f62
      br.ret.sptk   b0                // Exit for x > 0, y = 0.5
}
;;

GLOBAL_LIBM_END(powl)
libm_alias_ldouble_other (pow, pow)


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
        stfe [GR_Parameter_Y] = FR_Input_Y,16   // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfe [GR_Parameter_X] = FR_save_Input_X // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfe [GR_Parameter_Y] = FR_Result       // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
};;
{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;
{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]        // Get return result off stack
.restore sp
        add   sp = 64,sp                        // Restore stack pointer
        mov   b0 = GR_SAVE_B0                   // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
};;

LOCAL_LIBM_END(__libm_error_region#)
.type   __libm_error_support#,@function
.global __libm_error_support#
