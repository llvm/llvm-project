.file "fmodl.s"


// Copyright (c) 2000 - 2004, Intel Corporation
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
//====================================================================
// 02/02/00 Initial version
// 03/02/00 New Algorithm
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
// set [ the previously overwritten ] GR_Parameter_RESULT.
// 11/28/00 Set FR_Y to f9
// 03/11/02 Fixed flags for fmodl(qnan, zero)
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header:.section,.global,.proc,.align
// 04/28/03 Fix: fmod(sNaN, 0) no longer sets errno
// 11/23/04 Reformatted routine and improved speed
//
// API
//====================================================================
// long double fmodl(long double, long double);
//
// Overview of operation
//====================================================================
// fmod(a, b)= a-i*b,
// where i is an integer such that, if b!= 0,
// |i|<|a/b| and |a/b-i|<1
//
// Algorithm
//====================================================================
// a). if |a|<|b|, return a
// b). get quotient and reciprocal overestimates accurate to
// 33 bits (q2, y2)
// c). if the exponent difference (exponent(a)-exponent(b))
// is less than 32, truncate quotient to integer and
// finish in one iteration
// d). if exponent(a)-exponent(b)>= 32 (q2>= 2^32)
// round quotient estimate to single precision (k= RN(q2)),
// calculate partial remainder (a'= a-k*b),
// get quotient estimate (a'*y2), and repeat from c).
//
// Registers used
//====================================================================

GR_SMALLBIASEXP     = r2
GR_2P32             = r3
GR_SMALLBIASEXP     = r20
GR_ROUNDCONST       = r21
GR_SIG_B            = r22
GR_ARPFS            = r23
GR_TMP1             = r24
GR_TMP2             = r25
GR_TMP3             = r26

GR_SAVE_B0          = r33
GR_SAVE_PFS         = r34
GR_SAVE_GP          = r35
GR_SAVE_SP          = r36

GR_Parameter_X      = r37
GR_Parameter_Y      = r38
GR_Parameter_RESULT = r39
GR_Parameter_TAG    = r40

FR_X                = f10
FR_Y                = f9
FR_RESULT           = f8

FR_ABS_A            = f6
FR_ABS_B            = f7
FR_Y_INV            = f10
FR_SMALLBIAS        = f11
FR_E0               = f12
FR_Q                = f13
FR_E1               = f14
FR_2P32             = f15
FR_TMPX             = f32
FR_TMPY             = f33
FR_ROUNDCONST       = f34
FR_QINT             = f35
FR_QRND24           = f36
FR_NORM_B           = f37
FR_TMP              = f38
FR_TMP2             = f39
FR_DFLAG            = f40
FR_Y_INV0           = f41
FR_Y_INV1           = f42
FR_Q0               = f43
FR_Q1               = f44
FR_QINT_Z           = f45
FR_QREM             = f46
FR_B_SGN_A          = f47

.section .text
GLOBAL_IEEE754_ENTRY(fmodl)

// inputs in f8, f9
// result in f8

{ .mfi
       getf.sig GR_SIG_B = f9
       // FR_ABS_A = |a|
       fmerge.s FR_ABS_A = f0, f8
       mov GR_SMALLBIASEXP = 0x0ffdd
}
{ .mfi
       nop.m 0
       // FR_ABS_B = |b|
       fmerge.s FR_ABS_B = f0, f9
       nop.i 0
}
;;

{ .mfi
       setf.exp FR_SMALLBIAS = GR_SMALLBIASEXP
       // (1) y0
       frcpa.s1 FR_Y_INV0, p6 = FR_ABS_A, FR_ABS_B
       nop.i 0
}
;;

{ .mlx
       nop.m 0
       movl GR_ROUNDCONST = 0x33a00000
}
;;

// eliminate special cases
{ .mmi
       nop.m 0
       nop.m 0
       // y pseudo-zero ?
       cmp.eq p7, p10 = GR_SIG_B, r0
}
;;

// set p7 if b +/-NAN, +/-inf, +/-0
{ .mfi
       nop.m 0
 (p10) fclass.m p7, p10 = f9, 0xe7
       nop.i 0
}
;;

{ .mfi
       mov GR_2P32 = 0x1001f
       // (2) q0 = a*y0
 (p6)  fma.s1 FR_Q0 = FR_ABS_A, FR_Y_INV0, f0
       nop.i 0
}
{ .mfi
       nop.m 0
       // (3) e0 = 1 - b * y0
 (p6)  fnma.s1 FR_E0 = FR_ABS_B, FR_Y_INV0, f1
       nop.i 0
}
;;

// set p9 if a +/-NAN, +/-inf
{ .mfi
       nop.m 0
       fclass.m.unc p9, p11 = f8, 0xe3
       nop.i 0
}
       // |a| < |b|? Return a, p8=1
{ .mfi
       nop.m 0
 (p10) fcmp.lt.unc.s1 p8, p0 = FR_ABS_A, FR_ABS_B
       nop.i 0
}
;;

// set p7 if b +/-NAN, +/-inf, +/-0
{ .mfi
       nop.m 0
       // pseudo-NaN ?
 (p10) fclass.nm p7, p0 = f9, 0xff
       nop.i 0
}
;;

// set p9 if a is +/-NaN, +/-Inf
{ .mfi
       nop.m 0
 (p11) fclass.nm p9, p0 = f8, 0xff
       nop.i 0
}
{ .mfi
       nop.m 0
       // b denormal ? set D flag (if |a|<|b|)
 (p8)  fnma.s0 FR_DFLAG = f9, f1, f9
       nop.i 0
}
;;

{ .mfi
       // FR_2P32 = 2^32
       setf.exp FR_2P32 = GR_2P32
       // (4) q1 = q0+e0*q0
 (p6)  fma.s1 FR_Q1 = FR_E0, FR_Q0, FR_Q0
       nop.i 0
}
{ .mfi
       nop.m 0
       // (5) e1 = e0 * e0 + 2^-34
 (p6)  fma.s1 FR_E1 = FR_E0, FR_E0, FR_SMALLBIAS
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // normalize a (if |a|<|b|)
 (p8)  fma.s0 f8 = f8, f1, f0
       nop.i 0
}
{ .bbb
 (p9) br.cond.spnt FMOD_A_NAN_INF
 (p7) br.cond.spnt FMOD_B_NAN_INF_ZERO
       // if |a|<|b|, return
 (p8) br.ret.spnt b0
}
;;


{ .mfi
       nop.m 0
       // (6) y1 = y0 + e0 * y0
 (p6)  fma.s1 FR_Y_INV1 = FR_E0, FR_Y_INV0, FR_Y_INV0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // a denormal ? set D flag
       // b denormal ? set D flag
       fcmp.eq.s0 p12,p0 = FR_ABS_A, FR_ABS_B
       nop.i 0
}
{ .mfi
       // set FR_ROUNDCONST = 1.25*2^{-24}
       setf.s FR_ROUNDCONST = GR_ROUNDCONST
       // (7) q2 = q1+e1*q1
 (p6)  fma.s1 FR_Q = FR_Q1, FR_E1, FR_Q1
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       fmerge.s FR_B_SGN_A = f8, f9
       nop.i 0
}
{ .mfi
       nop.m 0
       // (8) y2 = y1 + e1 * y1
 (p6)  fma.s1 FR_Y_INV = FR_E1, FR_Y_INV1, FR_Y_INV1
       // set p6 = 0, p10 = 0
       cmp.ne.and p6, p10 = r0, r0
}
;;

//   will compute integer quotient bits (24 bits per iteration)
.align 32
loop64:
{ .mfi
       nop.m 0
       // compare q2, 2^32
       fcmp.lt.unc.s1 p8, p7 = FR_Q, FR_2P32
       nop.i 0
}
{ .mfi
       nop.m 0
       // will truncate quotient to integer, if exponent<32 (in advance)
       fcvt.fx.trunc.s1 FR_QINT = FR_Q
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // if exponent>32 round quotient to single precision (perform in advance)
       fma.s.s1 FR_QRND24 = FR_Q, f1, f0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // set FR_ROUNDCONST = sgn(a)
 (p8)  fmerge.s FR_ROUNDCONST = f8, f1
       nop.i 0
}
{ .mfi
       nop.m 0
       // normalize truncated quotient
 (p8)  fcvt.xf FR_QRND24 = FR_QINT
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // calculate remainder (assuming FR_QRND24 = RZ(Q))
 (p7)  fnma.s1 FR_E1 = FR_QRND24, FR_ABS_B, FR_ABS_A
       nop.i 0
}
{ .mfi
       nop.m 0
       // also if exponent>32, round quotient to single precision
       // and subtract 1 ulp: q = q-q*(1.25*2^{-24})
 (p7)  fnma.s.s1 FR_QINT_Z = FR_QRND24, FR_ROUNDCONST, FR_QRND24
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // (p8) calculate remainder (82-bit format)
 (p8)  fnma.s1 FR_QREM = FR_QRND24, FR_ABS_B, FR_ABS_A
       nop.i 0
}
{ .mfi
       nop.m 0
       // (p7) calculate remainder (assuming FR_QINT_Z = RZ(Q))
 (p7)  fnma.s1 FR_ABS_A = FR_QINT_Z, FR_ABS_B, FR_ABS_A
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // Final iteration (p8): is FR_ABS_A the correct remainder
       // (quotient was not overestimated) ?
 (p8)  fcmp.lt.unc.s1 p6, p10 = FR_QREM, f0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // get new quotient estimation: a'*y2
 (p7)  fma.s1 FR_Q = FR_E1, FR_Y_INV, f0
       nop.i 0
}
{ .mfb
       nop.m 0
       // was FR_Q = RZ(Q) ? (then new remainder FR_E1> = 0)
 (p7)  fcmp.lt.unc.s1 p7, p9 = FR_E1, f0
       nop.b 0
}
;;

.pred.rel "mutex", p6, p10
{ .mfb
       nop.m 0
       // add b to estimated remainder (to cover the case when the quotient was
       // overestimated)
       // also set correct sign by using
       // FR_B_SGN_A = |b|*sgn(a), FR_ROUNDCONST = sgn(a)
 (p6)  fma.s0 f8 = FR_QREM, FR_ROUNDCONST, FR_B_SGN_A
       nop.b 0
}
{ .mfb
       nop.m 0
       // set correct sign of result before returning: FR_ROUNDCONST = sgn(a)
 (p10) fma.s0 f8 = FR_QREM, FR_ROUNDCONST, f0
 (p8)  br.ret.sptk b0
}
;;

{ .mfi
       nop.m 0
       // if f13! = RZ(Q), get alternative quotient estimation: a''*y2
 (p7)  fma.s1 FR_Q = FR_ABS_A, FR_Y_INV, f0
       nop.i 0
}
{ .mfb
       nop.m 0
       // if FR_E1 was RZ(Q), set remainder to FR_E1
 (p9)  fma.s1 FR_ABS_A = FR_E1, f1, f0
       br.cond.sptk loop64
}
;;

FMOD_A_NAN_INF:

// b zero ?
{ .mfi
       nop.m 0
       fclass.m p10, p0 = f8, 0xc3 // Test a = nan
       nop.i 0
}
{ .mfi
       nop.m 0
       fma.s1 FR_NORM_B = f9, f1, f0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       fma.s0 f8 = f8, f1, f0
       nop.i 0
}
{ .mfi
       nop.m 0
 (p10) fclass.m p10, p0 = f9, 0x07 // Test x = nan, and y = zero
       nop.i 0
}
;;

{ .mfb
       nop.m 0
       fcmp.eq.unc.s1 p11, p0 = FR_NORM_B, f0
 (p10) br.ret.spnt b0 // Exit with result = a if a = nan and b = zero
}
;;

{ .mib
       nop.m 0
       nop.i 0
       // if Y zero
 (p11) br.cond.spnt FMOD_B_ZERO
}
;;

// a= infinity? Return QNAN indefinite
{ .mfi
       // set p7 t0 0
       cmp.ne p7, p0 = r0, r0
       fclass.m.unc p8, p9 = f8, 0x23
       nop.i 0
}
;;

// b NaN ?
{ .mfi
       nop.m 0
 (p8)  fclass.m p9, p8 = f9, 0xc3
       nop.i 0
}
;;

// b not pseudo-zero ? (GR_SIG_B holds significand)
{ .mii
       nop.m 0
 (p8)  cmp.ne p7, p0 = GR_SIG_B, r0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
 (p8)  frcpa.s0 f8, p0 = f8, f8
       nop.i 0
}
{ .mfi
       nop.m 0
       // also set Denormal flag if necessary
 (p7)  fnma.s0 f9 = f9, f1, f9
       nop.i 0
}
;;

{ .mfb
       nop.m 0
 (p8)  fma.s0 f8 = f8, f1, f0
       nop.b 0
}
;;

{ .mfb
       nop.m 0
 (p9)  frcpa.s0 f8, p7 = f8, f9
       br.ret.sptk b0
}
;;

FMOD_B_NAN_INF_ZERO:
// b INF
{ .mfi
       nop.m 0
       fclass.m.unc p7, p0 = f9, 0x23
       nop.i 0
}
;;

{ .mfb
       nop.m 0
 (p7)  fma.s0 f8 = f8, f1, f0
 (p7)  br.ret.spnt b0
}
;;

// b NAN?
{ .mfi
       nop.m 0
       fclass.m.unc p9, p10 = f9, 0xc3
       nop.i 0
}
;;

{ .mfi
       nop.m 0
 (p10) fclass.nm p9, p0 = f9, 0xff
       nop.i 0
}
;;

{ .mfb
       nop.m 0
 (p9)  fma.s0 f8 = f9, f1, f0
 (p9)  br.ret.spnt b0
}
;;

FMOD_B_ZERO:
// Y zero? Must be zero at this point
// because it is the only choice left.
// Return QNAN indefinite

{ .mfi
       nop.m 0
       // set Invalid
       frcpa.s0 FR_TMP, p0 = f0, f0
       nop.i 0
}
;;

// a NAN?
{ .mfi
       nop.m 0
       fclass.m.unc p9, p10 = f8, 0xc3
       nop.i 0
}
;;

{ .mfi
       alloc GR_ARPFS = ar.pfs, 1, 4, 4, 0
 (p10) fclass.nm p9, p10 = f8, 0xff
       nop.i 0
}
;;

{ .mfi
       nop.m 0
 (p9)  frcpa.s0 FR_TMP2, p7 = f8, f0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
 (p10) frcpa.s0 FR_TMP2, p7 = f9, f9
       mov GR_Parameter_TAG = 120
}
;;

{ .mfi
       nop.m 0
       fmerge.s FR_X = f8, f8
       nop.i 0
}
{ .mfb
       nop.m 0
       fma.s0 f8 = FR_TMP2, f1, f0
       br.sptk __libm_error_region
}
;;

GLOBAL_IEEE754_END(fmodl)
libm_alias_ldouble_other (__fmod, fmod)

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
       add GR_Parameter_Y = -32, sp // Parameter 2 value
       nop.f 0
.save ar.pfs, GR_SAVE_PFS
       mov GR_SAVE_PFS = ar.pfs     // Save ar.pfs
}
{ .mfi
.fframe 64
       add sp = -64, sp             // Create new stack
       nop.f 0
       mov GR_SAVE_GP = gp          // Save gp
}
;;

{ .mmi
       stfe [ GR_Parameter_Y ] = FR_Y, 16 // Save Parameter 2 on stack
       add GR_Parameter_X = 16, sp  // Parameter 1 address
.save b0, GR_SAVE_B0
       mov GR_SAVE_B0 = b0          // Save b0
}
;;

.body
{ .mib
       stfe [ GR_Parameter_X ] = FR_X // Store Parameter 1 on stack
       add GR_Parameter_RESULT = 0, GR_Parameter_Y
       nop.b 0                      // Parameter 3 address
}
{ .mib
       stfe [ GR_Parameter_Y ] = FR_RESULT // Store Parameter 3 on stack
       add GR_Parameter_Y = -16, GR_Parameter_Y
       br.call.sptk b0 = __libm_error_support# // Call error handling function
}
;;

{ .mmi
       nop.m 0
       nop.m 0
       add GR_Parameter_RESULT = 48, sp
}
;;

{ .mmi
       ldfe f8 = [ GR_Parameter_RESULT ] // Get return result off stack
.restore sp
       add sp = 64, sp                   // Restore stack pointer
       mov b0 = GR_SAVE_B0               // Restore return address
}
;;

{ .mib
       mov gp = GR_SAVE_GP               // Restore gp
       mov ar.pfs = GR_SAVE_PFS          // Restore ar.pfs
       br.ret.sptk b0                    // Return
}
;;

LOCAL_LIBM_END(__libm_error_region)

.type __libm_error_support#, @function
.global __libm_error_support#
