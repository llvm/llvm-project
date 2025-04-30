.file "fmod.s"


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
// History
//====================================================================
// 02/02/00 Initial version
// 03/02/00 New Algorithm
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 11/28/00 Set FR_Y to f9
// 03/11/02 Fixed flags for fmod(qnan,zero)
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 04/28/03 Fix: fmod(sNaN,0) no longer sets errno
//
// API
//====================================================================
// double fmod(double,double);
//
// Overview of operation
//====================================================================
//  fmod(a,b)=a-i*b,
//  where i is an integer such that, if b!=0,
//  |i|<|a/b| and |a/b-i|<1
//
// Algorithm
//====================================================================
// a). if |a|<|b|, return a
// b). get quotient and reciprocal overestimates accurate to
//     33 bits (q2,y2)
// c). if the exponent difference (exponent(a)-exponent(b))
//     is less than 32, truncate quotient to integer and
//     finish in one iteration
// d). if exponent(a)-exponent(b)>=32 (q2>=2^32)
//     round quotient estimate to single precision (k=RN(q2)),
//     calculate partial remainder (a'=a-k*b),
//     get quotient estimate (a'*y2), and repeat from c).
//
// Special cases
//====================================================================
// b=+/-0: return NaN, call libm_error_support
// a=+/-Inf, a=NaN or b=NaN: return NaN
//
// Registers used
//====================================================================
// Predicate registers: p6-p11
// General registers:   r2,r29,r32 (ar.pfs), r33-r39
// Floating point registers: f6-f15

GR_SAVE_B0                    = r33
GR_SAVE_PFS                   = r34
GR_SAVE_GP                    = r35
GR_SAVE_SP                    = r36

GR_Parameter_X                = r37
GR_Parameter_Y                = r38
GR_Parameter_RESULT           = r39
GR_Parameter_TAG              = r40

FR_X             = f10
FR_Y             = f9
FR_RESULT        = f8


.section .text
GLOBAL_IEEE754_ENTRY(fmod)

// inputs in f8, f9
// result in f8

{ .mfi
  alloc r32=ar.pfs,1,4,4,0
  // f6=|a|
  fmerge.s f6=f0,f8
  mov r2 = 0x0ffdd
}
  {.mfi
  nop.m 0
  // f7=|b|
  fmerge.s f7=f0,f9
  nop.i 0;;
}

{ .mfi
  setf.exp f11 = r2
  // (1) y0
  frcpa.s1 f10,p6=f6,f7
  nop.i 0
}

// Y +-NAN, +-inf, +-0?     p7
{ .mfi
      nop.m 999
      fclass.m.unc  p7,p0 = f9, 0xe7
      nop.i 999;;
}

// qnan snan inf norm     unorm 0 -+
// 1    1    1   0        0     0 11
// e                      3
// X +-NAN, +-inf, ?        p9

{ .mfi
      nop.m 999
      fclass.m.unc  p9,p0 = f8, 0xe3
      nop.i 999
}

// |x| < |y|? Return x p8
{ .mfi
      nop.m 999
      fcmp.lt.unc.s1 p8,p0 = f6,f7
      nop.i 999 ;;
}

{ .mfi
  nop.m 0
  // normalize y (if |x|<|y|)
  (p8) fma.s0 f9=f9,f1,f0
  nop.i 0;;
}

  { .mfi
  mov r2=0x1001f
  // (2) q0=a*y0
  (p6) fma.s1 f13=f6,f10,f0
  nop.i 0
}
{ .mfi
  nop.m 0
  // (3) e0 = 1 - b * y0
  (p6) fnma.s1 f12=f7,f10,f1
  nop.i 0;;
}

  {.mfi
  nop.m 0
  // normalize x (if |x|<|y|)
  (p8) fma.d.s0 f8=f8,f1,f0
  nop.i 0
}
{.bbb
  (p9) br.cond.spnt FMOD_X_NAN_INF
  (p7) br.cond.spnt FMOD_Y_NAN_INF_ZERO
  // if |x|<|y|, return
  (p8) br.ret.spnt    b0;;
}

  {.mfi
  nop.m 0
  // normalize x
  fma.s0 f6=f6,f1,f0
  nop.i 0
}
{.mfi
  nop.m 0
  // normalize y
  fma.s0 f7=f7,f1,f0
  nop.i 0;;
}

  {.mfi
  // f15=2^32
  setf.exp f15=r2
  // (4) q1=q0+e0*q0
  (p6) fma.s1 f13=f12,f13,f13
  nop.i 0
}
{ .mfi
  nop.m 0
  // (5) e1 = e0 * e0 + 2^-34
  (p6) fma.s1 f14=f12,f12,f11
  nop.i 0;;
}
{.mlx
  nop.m 0
  movl r2=0x33a00000;;
}
{ .mfi
  nop.m 0
  // (6) y1 = y0 + e0 * y0
  (p6) fma.s1 f10=f12,f10,f10
  nop.i 0;;
}
{.mfi
  // set f12=1.25*2^{-24}
  setf.s f12=r2
  // (7) q2=q1+e1*q1
  (p6) fma.s1 f13=f13,f14,f13
  nop.i 0;;
}
{.mfi
  nop.m 0
  fmerge.s f9=f8,f9
  nop.i 0
}
{ .mfi
  nop.m 0
  // (8) y2 = y1 + e1 * y1
  (p6) fma.s1 f10=f14,f10,f10
  // set p6=0, p10=0
  cmp.ne.and p6,p10=r0,r0;;
}

.align 32
loop53:
  {.mfi
  nop.m 0
  // compare q2, 2^32
  fcmp.lt.unc.s1 p8,p7=f13,f15
  nop.i 0
}
  {.mfi
  nop.m 0
  // will truncate quotient to integer, if exponent<32 (in advance)
  fcvt.fx.trunc.s1 f11=f13
  nop.i 0;;
}
  {.mfi
  nop.m 0
  // if exponent>32, round quotient to single precision (perform in advance)
  fma.s.s1 f13=f13,f1,f0
  nop.i 0;;
}
  {.mfi
  nop.m 0
  // set f12=sgn(a)
  (p8) fmerge.s f12=f8,f1
  nop.i 0
}
  {.mfi
  nop.m 0
  // normalize truncated quotient
  (p8) fcvt.xf f13=f11
  nop.i 0;;
}
  { .mfi
  nop.m 0
  // calculate remainder (assuming f13=RZ(Q))
  (p7) fnma.s1 f14=f13,f7,f6
  nop.i 0
}
  {.mfi
  nop.m 0
  // also if exponent>32, round quotient to single precision
  // and subtract 1 ulp: q=q-q*(1.25*2^{-24})
  (p7) fnma.s.s1 f11=f13,f12,f13
  nop.i 0;;
}

  {.mfi
  nop.m 0
  // (p8) calculate remainder (82-bit format)
  (p8) fnma.s1 f11=f13,f7,f6
  nop.i 0
}
  {.mfi
  nop.m 0
  // (p7) calculate remainder (assuming f11=RZ(Q))
  (p7) fnma.s1 f6=f11,f7,f6
  nop.i 0;;
}


  {.mfi
  nop.m 0
  // Final iteration (p8): is f6 the correct remainder (quotient was not overestimated) ?
  (p8) fcmp.lt.unc.s1 p6,p10=f11,f0
  nop.i 0;;
}
  {.mfi
  nop.m 0
  // get new quotient estimation: a'*y2
  (p7) fma.s1 f13=f14,f10,f0
  nop.i 0
}
  {.mfb
  nop.m 0
  // was f14=RZ(Q) ? (then new remainder f14>=0)
  (p7) fcmp.lt.unc.s1 p7,p9=f14,f0
  nop.b 0;;
}


.pred.rel "mutex",p6,p10
  {.mfb
  nop.m 0
  // add b to estimated remainder (to cover the case when the quotient was overestimated)
  // also set correct sign by using f9=|b|*sgn(a), f12=sgn(a)
  (p6) fma.d.s0 f8=f11,f12,f9
  nop.b 0
}
  {.mfb
  nop.m 0
  // calculate remainder (single precision)
  // set correct sign of result before returning
  (p10) fma.d.s0 f8=f11,f12,f0
  (p8) br.ret.sptk b0;;
}
  {.mfi
  nop.m 0
  // if f13!=RZ(Q), get alternative quotient estimation: a''*y2
  (p7) fma.s1 f13=f6,f10,f0
  nop.i 0
}
  {.mfb
  nop.m 0
  // if f14 was RZ(Q), set remainder to f14
  (p9) mov f6=f14
  br.cond.sptk loop53;;
}



FMOD_X_NAN_INF:

// Y zero ?
{.mfi
  nop.m 0
  fclass.m p10,p0=f8,0xc3     // Test x=nan
  nop.i 0
}
{.mfi
  nop.m 0
  fma.s1 f10=f9,f1,f0
  nop.i 0;;
}

{.mfi
  nop.m 0
  fma.s0 f8=f8,f1,f0
  nop.i 0
}
{.mfi
  nop.m 0
(p10) fclass.m p10,p0=f9,0x07 // Test x=nan, and y=zero
  nop.i 0;;
}

{.mfb
 nop.m 0
 fcmp.eq.unc.s1 p11,p0=f10,f0
(p10) br.ret.spnt b0;;        // Exit with result=x if x=nan and y=zero
}
{.mib
  nop.m 0
  nop.i 0
  // if Y zero
  (p11) br.cond.spnt FMOD_Y_ZERO;;
}

// X infinity? Return QNAN indefinite
{ .mfi
      nop.m 999
      fclass.m.unc  p8,p9 = f8, 0x23
      nop.i 999;;
}
// Y NaN ?
{.mfi
     nop.m 999
(p8) fclass.m p9,p8=f9,0xc3
     nop.i 0;;
}
{.mfi
      nop.m 999
(p8)  frcpa.s0 f8,p0 = f8,f8
      nop.i 0
}
{ .mfi
      nop.m 999
    // also set Denormal flag if necessary
(p8)  fma.s0 f9=f9,f1,f0
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
(p8)  fma.d.s0 f8=f8,f1,f0
      nop.b 999 ;;
}

{ .mfb
      nop.m 999
(p9)  frcpa.s0 f8,p7=f8,f9
      br.ret.sptk   b0 ;;
}


FMOD_Y_NAN_INF_ZERO:

// Y INF
{ .mfi
      nop.m 999
      fclass.m.unc  p7,p0 = f9, 0x23
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
(p7)  fma.d.s0 f8=f8,f1,f0
(p7)  br.ret.spnt    b0 ;;
}

// Y NAN?
{ .mfi
      nop.m 999
      fclass.m.unc  p9,p0 = f9, 0xc3
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
(p9)  fma.d.s0 f8=f9,f1,f0
(p9)  br.ret.spnt    b0 ;;
}

FMOD_Y_ZERO:
// Y zero? Must be zero at this point
// because it is the only choice left.
// Return QNAN indefinite

{.mfi
  nop.m 0
  // set Invalid
  frcpa.s0 f12,p0=f0,f0
  nop.i 0
}
// X NAN?
{ .mfi
      nop.m 999
      fclass.m.unc  p9,p10 = f8, 0xc3
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p10)  fclass.nm  p9,p10 = f8, 0xff
      nop.i 999 ;;
}

{.mfi
 nop.m 999
 (p9) frcpa.s0 f11,p7=f8,f0
 nop.i 0;;
}

{ .mfi
      nop.m 999
(p10)  frcpa.s0         f11,p7 = f9,f9
      mov        GR_Parameter_TAG = 121 ;;
}

{ .mfi
      nop.m 999
      fmerge.s      f10 = f8, f8
      nop.i 999
}

{ .mfb
      nop.m 999
      fma.d.s0 f8=f11,f1,f0
      br.sptk __libm_error_region;;
}

GLOBAL_IEEE754_END(fmod)
libm_alias_double_other (__fmod, fmod)

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
        stfd [GR_Parameter_Y] = FR_Y,16         // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfd [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
    nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT      // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
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
