.file "remainderf.s"


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
// 03/02/00 New algorithm
// 04/04/00 Unwind support added
// 07/21/00 Fixed quotient=2^{24*m+23} bug
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 11/29/00 Set FR_Y to f9
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//====================================================================
// float remainderf(float,float);
//
// Overview of operation
//====================================================================
//  remainder(a,b)=a-i*b,
//  where i is an integer such that, if b!=0 and a is finite,
//  |a/b-i|<=1/2. If |a/b-i|=1/2, i is even.
//
// Algorithm
//====================================================================
// a). eliminate special cases
// b). if |a/b|<0.25 (first quotient estimate), return a
// c). use single precision divide algorithm to get quotient q
//     rounded to 24 bits of precision
// d). calculate partial remainders (using both q and q-ulp);
//     select one and RZ(a/b) based on the sign of |a|-|b|*q
// e). if the exponent difference (exponent(a)-exponent(b))
//     is less than 24 (quotient estimate<2^{24}-2), use RZ(a/b)
//     and sticky bits to round to integer; exit loop and
//     calculate final remainder
// f). if exponent(a)-exponent(b)>=24, select new value of a as
//     the partial remainder calculated using RZ(a/b);
//     repeat from c).
//
// Special cases
//====================================================================
// a=+/- Inf, or b=+/-0: return NaN, call libm_error_support
// a=NaN or b=NaN: return NaN
//
// Registers used
//====================================================================
// Predicate registers: p6-p12
// General registers:   r2,r3,r28,r29,r32 (ar.pfs), r33-r39
// Floating point registers: f6-f15
//

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
GLOBAL_IEEE754_ENTRY(remainderf)

// inputs in f8, f9
// result in f8

{ .mfi
  alloc r32=ar.pfs,1,4,4,0
  // f13=|a|
  fmerge.s f13=f0,f8
  nop.i 0
}
  {.mfi
  nop.m 0
  // f14=|b|
  fmerge.s f14=f0,f9
  nop.i 0;;
}
 {.mlx
  nop.m 0
  // r2=2^{24}-2
  movl r3=0x4b7ffffe;;
}

// Y +-NAN, +-inf, +-0?     p11
{ .mfi
      nop.m 999
      fclass.m.unc  p11,p0 = f9, 0xe7
      nop.i 999
}
// qnan snan inf norm     unorm 0 -+
// 1    1    1   0        0     0 11
// e                      3
// X +-NAN, +-inf, ?        p9
{ .mfi
      nop.m 999
      fclass.m.unc  p9,p0 = f8, 0xe3
      nop.i 999;;
}

{.mfi
  nop.m 0
  mov f15=f0
  nop.i 0
}
{ .mfi
  // set p7=1
  cmp.eq.unc p7,p0=r0,r0
  // Step (1)
  // y0 = 1 / b in f10
  frcpa.s1 f10,p6=f13,f14
  nop.i 0;;
}
{.bbb
  (p9) br.cond.spnt FREM_X_NAN_INF
  (p11) br.cond.spnt FREM_Y_NAN_INF_ZERO
  nop.b 0
}  {.mfi
   nop.m 0
   // set D flag if a (f8) is denormal
   fnma.s0 f6=f8,f1,f8
   nop.i 0;;
}

.align 32
remloop24:
  { .mfi
  // f12=2^{24}-2
  setf.s f12=r3
  // Step (2)
  // q0 = a * y0 in f15
  (p6) fma.s1 f15=f13,f10,f0
  nop.i 0
}
{ .mfi
  nop.m 0
  // Step (3)
  // e0 = 1 - b * y0 in f7
  (p6) fnma.s1 f7=f14,f10,f1
  nop.i 0;;
}
{.mlx
  nop.m 0
  // r2=1.25*2^{-24}
  movl r2=0x33a00000;;
}
  { .mfi
  nop.m 0
  // Step (4)
  // q1 = q0 + e0 * q0 in f6
  (p6) fma.s1 f6=f7,f15,f15
  nop.i 0
}
{ .mfi
  nop.m 0
  // Step (5)
  // e1 = e0 * e0 in f7
  (p6) fma.s1 f7=f7,f7,f0
  nop.i 0;;
}
 {.mii
  (p7) getf.exp r29=f15
  (p7) mov r28=0xfffd
  nop.i 0;;
}

 { .mfi
  // f15=1.25*2^{-24}
  setf.s f15=r2
  // Step (6)
  // q2 = q1 + e1 * q1 in f6
  (p6) fma.s1 f6=f7,f6,f6
  nop.i 0
}
{ .mfi
  mov r2=0x3e7
  // Step (7)
  // e2 = e1 * e1 in f7
  (p6) fma.s1 f7=f7,f7,f0
  nop.i 0;;
}

 {.mmi
  // q<1/4 ? (i.e. expon< -2)
  (p7) cmp.gt.unc p7,p0=r28,r29
  nop.m 0
  // r2=0x3e7000000
  shl r2=r2,24;;
}

{.mfb
  // r2=0x3e7000001
  add r2=1,r2
 // if |a/b|<1/4, set D flag before returning
 (p7) fma.s.s0 f9=f9,f0,f8
  nop.b 0;;
}
 {.mfb
 nop.m 0
 // can be combined with bundle above if sign of 0 or
 // FTZ enabled are not important
 (p7) fmerge.s f8=f8,f9
 // return if |a|<4*|b| (estimated quotient < 1/4)
 (p7) br.ret.spnt b0;;
}
  {.mfi
  nop.m 0
  // set f8 to current a value | sign
  fmerge.s f8=f8,f13
  // r2=2^{-24}+2^{-48} (double prec.)
  shl r2=r2,28;;
}


{ .mfi
  // r29= -32+bias
  mov r29=0xffdf
  // Step (8)
  // q3 = q2 + e2 * q2 in f6
  (p6) fma.d.s1 f6=f7,f6,f6
  nop.i 0;;
}
{ .mfi
  nop.m 0
  // Step (9)
  // q = q3 in f11
  (p6) fma.s.s1 f11=f6,f1,f0
  nop.i 0;;
}
  {.mfi
  // f7=2^{-24}
  setf.d f7=r2
  // last step ? (q3<2^{24}-2 --> q<2^{24})
  fcmp.lt.unc.s1 p0,p12=f6,f12
  nop.i 0
} {.mfi
  // f12=2^{-32}
   setf.exp f12=r29
   nop.f 0
   nop.i 0;;
}
  {.mfi
  nop.m 0
  // r=a-b*q
  fnma.s1 f6=f14,f11,f13
  nop.i 0
}
{.mfi
  nop.m 0
  // q'=q-q*(1.25*2^{-24})   (q'=q-ulp)
  fnma.s.s1 f15=f11,f15,f11
  nop.i 0;;
}

  {.mfi
  nop.m 0
  // r2=a-b*q'
  fnma.s1 f13=f14,f15,f13
  nop.i 0;;
}
  {.mfi
  nop.m 0
  // r>0 iff q=RZ(a/b) and inexact
  fcmp.gt.unc.s1 p8,p0=f6,f0
  nop.i 0
}
{.mfi
  nop.m 0
  // r<0 iff q'=RZ(a/b) and inexact
  fcmp.lt.unc.s1 p9,p10=f6,f0
  nop.i 0;;
}
.pred.rel "mutex",p8,p9
  {.mfi
  nop.m 0
  // (p8) Q=q+(last iteration ? sticky bits:0)
  // i.e. Q=q+q*x  (x=2^{-32} or 0)
  (p8) fma.s1 f11=f11,f12,f11
  nop.i 0
}
{.mfi
  nop.m 0
  // (p9) Q=q'+(last iteration ? sticky bits:0)
  // i.e. Q=q'+q'*x  (x=2^{-24} or 0: if expon. difference=23, want to round back to q)
  (p9) fma.s1 f11=f15,f7,f15
  nop.i 0;;
}

  {.mfb
  nop.m 0
  // (p9) set r=r2 (new a, if not last iteration)
  // (p10) new a =r
  (p10) mov f13=f6
  (p12) br.cond.sptk remloop24;;
}

// last iteration
  {.mfi
  nop.m 0
  // set f9=|b|*sgn(a)
  fmerge.s f9=f8,f9
  nop.i 0
}
  {.mfi
  nop.m 0
  // round to integer
  fcvt.fx.s1 f11=f11
  nop.i 0;;
}
  {.mfi
  nop.m 0
  // save sign of a
  fmerge.s f7=f8,f8
  nop.i 0
}
{.mfi
  nop.m 0
  // normalize
  fcvt.xf f11=f11
  nop.i 0;;
}
  {.mfi
  nop.m 0
  // This can be removed if sign of 0 is not important
  // get remainder using sf1
  fnma.s.s1 f12=f9,f11,f8
  nop.i 0
}
  {.mfi
  nop.m 0
  // get remainder
  fnma.s.s0 f8=f9,f11,f8
  nop.i 0;;
}



  {.mfi
  nop.m 0
  // f12=0?
  // This can be removed if sign of 0 is not important
  fcmp.eq.unc.s1 p8,p0=f12,f0
  nop.i 0;;
}
  {.mfb
  nop.m 0
  // if f8=0, set sign correctly
  // This can be removed if sign of 0 is not important
  (p8) fmerge.s f8=f7,f8
  // return
  br.ret.sptk b0;;
}


FREM_X_NAN_INF:

// Y zero ?
{.mfi
  nop.m 0
  fma.s1 f10=f9,f1,f0
  nop.i 0;;
}
{.mfi
 nop.m 0
 fcmp.eq.unc.s1 p11,p0=f10,f0
 nop.i 0;;
}
{.mib
  nop.m 0
  nop.i 0
  // if Y zero
  (p11) br.cond.spnt FREM_Y_ZERO;;
}

// X infinity? Return QNAN indefinite
{ .mfi
      nop.m 999
      fclass.m.unc  p8,p0 = f8, 0x23
      nop.i 999
}
// X infinity? Return QNAN indefinite
{ .mfi
      nop.m 999
      fclass.m.unc  p11,p0 = f8, 0x23
      nop.i 999;;
}
// Y NaN ?
{.mfi
	 nop.m 999
(p8) fclass.m.unc p0,p8=f9,0xc3
	 nop.i 0;;
}
{.mfi
	nop.m 999
	// also set Denormal flag if necessary
(p8) fma.s0 f9=f9,f1,f0
    nop.i 0
}
{ .mfi
      nop.m 999
(p8)  frcpa.s0 f8,p7 = f8,f8
      nop.i 999 ;;
}

{.mfi
      nop.m 999
(p11) mov f10=f8
	  nop.i 0
}
{ .mfi
      nop.m 999
(p8) fma.s.s0 f8=f8,f1,f0
	  nop.i 0 ;;
}

{ .mfb
      nop.m 999
      frcpa.s0 f8,p7=f8,f9
	  (p11) br.cond.spnt EXP_ERROR_RETURN;;
}
{ .mib
	nop.m 0
	nop.i 0
	br.ret.spnt    b0 ;;
}


FREM_Y_NAN_INF_ZERO:

// Y INF
{ .mfi
      nop.m 999
      fclass.m.unc  p7,p0 = f9, 0x23
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
(p7)  fma.s.s0 f8=f8,f1,f0
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
(p9)  fma.s.s0 f8=f9,f1,f0
(p9)  br.ret.spnt    b0 ;;
}

FREM_Y_ZERO:
// Y zero? Must be zero at this point
// because it is the only choice left.
// Return QNAN indefinite

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
(p10)  frcpa.s0         f11,p7 = f0,f0
nop.i 999;;
}

{ .mfi
      nop.m 999
      fmerge.s      f10 = f8, f8
      nop.i 999
}

{ .mfi
      nop.m 999
      fma.s.s0 f8=f11,f1,f0
      nop.i 999
}


EXP_ERROR_RETURN:

{ .mib
      mov   GR_Parameter_TAG = 125
	  nop.i 999
      br.sptk __libm_error_region;;
}

GLOBAL_IEEE754_END(remainderf)
libm_alias_float_other (__remainder, remainder)
weak_alias (__remainderf, dremf)



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
        stfs [GR_Parameter_Y] = FR_Y,16         // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfs [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
	nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfs [GR_Parameter_Y] = FR_RESULT      // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#;;  // Call error handling function
}
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
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
