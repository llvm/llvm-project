.file "nextafter.s"


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
//==============================================================
// 02/02/00 Initial version
// 03/03/00 Modified to conform to C9X, and improve speed of main path
// 03/14/00 Fixed case where x is a power of 2, and x > y, improved speed
// 04/04/00 Unwind support added
// 05/12/00 Fixed erroneous denormal flag setting for exponent change cases 1,3
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 09/09/00 Updated fcmp so that qnans do not raise invalid
// 12/15/00 Corrected behavior when both args are zero to conform to C99, and
//          fixed flag settings for several cases
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 12/14/04 Added error handling on underflow.
//
// API
//==============================================================
// double nextafter( double x, double y );
// input  floating point f8, f9
// output floating point f8
//
// Registers used
//==============================================================
GR_max_pexp     = r14
GR_min_pexp     = r15
GR_exp          = r16
GR_sig          = r17
GR_lnorm_sig    = r18
GR_sign_mask    = r19
GR_exp_mask     = r20
GR_sden_sig     = r21
GR_new_sig      = r22
GR_new_exp      = r23
GR_lden_sig     = r24
GR_snorm_sig    = r25
GR_exp1         = r26
GR_x_exp        = r27
GR_min_den_rexp = r28
// r36-39 parameters for libm_error_support

GR_SAVE_B0                = r34
GR_SAVE_GP                = r35
GR_SAVE_PFS               = r32

GR_Parameter_X            = r36
GR_Parameter_Y            = r37
GR_Parameter_RESULT       = r38
GR_Parameter_TAG          = r39

FR_lnorm_sig       = f10
FR_lnorm_exp       = f11
FR_lnorm           = f12
FR_sden_sig        = f13
FR_sden_exp        = f14
FR_sden            = f15
FR_save_f8         = f33
FR_new_exp         = f34
FR_new_sig         = f35
FR_lden_sig        = f36
FR_snorm_sig       = f37
FR_exp1            = f38
FR_tmp             = f39

//
// Overview of operation
//==============================================================
// nextafter determines the next representable value
// after x in the direction of y.


.section .text
GLOBAL_LIBM_ENTRY(nextafter)

// Extract signexp from x
// Is x < y ?  p10 if yes, p11 if no
// Form smallest denormal significand = ulp size
{ .mfi
      getf.exp GR_exp      = f8
      fcmp.lt.s1 p10,p11 = f8, f9
      addl GR_sden_sig = 0x800, r0
}
// Form largest normal significand 0xfffffffffffff800
// Form smallest normal exponent
{ .mfi
      addl GR_lnorm_sig = -0x800,r0
      nop.f 999
      addl GR_min_pexp = 0x0fc01, r0 ;;
}
// Extract significand from x
// Is x=y?
// Form largest normal exponent
{ .mfi
      getf.sig GR_sig      = f8
      fcmp.eq.s0 p6,p0 = f8, f9
      addl GR_max_pexp = 0x103fe, r0
}
// Move largest normal significand to fp reg for special cases
{ .mfi
      setf.sig FR_lnorm_sig = GR_lnorm_sig
      nop.f 999
      addl GR_sign_mask = 0x20000, r0 ;;
}

// Move smallest denormal significand and signexp to fp regs
// Is x=nan?
// Set p12 and p13 based on whether significand increases or decreases
// It increases (p12 set) if x<y and x>=0 or if x>y and x<0
// It decreases (p13 set) if x<y and x<0  or if x>y and x>=0
{ .mfi
      setf.sig FR_sden_sig = GR_sden_sig
      fclass.m  p8,p0 = f8, 0xc3
(p10) cmp.lt p12,p13 = GR_exp, GR_sign_mask
}
{ .mfi
      setf.exp FR_sden_exp = GR_min_pexp
(p11) cmp.ge p12,p13 = GR_exp, GR_sign_mask ;;
}

.pred.rel "mutex",p12,p13

// Form expected new significand, adding or subtracting 1 ulp increment
// If x=y set result to y
// Form smallest normal significand and largest denormal significand
{ .mfi
(p12) add GR_new_sig = GR_sig, GR_sden_sig
(p6)  fmerge.s f8=f9,f9
      dep.z GR_snorm_sig = 1,63,1 // 0x8000000000000000
}
{ .mlx
(p13) sub GR_new_sig = GR_sig, GR_sden_sig
      movl GR_lden_sig = 0x7ffffffffffff800 ;;
}

// Move expected result significand and signexp to fp regs
// Is y=nan?
// Form new exponent in case result exponent needs incrementing or decrementing
{ .mfi
      setf.exp FR_new_exp = GR_exp
      fclass.m  p9,p0 = f9, 0xc3
(p12) add GR_exp1 = 1, GR_exp
}
{ .mib
      setf.sig FR_new_sig = GR_new_sig
(p13) add GR_exp1 = -1, GR_exp
(p6)  br.ret.spnt    b0 ;;             // Exit if x=y
}

// Move largest normal signexp to fp reg for special cases
// Is x=zero?
{ .mfi
      setf.exp FR_lnorm_exp = GR_max_pexp
      fclass.m  p7,p0 = f8, 0x7
      nop.i 999
}
{ .mfb
      nop.m 999
(p8)  fma.s0 f8 = f8,f1,f9
(p8)  br.ret.spnt    b0 ;;             // Exit if x=nan
}

// Move exp+-1 and smallest normal significand to fp regs for special cases
// Is x=inf?
{ .mfi
      setf.exp FR_exp1 = GR_exp1
      fclass.m  p6,p0 = f8, 0x23
      addl GR_exp_mask = 0x1ffff, r0
}
{ .mfb
      setf.sig FR_snorm_sig = GR_snorm_sig
(p9)  fma.s0 f8 = f8,f1,f9
(p9)  br.ret.spnt    b0 ;;             // Exit if y=nan
}

// Move largest denormal significand to fp regs for special cases
// Save x
{ .mfb
      setf.sig FR_lden_sig = GR_lden_sig
      mov FR_save_f8 = f8
(p7)  br.cond.spnt NEXT_ZERO ;;   // Exit if x=0
}

// Mask off the sign to get x_exp
{ .mfb
      and GR_x_exp = GR_exp_mask, GR_exp
      nop.f 999
(p6)  br.cond.spnt NEXT_INF ;;   // Exit if x=inf
}

// Check 6 special cases when significand rolls over:
//  1 sig size incr, x_sig=max_sig, x_exp < max_exp
//     Set p6, result is sig=min_sig, exp++
//  2 sig size incr, x_sig=max_sig, x_exp >= max_exp
//     Set p7, result is inf, signal overflow
//  3 sig size decr, x_sig=min_sig, x_exp > min_exp
//     Set p8, result is sig=max_sig, exp--
//  4 sig size decr, x_sig=min_sig, x_exp = min_exp
//     Set p9, result is sig=max_den_sig, exp same, signal underflow and inexact
//  5 sig size decr, x_sig=min_den_sig, x_exp = min_exp
//     Set p10, result is zero, sign of x, signal underflow and inexact
//  6 sig size decr, x_sig=min_sig, x_exp < min_exp
//     Set p14, result is zero, sign of x, signal underflow and inexact
//
// Form exponent of smallest double denormal (if normalized register format)
{ .mmi
      adds GR_min_den_rexp = -52, GR_min_pexp
(p12) cmp.eq.unc p6,p0 = GR_new_sig, r0
(p13) cmp.eq.unc p8,p10 = GR_new_sig, GR_lden_sig ;;
}

{ .mmi
(p6)  cmp.lt.unc p6,p7 = GR_x_exp, GR_max_pexp
(p8)  cmp.gt.unc p8,p9 = GR_x_exp, GR_min_pexp
(p10) cmp.eq.unc p10,p0 = GR_new_sig, r0 ;;
}

// Create small normal in case need to generate underflow flag
{ .mfi
(p10) cmp.le.unc p10,p0 = GR_x_exp, GR_min_pexp
      fmerge.se FR_tmp = FR_sden_exp, FR_lnorm_sig
(p9)  cmp.gt.unc p9,p14 = GR_x_exp, GR_min_den_rexp
}
// Branch if cases 1, 2, 3
{ .bbb
(p6)  br.cond.spnt NEXT_EXPUP
(p7)  br.cond.spnt NEXT_OVERFLOW
(p8)  br.cond.spnt NEXT_EXPDOWN ;;
}

// Branch if cases 4, 5, 6
{ .bbb
(p9)  br.cond.spnt NEXT_NORM_TO_DENORM
(p10) br.cond.spnt NEXT_UNDERFLOW_TO_ZERO
(p14) br.cond.spnt NEXT_UNDERFLOW_TO_ZERO ;;
}

// Here if no special cases
// Set p6 if result will be a denormal, so can force underflow flag
//    Case 1:  x_exp=min_exp, x_sig=unnormalized
//    Case 2:  x_exp<min_exp
{ .mfi
      cmp.lt p6,p7 = GR_x_exp, GR_min_pexp
      fmerge.se f8 = FR_new_exp, FR_new_sig
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
      nop.f 999
(p7)  tbit.z p6,p0 = GR_new_sig, 63 ;;
}

NEXT_COMMON_FINISH:
// Force underflow and inexact if denormal result
{ .mfi
      nop.m 999
(p6)  fma.d.s0 FR_tmp = FR_tmp,FR_tmp,f0
      nop.i 999
}
{ .mfb
      nop.m 999
      fnorm.d.s0 f8 = f8 // Final normalization to result precision
(p6)  br.cond.spnt NEXT_UNDERFLOW ;;
}

{ .mfb
      nop.m 999
      nop.f 999
      br.ret.sptk b0;;
}

//Special cases
NEXT_EXPUP:
{ .mfb
      cmp.lt p6,p7 = GR_x_exp, GR_min_pexp
      fmerge.se f8 = FR_exp1, FR_snorm_sig
      br.cond.sptk NEXT_COMMON_FINISH ;;
}

NEXT_EXPDOWN:
{ .mfb
      cmp.lt p6,p7 = GR_x_exp, GR_min_pexp
      fmerge.se f8 = FR_exp1, FR_lnorm_sig
      br.cond.sptk NEXT_COMMON_FINISH ;;
}

NEXT_NORM_TO_DENORM:
{ .mfi
      nop.m 999
      fmerge.se f8 = FR_new_exp, FR_lden_sig
      nop.i 999
}
// Force underflow and inexact if denormal result
{ .mfb
      nop.m 999
      fma.d.s0 FR_tmp = FR_tmp,FR_tmp,f0
      br.cond.sptk NEXT_UNDERFLOW ;;
}

NEXT_UNDERFLOW_TO_ZERO:
{ .mfb
      cmp.eq p6,p0 = r0,r0
      fmerge.s f8 = FR_save_f8,f0
      br.cond.sptk NEXT_COMMON_FINISH ;;
}

NEXT_INF:
// Here if f8 is +- infinity
// INF
// if f8 is +inf, no matter what y is return  largest double
// if f8 is -inf, no matter what y is return -largest double

{ .mfi
      nop.m 999
      fmerge.se FR_lnorm = FR_lnorm_exp,FR_lnorm_sig
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
      fmerge.s f8 = f8,FR_lnorm
      br.ret.sptk    b0 ;;
}

NEXT_ZERO:

// Here if f8 is +- zero
// ZERO
// if f8 is zero and y is +, return + smallest double denormal
// if f8 is zero and y is -, return - smallest double denormal

{ .mfi
      nop.m 999
      fmerge.se FR_sden = FR_sden_exp,FR_sden_sig
      nop.i 999 ;;
}

// Create small normal to generate underflow flag
{ .mfi
      nop.m 999
      fmerge.se FR_tmp = FR_sden_exp, FR_lnorm_sig
      nop.i 999 ;;
}

// Add correct sign from direction arg
{ .mfi
      nop.m 999
      fmerge.s f8 = f9,FR_sden
      nop.i 999 ;;
}

// Force underflow and inexact flags
{ .mfb
      nop.m 999
      fma.d.s0 FR_tmp = FR_tmp,FR_tmp,f0
      br.cond.sptk NEXT_UNDERFLOW ;;
}

NEXT_UNDERFLOW:
// Here if result is a denorm, or input is finite and result is zero
// Call error support to report possible range error
{ .mib
      alloc          r32=ar.pfs,2,2,4,0
      mov           GR_Parameter_TAG = 268      // Error code
      br.cond.sptk  __libm_error_region    // Branch to error call
}
;;

NEXT_OVERFLOW:
// Here if input is finite, but result will be infinite
// Use frcpa to generate infinity of correct sign
// Call error support to report possible range error
{ .mfi
      alloc          r32=ar.pfs,2,2,4,0
      frcpa.s1 f8,p6 = FR_save_f8, f0
      nop.i 999 ;;
}

// Create largest double
{ .mfi
      nop.m 999
      fmerge.se FR_lnorm = FR_lnorm_exp,FR_lnorm_sig
      nop.i 999 ;;
}

// Force overflow and inexact flags to be set
{ .mfb
      mov           GR_Parameter_TAG = 154      // Error code
      fma.d.s0 FR_tmp = FR_lnorm,FR_lnorm,f0
      br.cond.sptk  __libm_error_region    // Branch to error call
}
;;

GLOBAL_LIBM_END(nextafter)
libm_alias_double_other (nextafter, nextafter)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

// (1)
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                          // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                      // Save gp
};;


// (2)
{ .mmi
        stfd [GR_Parameter_Y] = f9,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                     // Save b0
};;

.body
// (3)
{ .mib
        stfd [GR_Parameter_X] = FR_save_f8              // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y           // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = f8              // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;

// (4)
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
