.file "scalbl.s"


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
//==============================================================
// 02/02/00 Initial version
// 01/26/01 Scalb completely reworked and now standalone version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 08/06/03 Improved performance
//
// API
//==============================================================
// long double = scalbl  (long double x, long double n)
// input  floating point f8 and floating point f9
// output floating point f8
//
// int_type = 0 if int is 32 bits
// int_type = 1 if int is 64 bits
//
// Returns x* 2**n using an fma and detects overflow
// and underflow.
//
//
// Strategy:
//  Compute biased exponent of result exp_Result = N + exp_X
//  Break into ranges:
//   exp_Result > 0x13ffe                 -> Certain overflow
//   exp_Result = 0x13ffe                 -> Possible overflow
//   0x0c001 <= exp_Result < 0x13ffe      -> No over/underflow (main path)
//   0x0c001 - 63 <= exp_Result < 0x0c001 -> Possible underflow
//   exp_Result < 0x0c001 - 63            -> Certain underflow

FR_Big         = f6
FR_NBig        = f7
FR_Floating_X  = f8
FR_Result      = f8
FR_Floating_N  = f9
FR_Result2     = f9
FR_Result3     = f10
FR_Norm_X      = f11
FR_Two_N       = f12
FR_N_float_int = f13
FR_Norm_N      = f14

GR_neg_ov_limit= r14
GR_big_exp     = r14
GR_N_Biased    = r15
GR_Big         = r16
GR_exp_Result  = r18
GR_pos_ov_limit= r19
GR_exp_sure_ou = r19
GR_Bias        = r20
GR_N_as_int    = r21
GR_signexp_X   = r22
GR_exp_X       = r23
GR_exp_mask    = r24
GR_max_exp     = r25
GR_min_exp     = r26
GR_min_den_exp = r27
GR_Scratch     = r28
GR_signexp_N   = r29
GR_exp_N       = r30

GR_SAVE_B0          = r32
GR_SAVE_GP          = r33
GR_SAVE_PFS         = r34
GR_Parameter_X      = r35
GR_Parameter_Y      = r36
GR_Parameter_RESULT = r37
GR_Tag              = r38

.section .text
GLOBAL_IEEE754_ENTRY(scalbl)

//
//   Is x NAN, INF, ZERO, +-?
//   Build the exponent Bias
//
{    .mfi
     getf.exp      GR_signexp_N = FR_Floating_N // Get signexp of n
     fclass.m      p6,p0 = FR_Floating_X, 0xe7  // @snan | @qnan | @inf | @zero
     mov           GR_Bias = 0x0ffff
}
{    .mfi
     mov           GR_Big = 35000      // If N this big then certain overflow
     fcvt.fx.trunc.s1   FR_N_float_int = FR_Floating_N // Get N in significand
     nop.i         0
}
;;

{    .mfi
     getf.exp      GR_signexp_X = FR_Floating_X // Get signexp of x
     fclass.m      p7,p0 = FR_Floating_N, 0x0b  // Test for n=unorm
     nop.i         0
}
//
//   Normalize n
//
{    .mfi
     mov           GR_exp_mask = 0x1ffff     // Exponent mask
     fnorm.s1      FR_Norm_N = FR_Floating_N
     nop.i         0
}
;;

//
//   Is n NAN, INF, ZERO, +-?
//
{    .mfi
     mov           GR_big_exp = 0x1003e      // Exponent at which n is integer
     fclass.m      p9,p0 = FR_Floating_N, 0xe7  // @snan | @qnan | @inf | @zero
     mov           GR_max_exp = 0x13ffe      // Exponent of maximum long double
}
//
//   Normalize x
//
{ .mfb
     nop.m         0
     fnorm.s1      FR_Norm_X = FR_Floating_X
(p7) br.cond.spnt  SCALBL_N_UNORM             // Branch if n=unorm
}
;;

SCALBL_COMMON1:
// Main path continues.  Also return here from u=unorm path.
//   Handle special cases if x = Nan, Inf, Zero
{ .mfb
     nop.m         0
     fcmp.lt.s1    p7,p0 = FR_Floating_N, f0  // Test N negative
(p6) br.cond.spnt  SCALBL_NAN_INF_ZERO
}
;;

//   Handle special cases if n = Nan, Inf, Zero
{    .mfi
     getf.sig      GR_N_as_int = FR_N_float_int // Get n from significand
     fclass.m      p8,p0 = FR_Floating_X, 0x0b // Test for x=unorm
     mov           GR_exp_sure_ou = 0x1000e // Exp_N where x*2^N sure over/under
}
{    .mfb
     mov           GR_min_exp = 0x0c001      // Exponent of minimum long double
     fcvt.xf       FR_N_float_int = FR_N_float_int // Convert N to FP integer
(p9) br.cond.spnt  SCALBL_NAN_INF_ZERO
}
;;

{    .mmi
     and           GR_exp_N = GR_exp_mask, GR_signexp_N // Get exponent of N
(p7) sub           GR_Big = r0, GR_Big          // Limit for N
     nop.i         0
}
;;

{    .mib
     cmp.lt        p9,p0 = GR_exp_N, GR_big_exp // N possible non-integer?
     cmp.ge        p6,p0 = GR_exp_N, GR_exp_sure_ou // N certain over/under?
(p8) br.cond.spnt  SCALBL_X_UNORM             // Branch if x=unorm
}
;;

SCALBL_COMMON2:
// Main path continues.  Also return here from x=unorm path.
//   Create biased exponent for 2**N
{    .mmi
(p6) mov           GR_N_as_int = GR_Big      // Limit N
;;
     add           GR_N_Biased = GR_Bias,GR_N_as_int
     nop.i         0
}
;;

{    .mfi
     setf.exp      FR_Two_N = GR_N_Biased               // Form 2**N
(p9) fcmp.neq.unc.s1 p9,p0 = FR_Norm_N, FR_N_float_int  // Test if N an integer
     and           GR_exp_X = GR_exp_mask, GR_signexp_X // Get exponent of X
}
;;

//
//   Compute biased result exponent
//   Branch if N is not an integer
//
{    .mib
     add           GR_exp_Result = GR_exp_X, GR_N_as_int
     mov           GR_min_den_exp = 0x0c001 - 63 // Exp of min denorm long dble
(p9) br.cond.spnt  SCALBL_N_NOT_INT
}
;;

//
//   Raise Denormal operand flag with compare
//   Do final operation
//
{    .mfi
     cmp.lt        p7,p6 = GR_exp_Result, GR_max_exp  // Test no overflow
     fcmp.ge.s0    p0,p11 = FR_Floating_X,FR_Floating_N  // Dummy to set denorm
     cmp.lt        p9,p0 = GR_exp_Result, GR_min_den_exp // Test sure underflow
}
{    .mfb
     nop.m         0
     fma.s0        FR_Result = FR_Two_N,FR_Norm_X,f0
(p9) br.cond.spnt  SCALBL_UNDERFLOW           // Branch if certain underflow
}
;;

{    .mib
(p6) cmp.gt.unc    p6,p8 = GR_exp_Result, GR_max_exp  // Test sure overflow
(p7) cmp.ge.unc    p7,p9 = GR_exp_Result, GR_min_exp  // Test no over/underflow
(p7) br.ret.sptk   b0                         // Return from main path
}
;;

{    .bbb
(p6) br.cond.spnt  SCALBL_OVERFLOW            // Branch if certain overflow
(p8) br.cond.spnt  SCALBL_POSSIBLE_OVERFLOW   // Branch if possible overflow
(p9) br.cond.spnt  SCALBL_POSSIBLE_UNDERFLOW  // Branch if possible underflow
}
;;

// Here if possible underflow.
// Resulting exponent: 0x0c001-63 <= exp_Result < 0x0c001
SCALBL_POSSIBLE_UNDERFLOW:
//
// Here if possible overflow.
// Resulting exponent: 0x13ffe = exp_Result
SCALBL_POSSIBLE_OVERFLOW:

//   Set up necessary status fields
//
//   S0 user supplied status
//   S2 user supplied status + WRE + TD  (Overflows)
//   S3 user supplied status + FZ + TD   (Underflows)
//
{    .mfi
     mov           GR_pos_ov_limit = 0x13fff // Exponent for positive overflow
     fsetc.s3      0x7F,0x41
     nop.i         0
}
{    .mfi
     mov           GR_neg_ov_limit = 0x33fff // Exponent for negative overflow
     fsetc.s2      0x7F,0x42
     nop.i         0
}
;;

//
//   Do final operation with s2 and s3
//
{    .mfi
     setf.exp      FR_NBig = GR_neg_ov_limit
     fma.s3        FR_Result3 = FR_Two_N,FR_Norm_X,f0
     nop.i         0
}
{    .mfi
     setf.exp      FR_Big = GR_pos_ov_limit
     fma.s2        FR_Result2 = FR_Two_N,FR_Norm_X,f0
     nop.i         0
}
;;

//   Check for overflow or underflow.
//   Restore s3
//   Restore s2
//
{    .mfi
     nop.m         0
     fsetc.s3      0x7F,0x40
     nop.i         0
}
{    .mfi
     nop.m         0
     fsetc.s2      0x7F,0x40
     nop.i         0
}
;;

//
//   Is the result zero?
//
{    .mfi
     nop.m         0
     fclass.m      p6, p0 =  FR_Result3, 0x007
     nop.i         0
}
{    .mfi
     nop.m         0
     fcmp.ge.s1    p7, p8 = FR_Result2 , FR_Big
     nop.i         0
}
;;

//
//   Detect masked underflow - Tiny + Inexact Only
//
{    .mfi
     nop.m         0
(p6) fcmp.neq.unc.s1 p6, p0 = FR_Result , FR_Result2
     nop.i         0
}
;;

//
//   Is result bigger the allowed range?
//   Branch out for underflow
//
{    .mfb
     nop.m          0
(p8) fcmp.le.unc.s1 p9, p10 = FR_Result2 , FR_NBig
(p6) br.cond.spnt   SCALBL_UNDERFLOW
}
;;

//
//   Branch out for overflow
//
{ .bbb
(p7) br.cond.spnt   SCALBL_OVERFLOW
(p9) br.cond.spnt   SCALBL_OVERFLOW
     br.ret.sptk    b0             //   Return from main path.
}
;;

// Here if result overflows
SCALBL_OVERFLOW:
{ .mib
     alloc         r32=ar.pfs,3,0,4,0
     addl          GR_Tag = 51, r0     // Set error tag for overflow
     br.cond.sptk  __libm_error_region // Call error support for overflow
}
;;

// Here if result underflows
SCALBL_UNDERFLOW:
{ .mib
     alloc         r32=ar.pfs,3,0,4,0
     addl          GR_Tag = 52, r0     // Set error tag for underflow
     br.cond.sptk  __libm_error_region // Call error support for underflow
}
;;

SCALBL_NAN_INF_ZERO:

//
//   Before entry, N has been converted to a fp integer in significand of
//     FR_N_float_int
//
//   Convert  N_float_int to floating point value
//
{    .mfi
     getf.sig     GR_N_as_int = FR_N_float_int
     fclass.m     p6,p0 = FR_Floating_N, 0xc3 //@snan | @qnan
     nop.i        0
}
{    .mfi
     addl         GR_Scratch = 1,r0
     fcvt.xf      FR_N_float_int = FR_N_float_int
     nop.i        0
}
;;

{    .mfi
     nop.m        0
     fclass.m     p7,p0 = FR_Floating_X, 0xc3 //@snan | @qnan
     shl          GR_Scratch = GR_Scratch,63
}
;;

{    .mfi
     nop.m        0
     fclass.m     p8,p0 = FR_Floating_N, 0x21 // @inf
     nop.i        0
}
{    .mfi
     nop.m        0
     fclass.m     p9,p0 = FR_Floating_N, 0x22 // @-inf
     nop.i        0
}
;;

//
//   Either X or N is a Nan, return result and possible raise invalid.
//
{    .mfb
     nop.m        0
(p6) fma.s0       FR_Result = FR_Floating_N,FR_Floating_X,f0
(p6) br.ret.spnt  b0
}
;;

{    .mfb
     nop.m        0
(p7) fma.s0       FR_Result = FR_Floating_N,FR_Floating_X,f0
(p7) br.ret.spnt  b0
}
;;

//
//   If N + Inf do something special
//   For N = -Inf, create Int
//
{    .mfb
     nop.m        0
(p8) fma.s0       FR_Result = FR_Floating_X, FR_Floating_N,f0
(p8) br.ret.spnt  b0
}
{    .mfi
     nop.m        0
(p9) fnma.s0      FR_Floating_N = FR_Floating_N, f1, f0
     nop.i        0
}
;;

//
//   If N==-Inf,return x/(-N)
//
{    .mfb
     cmp.ne       p7,p0 = GR_N_as_int,GR_Scratch
(p9) frcpa.s0     FR_Result,p0 = FR_Floating_X,FR_Floating_N
(p9) br.ret.spnt  b0
}
;;

//
//   Is N an integer.
//
{    .mfi
     nop.m        0
(p7) fcmp.neq.unc.s1 p7,p0 = FR_Norm_N, FR_N_float_int
     nop.i        0
}
;;

//
//   If N not an int, return NaN and raise invalid.
//
{    .mfb
     nop.m        0
(p7) frcpa.s0     FR_Result,p0 = f0,f0
(p7) br.ret.spnt  b0
}
;;

//
//   Always return x in other path.
//
{    .mfb
     nop.m        0
     fma.s0       FR_Result = FR_Floating_X,f1,f0
     br.ret.sptk  b0
}
;;

// Here if n not int
// Return NaN and raise invalid.
SCALBL_N_NOT_INT:
{    .mfb
     nop.m        0
     frcpa.s0     FR_Result,p0 = f0,f0
     br.ret.sptk  b0
}
;;

// Here if n=unorm
SCALBL_N_UNORM:
{ .mfb
     getf.exp      GR_signexp_N = FR_Norm_N // Get signexp of normalized n
     fcvt.fx.trunc.s1   FR_N_float_int = FR_Norm_N // Get N in significand
     br.cond.sptk  SCALBL_COMMON1            // Return to main path
}
;;

// Here if x=unorm
SCALBL_X_UNORM:
{ .mib
     getf.exp      GR_signexp_X = FR_Norm_X // Get signexp of normalized x
     nop.i         0
     br.cond.sptk  SCALBL_COMMON2            // Return to main path
}
;;

GLOBAL_IEEE754_END(scalbl)
LOCAL_LIBM_ENTRY(__libm_error_region)

//
// Get stack address of N
//
.prologue
{ .mfi
    add   GR_Parameter_Y=-32,sp
    nop.f 0
.save   ar.pfs,GR_SAVE_PFS
    mov  GR_SAVE_PFS=ar.pfs
}
//
// Adjust sp
//
{ .mfi
.fframe 64
   add sp=-64,sp
   nop.f 0
   mov GR_SAVE_GP=gp
};;

//
//  Store N on stack in correct position
//  Locate the address of x on stack
//
{ .mmi
   stfe [GR_Parameter_Y] = FR_Norm_N,16
   add GR_Parameter_X = 16,sp
.save   b0, GR_SAVE_B0
   mov GR_SAVE_B0=b0
};;

//
// Store x on the stack.
// Get address for result on stack.
//
.body
{ .mib
   stfe [GR_Parameter_X] = FR_Norm_X
   add   GR_Parameter_RESULT = 0,GR_Parameter_Y
   nop.b 0
}
{ .mib
   stfe [GR_Parameter_Y] = FR_Result
   add   GR_Parameter_Y = -16,GR_Parameter_Y
   br.call.sptk b0=__libm_error_support#
};;

//
//  Get location of result on stack
//
{ .mmi
   add   GR_Parameter_RESULT = 48,sp
   nop.m 0
   nop.i 0
};;

//
//  Get the new result
//
{ .mmi
   ldfe  FR_Result = [GR_Parameter_RESULT]
.restore sp
   add   sp = 64,sp
   mov   b0 = GR_SAVE_B0
};;

//
//  Restore gp, ar.pfs and return
//
{ .mib
   mov   gp = GR_SAVE_GP
   mov   ar.pfs = GR_SAVE_PFS
   br.ret.sptk     b0
};;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
