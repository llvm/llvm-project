.file "libm_ldexpf.s"


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
// 01/26/01 ldexpf completely reworked and now standalone version
// 01/04/02 Added handling for int 32 or 64 bits
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 08/04/03 Improved performance
//
// API
//==============================================================
// float __libm_ldexpf  (float x, int n, int int_type)
// input  floating point f8 and int n (r33), int int_type (r34)
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
//   exp_Result > 0x1007e                 -> Certain overflow
//   exp_Result = 0x1007e                 -> Possible overflow
//   0x0ff81 <= exp_Result < 0x1007e      -> No over/underflow (main path)
//   0x0ff81 - 23 <= exp_Result < 0x0ff81 -> Possible underflow
//   exp_Result < 0x0ff81 - 23            -> Certain underflow

FR_Big         = f6
FR_NBig        = f7
FR_Floating_X  = f8
FR_Result      = f8
FR_Result2     = f9
FR_Result3     = f10
FR_Norm_X      = f11
FR_Two_N       = f12

GR_neg_ov_limit= r14
GR_N_Biased    = r15
GR_Big         = r16
GR_NBig        = r17
GR_exp_Result  = r18
GR_pos_ov_limit= r19
GR_Bias        = r20
GR_N_as_int    = r21
GR_signexp_X   = r22
GR_exp_X       = r23
GR_exp_mask    = r24
GR_max_exp     = r25
GR_min_exp     = r26
GR_min_den_exp = r27

GR_SAVE_B0          = r32
GR_SAVE_GP          = r33
GR_SAVE_PFS         = r34
GR_Parameter_X      = r35
GR_Parameter_Y      = r36
GR_Parameter_RESULT = r37
GR_Tag              = r38

.section .text
GLOBAL_LIBM_ENTRY(__libm_ldexpf)

//
//   Is x NAN, INF, ZERO, +-?
//   Build the exponent Bias
//
{    .mfi
     getf.exp      GR_signexp_X = FR_Floating_X // Get signexp of x
     fclass.m      p6,p0 = FR_Floating_X, 0xe7  // @snan | @qnan | @inf | @zero
     mov           GR_Bias = 0x0ffff
}
//
//   Normalize x
//   Is integer type 32 bits?
//
{    .mfi
     mov           GR_Big = 35000      // If N this big then certain overflow
     fnorm.s1      FR_Norm_X = FR_Floating_X
     cmp.eq        p8,p9 = r34,r0
}
;;

//   Sign extend N if int is 32 bits
{    .mfi
(p9) mov           GR_N_as_int = r33     // Copy N if int is 64 bits
     fclass.m      p9,p0 = FR_Floating_X, 0x0b // Test for x=unorm
(p8) sxt4          GR_N_as_int = r33     // Sign extend N if int is 32 bits
}
{ .mfi
     mov           GR_NBig = -35000    // If N this small then certain underflow
     nop.f         0
     mov           GR_max_exp = 0x1007e      // Exponent of maximum float
}
;;

//   Create biased exponent for 2**N
{    .mfi
     add           GR_N_Biased = GR_Bias,GR_N_as_int
     nop.f         0
     cmp.ge        p7, p0 = GR_N_as_int, GR_Big  // Certain overflow?
}
{    .mib
     cmp.le        p8, p0 = GR_N_as_int, GR_NBig // Certain underflow?
     mov           GR_min_exp = 0x0ff81      // Exponent of minimum float
(p9) br.cond.spnt  LDEXPF_UNORM              // Branch if x=unorm
}
;;

LDEXPF_COMMON:
// Main path continues.  Also return here from x=unorm path.
//   Create 2**N
.pred.rel "mutex",p7,p8
{    .mfi
     setf.exp      FR_Two_N = GR_N_Biased
     nop.f         0
(p7) mov           GR_N_as_int = GR_Big      // Limit max N
}
{    .mfi
(p8) mov           GR_N_as_int = GR_NBig     // Limit min N
     nop.f         0
(p8) cmp.eq        p7,p0 = r0,r0             // Set p7 if |N| big
}
;;

//
//   Create biased exponent for 2**N for N big
//   Is N zero?
//
{    .mfi
(p7) add           GR_N_Biased = GR_Bias,GR_N_as_int
     nop.f         0
     cmp.eq.or     p6,p0 = r33,r0
}
{    .mfi
     mov           GR_pos_ov_limit = 0x1007f // Exponent for positive overflow
     nop.f         0
     mov           GR_exp_mask = 0x1ffff     // Exponent mask
}
;;

//
//   Create 2**N for N big
//   Return x when N = 0 or X = Nan, Inf, Zero
//
{    .mfi
(p7) setf.exp      FR_Two_N = GR_N_Biased
     nop.f         0
     mov           GR_min_den_exp = 0x0ff81 - 23 // Exponent of min denorm float
}
{    .mfb
     and           GR_exp_X = GR_exp_mask, GR_signexp_X
(p6) fma.s.s0      FR_Result = FR_Floating_X, f1, f0
(p6) br.ret.spnt   b0
}
;;

//
//   Raise Denormal operand flag with compare
//   Compute biased result exponent
//
{    .mfi
     add           GR_exp_Result = GR_exp_X, GR_N_as_int
     fcmp.ge.s0    p0,p11 = FR_Floating_X,f0
     mov           GR_neg_ov_limit = 0x3007f // Exponent for negative overflow
}
;;

//
//   Do final operation
//
{    .mfi
     cmp.lt        p7,p6 = GR_exp_Result, GR_max_exp  // Test no overflow
     fma.s.s0      FR_Result = FR_Two_N,FR_Norm_X,f0
     cmp.lt        p9,p0 = GR_exp_Result, GR_min_den_exp // Test sure underflow
}
{    .mfb
     nop.m         0
     nop.f         0
(p9) br.cond.spnt  LDEXPF_UNDERFLOW           // Branch if certain underflow
}
;;

{    .mib
(p6) cmp.gt.unc    p6,p8 = GR_exp_Result, GR_max_exp  // Test sure overflow
(p7) cmp.ge.unc    p7,p9 = GR_exp_Result, GR_min_exp  // Test no over/underflow
(p7) br.ret.sptk   b0                         // Return from main path
}
;;

{    .bbb
(p6) br.cond.spnt  LDEXPF_OVERFLOW            // Branch if certain overflow
(p8) br.cond.spnt  LDEXPF_POSSIBLE_OVERFLOW   // Branch if possible overflow
(p9) br.cond.spnt  LDEXPF_POSSIBLE_UNDERFLOW  // Branch if possible underflow
}
;;

// Here if possible underflow.
// Resulting exponent: 0x0ff81-23 <= exp_Result < 0x0ff81
LDEXPF_POSSIBLE_UNDERFLOW:
//
// Here if possible overflow.
// Resulting exponent: 0x1007e = exp_Result
LDEXPF_POSSIBLE_OVERFLOW:

//   Set up necessary status fields
//
//   S0 user supplied status
//   S2 user supplied status + WRE + TD  (Overflows)
//   S3 user supplied status + FZ + TD   (Underflows)
//
{    .mfi
     nop.m         0
     fsetc.s3      0x7F,0x41
     nop.i         0
}
{    .mfi
     nop.m         0
     fsetc.s2      0x7F,0x42
     nop.i         0
}
;;

//
//   Do final operation with s2 and s3
//
{    .mfi
     setf.exp      FR_NBig = GR_neg_ov_limit
     fma.s.s3      FR_Result3 = FR_Two_N,FR_Norm_X,f0
     nop.i         0
}
{    .mfi
     setf.exp      FR_Big = GR_pos_ov_limit
     fma.s.s2      FR_Result2 = FR_Two_N,FR_Norm_X,f0
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
(p6) br.cond.spnt   LDEXPF_UNDERFLOW
}
;;

//
//   Branch out for overflow
//
{ .bbb
(p7) br.cond.spnt   LDEXPF_OVERFLOW
(p9) br.cond.spnt   LDEXPF_OVERFLOW
     br.ret.sptk    b0             //   Return from main path.
}
;;

// Here if result overflows
LDEXPF_OVERFLOW:
{ .mib
     alloc         r32=ar.pfs,3,0,4,0
     addl          GR_Tag = 148, r0    // Set error tag for overflow
     br.cond.sptk  __libm_error_region // Call error support for overflow
}
;;

// Here if result underflows
LDEXPF_UNDERFLOW:
{ .mib
     alloc         r32=ar.pfs,3,0,4,0
     addl          GR_Tag = 149, r0    // Set error tag for underflow
     br.cond.sptk  __libm_error_region // Call error support for underflow
}
;;

// Here if x=unorm
LDEXPF_UNORM:
{ .mib
     getf.exp      GR_signexp_X = FR_Norm_X // Get signexp of normalized x
     nop.i         0
     br.cond.sptk  LDEXPF_COMMON            // Return to main path
}
;;


GLOBAL_LIBM_END(__libm_ldexpf)
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
   st8 [GR_Parameter_Y] =  GR_N_as_int,16
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
   stfs [GR_Parameter_X] = FR_Norm_X
   add   GR_Parameter_RESULT = 0,GR_Parameter_Y
   nop.b 0
}
{ .mib
   stfs [GR_Parameter_Y] = FR_Result
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
   ldfs  FR_Result = [GR_Parameter_RESULT]
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
