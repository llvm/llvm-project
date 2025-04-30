.file "exp10.s"


// Copyright (c) 2000 - 2005, Intel Corporation
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
// 08/25/00 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 09/06/02 Improved performance; no inexact flags on exact cases
// 01/29/03 Added missing } to bundle templates
// 12/16/04 Call error handling on underflow.
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// double exp10(double)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// Let x= (K + fh + fl + r)/log2(10), where
// K is an integer, fh= 0.b1 b2 b3 b4 b5,
// fl= 2^{-5}* 0.b6 b7 b8 b8 b10 (fh, fl >= 0),
// and |r|<2^{-11}
// Th is a table that stores 2^fh (32 entries) rounded to
// double extended precision (only mantissa is stored)
// Tl is a table that stores 2^fl (32 entries) rounded to
// double extended precision (only mantissa is stored)
//
// 10^x is approximated as
// 2^K * Th [ f ] * Tl [ f ] * (1+c1*e+c1*r+c2*r^2+c3*r^3+c4*r^4),
// where e= (x*log2(10)_hi-RN(x*log2(10)_hi))+log2(10)_lo*x

// Note there are only 22 non-zero values that produce an exact result:
//  1.0, 2.0, ... 22.0.
// We test for these cases and use s1 to avoid setting the inexact flag.

// Special values
//==============================================================
// exp10(0)= 1
// exp10(+inf)= inf
// exp10(-inf)= 0
//

// Registers used
//==============================================================
// r2-r3, r14-r40
// f6-f15, f32-f52
// p6-p12
//

#include <shlib-compat.h>


GR_TBL_START        = r2
GR_LOG_TBL          = r3

GR_OF_LIMIT         = r14
GR_UF_LIMIT         = r15
GR_EXP_CORR         = r16
GR_F_low            = r17
GR_F_high           = r18
GR_K                = r19
GR_Flow_ADDR        = r20

GR_BIAS             = r21
GR_Fh               = r22
GR_Fh_ADDR          = r23
GR_EXPMAX           = r24
GR_BIAS53           = r25

GR_ROUNDVAL         = r26
GR_SNORM_LIMIT      = r26
GR_MASK             = r27
GR_KF0              = r28
GR_MASK_low         = r29
GR_COEFF_START      = r30
GR_exact_limit      = r31

GR_SAVE_B0          = r33
GR_SAVE_PFS         = r34
GR_SAVE_GP          = r35
GR_SAVE_SP          = r36

GR_Parameter_X      = r37
GR_Parameter_Y      = r38
GR_Parameter_RESULT = r39
GR_Parameter_TAG    = r40


FR_X                = f10
FR_Y                = f1
FR_RESULT           = f8


FR_COEFF1           = f6
FR_COEFF2           = f7
FR_R                = f9
FR_LOG2_10          = f10

FR_2P53             = f11
FR_KF0              = f12
FR_COEFF3           = f13
FR_COEFF4           = f14
FR_UF_LIMIT         = f15

FR_OF_LIMIT         = f32
FR_DX_L210          = f33
FR_ROUNDVAL         = f34
FR_KF               = f35

FR_2_TO_K           = f36
FR_T_low            = f37
FR_T_high           = f38
FR_P34              = f39
FR_R2               = f40

FR_P12              = f41
FR_T_low_K          = f42
FR_P14              = f43
FR_T                = f44
FR_P                = f45

FR_L2_10_low        = f46
FR_L2_10_high       = f47
FR_E0               = f48
FR_E                = f49
FR_exact_limit      = f50

FR_int_x            = f51
FR_SNORM_LIMIT      = f52


// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(poly_coeffs)

data8 0xd49a784bcd1b8afe, 0x00003fcb // log2(10)*2^(10-63)
data8 0x9257edfe9b5fb698, 0x3fbf // log2(10)_low (bits 64...127)
data8 0x3fac6b08d704a0c0, 0x3f83b2ab6fba4e77 // C_3 and C_4
data8 0xb17217f7d1cf79ab, 0x00003ffe // C_1
data8 0xf5fdeffc162c7541, 0x00003ffc // C_2
LOCAL_OBJECT_END(poly_coeffs)


LOCAL_OBJECT_START(T_table)

// 2^{0.00000 b6 b7 b8 b9 b10}
data8 0x8000000000000000, 0x8016302f17467628
data8 0x802c6436d0e04f50, 0x80429c17d77c18ed
data8 0x8058d7d2d5e5f6b0, 0x806f17687707a7af
data8 0x80855ad965e88b83, 0x809ba2264dada76a
data8 0x80b1ed4fd999ab6c, 0x80c83c56b50cf77f
data8 0x80de8f3b8b85a0af, 0x80f4e5ff089f763e
data8 0x810b40a1d81406d4, 0x81219f24a5baa59d
data8 0x813801881d886f7b, 0x814e67cceb90502c
data8 0x8164d1f3bc030773, 0x817b3ffd3b2f2e47
data8 0x8191b1ea15813bfd, 0x81a827baf7838b78
data8 0x81bea1708dde6055, 0x81d51f0b8557ec1c
data8 0x81eba08c8ad4536f, 0x820225f44b55b33b
data8 0x8218af4373fc25eb, 0x822f3c7ab205c89a
data8 0x8245cd9ab2cec048, 0x825c62a423d13f0c
data8 0x8272fb97b2a5894c, 0x828998760d01faf3
data8 0x82a0393fe0bb0ca8, 0x82b6ddf5dbc35906
//
// 2^{0.b1 b2 b3 b4 b5}
data8 0x8000000000000000, 0x82cd8698ac2ba1d7
data8 0x85aac367cc487b14, 0x88980e8092da8527
data8 0x8b95c1e3ea8bd6e6, 0x8ea4398b45cd53c0
data8 0x91c3d373ab11c336, 0x94f4efa8fef70961
data8 0x9837f0518db8a96f, 0x9b8d39b9d54e5538
data8 0x9ef5326091a111ad, 0xa27043030c496818
data8 0xa5fed6a9b15138ea, 0xa9a15ab4ea7c0ef8
data8 0xad583eea42a14ac6, 0xb123f581d2ac258f
data8 0xb504f333f9de6484, 0xb8fbaf4762fb9ee9
data8 0xbd08a39f580c36be, 0xc12c4cca66709456
data8 0xc5672a115506dadd, 0xc9b9bd866e2f27a2
data8 0xce248c151f8480e3, 0xd2a81d91f12ae45a
data8 0xd744fccad69d6af4, 0xdbfbb797daf23755
data8 0xe0ccdeec2a94e111, 0xe5b906e77c8348a8
data8 0xeac0c6e7dd24392e, 0xefe4b99bdcdaf5cb
data8 0xf5257d152486cc2c, 0xfa83b2db722a033a
LOCAL_OBJECT_END(T_table)



.section .text
GLOBAL_IEEE754_ENTRY(exp10)


{.mfi
       alloc r32= ar.pfs, 1, 4, 4, 0
       // will continue only for non-zero normal/denormal numbers
       fclass.nm.unc p12, p7= f8, 0x1b
       mov GR_BIAS53= 0xffff+63-10
}
{.mlx
       // GR_TBL_START= pointer to log2(10), C_1...C_4 followed by T_table
       addl GR_TBL_START= @ltoff(poly_coeffs), gp
       movl GR_ROUNDVAL= 0x3fc00000             // 1.5 (SP)
}
;;

{.mfi
       ld8 GR_COEFF_START= [ GR_TBL_START ]     // Load pointer to coeff table
       fcmp.lt.s1 p6, p8= f8, f0                // X<0 ?
       nop.i 0
}
;;

{.mlx
       setf.exp FR_2P53= GR_BIAS53              // 2^{63-10}
       movl GR_UF_LIMIT= 0xc07439b746e36b52     // (-2^10-51) / log2(10)
}
{.mlx
       setf.s FR_ROUNDVAL= GR_ROUNDVAL
       movl GR_OF_LIMIT= 0x40734413509f79fe     // Overflow threshold
}
;;

{.mlx
       ldfe FR_LOG2_10= [ GR_COEFF_START ], 16  // load log2(10)*2^(10-63)
       movl GR_SNORM_LIMIT= 0xc0733a7146f72a41  // Smallest normal threshold
}
{.mib
       nop.m 0
       nop.i 0
 (p12) br.cond.spnt SPECIAL_exp10               // Branch if nan, inf, zero
}
;;

{.mmf
       ldfe FR_L2_10_low= [ GR_COEFF_START ], 16 // load log2(10)_low
       setf.d FR_OF_LIMIT= GR_OF_LIMIT           // Set overflow limit
       fma.s0 f8= f8, f1, f0                     // normalize x
}
;;

{.mfi
       ldfpd FR_COEFF3, FR_COEFF4= [ GR_COEFF_START ], 16 // load C_3, C_4
 (p8)  fcvt.fx.s1 FR_int_x = f8                   // Convert x to integer
       nop.i 0
}
{.mfi
       setf.d FR_UF_LIMIT= GR_UF_LIMIT            // Set underflow limit
       fma.s1 FR_KF0= f8, FR_LOG2_10, FR_ROUNDVAL // y= (x*log2(10)*2^10 +
                                                  //    1.5*2^63) * 2^(-63)
       mov GR_EXP_CORR= 0xffff-126
}
;;

{.mfi
       setf.d FR_SNORM_LIMIT= GR_SNORM_LIMIT      // Set smallest normal limit
       fma.s1 FR_L2_10_high= FR_LOG2_10, FR_2P53, f0 // FR_LOG2_10= log2(10)_hi
       nop.i 0
}
;;

{.mfi
       ldfe FR_COEFF1= [ GR_COEFF_START ], 16    // load C_1
       fms.s1 FR_KF= FR_KF0, f1, FR_ROUNDVAL     // (K+f)*2^(10-63)
       mov GR_MASK= 1023
}
;;

{.mfi
       ldfe FR_COEFF2= [ GR_COEFF_START ], 16    // load C_2
       fma.s1 FR_LOG2_10= f8, FR_L2_10_high, f0  // y0= x*log2(10)_hi
       mov GR_MASK_low= 31
}
;;

{.mlx
       getf.sig GR_KF0= FR_KF0                   // (K+f)*2^10= round_to_int(y)
 (p8)  movl GR_exact_limit= 0x41b00000           // Largest x for exact result,
                                                 //  +22.0
}
;;

{.mfi
       add GR_LOG_TBL= 256, GR_COEFF_START       // Pointer to high T_table
       fcmp.gt.s1 p12, p7= f8, FR_OF_LIMIT       // x>overflow threshold ?
       nop.i 0
}
;;

{.mfi
 (p8)  setf.s FR_exact_limit = GR_exact_limit    // Largest x for exact result
 (p8)  fcvt.xf FR_int_x = FR_int_x               // Integral part of x
       shr GR_K= GR_KF0, 10                      // K
}
{.mfi
       and GR_F_high= GR_MASK, GR_KF0            // f_high*32
       fnma.s1 FR_R= FR_KF, FR_2P53, FR_LOG2_10  // r= x*log2(10)-2^{63-10}*
                                                 //    [ (K+f)*2^{10-63} ]
       and GR_F_low= GR_KF0, GR_MASK_low         // f_low
}
;;

{.mmi
       shladd GR_Flow_ADDR= GR_F_low, 3, GR_COEFF_START // address of 2^{f_low}
       add GR_BIAS= GR_K, GR_EXP_CORR            // K= bias-2*63
       shr GR_Fh= GR_F_high, 5                   // f_high
}
;;

{.mfi
       setf.exp FR_2_TO_K= GR_BIAS               // 2^{K-126}
 (p7)  fcmp.lt.s1 p12, p7= f8, FR_UF_LIMIT       // x<underflow threshold ?
       shladd GR_Fh_ADDR= GR_Fh, 3, GR_LOG_TBL   // address of 2^{f_high}
}
{.mfi
       ldf8 FR_T_low= [ GR_Flow_ADDR ]           // load T_low= 2^{f_low}
       fms.s1 FR_DX_L210= f8, FR_L2_10_high, FR_LOG2_10 // x*log2(10)_hi-
                                                 //        RN(x*log2(10)_hi)
       nop.i 0
}
;;

{.mfi
       ldf8 FR_T_high= [ GR_Fh_ADDR ]            // load T_high= 2^{f_high}
       fma.s1 FR_P34= FR_COEFF4, FR_R, FR_COEFF3 // P34= C_3+C_4*r
       nop.i 0
}
{.mfb
       nop.m 0
       fma.s1 FR_R2= FR_R, FR_R, f0              // r*r
 (p12) br.cond.spnt OUT_RANGE_exp10
}
;;

{.mfi
       nop.m 0
       // e= (x*log2(10)_hi-RN(x*log2(10)_hi))+log2(10)_lo*x
       fma.s1 FR_E0= f8, FR_L2_10_low, FR_DX_L210
       cmp.eq p7,p9= r0,r0                       // Assume inexact result
}
{.mfi
       nop.m 0
       fma.s1 FR_P12= FR_COEFF2, FR_R, FR_COEFF1 // P12= C_1+C_2*r
       nop.i 0
}
;;

{.mfi
       nop.m 0
 (p8)  fcmp.eq.s1 p9,p7= FR_int_x, f8            // Test x positive integer
       nop.i 0
}
{.mfi
       nop.m 0
       fma.s1 FR_T_low_K= FR_T_low, FR_2_TO_K, f0 // T= 2^{K-126}*T_low
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fcmp.ge.s1 p11,p0= f8, FR_SNORM_LIMIT      // Test x for normal range
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fma.s1 FR_E= FR_E0, FR_COEFF1, f0          // E= C_1*e
       nop.i 0
}
{.mfi
       nop.m 0
       fma.s1 FR_P14= FR_R2, FR_P34, FR_P12       // P14= P12+r2*P34
       nop.i 0
}
;;

// If x a positive integer, will it produce an exact result?
//   p7 result will be inexact
//   p9 result will be exact
{.mfi
       nop.m 0
 (p9)  fcmp.le.s1 p9,p7= f8, FR_exact_limit       // Test x gives exact result
       nop.i 0
}
{.mfi
       nop.m 0
       fma.s1 FR_T= FR_T_low_K, FR_T_high, f0     // T= T*T_high
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fma.s1 FR_P= FR_P14, FR_R, FR_E            // P= P14*r+E
       nop.i 0
}
;;

.pred.rel "mutex",p7,p9
{.mfi
       nop.m 0
 (p7)  fma.d.s0 f8= FR_P, FR_T, FR_T              // result= T+T*P, inexact set
       nop.i 0
}
{.mfb
       nop.m 0
 (p9)  fma.d.s1 f8= FR_P, FR_T, FR_T              // result= T+T*P, exact use s1
 (p11) br.ret.sptk b0                             // return, if result normal
}
;;

// Here if result in denormal range (and not zero)
{.mib
       nop.m 0
       mov GR_Parameter_TAG= 265
       br.cond.sptk __libm_error_region           // Branch to error handling
}
;;

SPECIAL_exp10:
{.mfi
       nop.m 0
       fclass.m p6, p0= f8, 0x22                  // x= -Infinity ?
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fclass.m p7, p0= f8, 0x21                  // x= +Infinity ?
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fclass.m p8, p0= f8, 0x7                   // x= +/-Zero ?
       nop.i 0
}
{.mfb
       nop.m 0
 (p6)  mov f8= f0                                 // exp10(-Infinity)= 0
 (p6)  br.ret.spnt b0
}
;;

{.mfb
       nop.m 0
       nop.f 0
 (p7)  br.ret.spnt b0                             // exp10(+Infinity)= +Infinity
}
;;

{.mfb
       nop.m 0
 (p8)  mov f8= f1                                 // exp10(+/-0)= 1
 (p8)  br.ret.spnt b0
}
;;

{.mfb
       nop.m 0
       fma.d.s0 f8= f8, f1, f0                    // Remaining cases: NaNs
       br.ret.sptk b0
}
;;


OUT_RANGE_exp10:

// underflow: p6= 1
// overflow: p8= 1

.pred.rel "mutex",p6,p8
{.mmi
 (p8)  mov GR_EXPMAX= 0x1fffe
 (p6)  mov GR_EXPMAX= 1
       nop.i 0
}
;;

{.mii
       setf.exp FR_R= GR_EXPMAX
 (p8)  mov GR_Parameter_TAG= 166
 (p6)  mov GR_Parameter_TAG= 265
}
;;

{.mfb
       nop.m 0
       fma.d.s0 f8= FR_R, FR_R, f0                // Create overflow/underflow
       br.cond.sptk __libm_error_region           // Branch to error handling
}
;;

GLOBAL_IEEE754_END(exp10)
libm_alias_double_other (__exp10, exp10)
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_27)
compat_symbol (libm, exp10, pow10, GLIBC_2_2)
#endif


LOCAL_LIBM_ENTRY(__libm_error_region)

.prologue
{.mfi
       add GR_Parameter_Y= -32, sp                // Parameter 2 value
       nop.f 0
.save ar.pfs, GR_SAVE_PFS
       mov GR_SAVE_PFS= ar.pfs                    // Save ar.pfs
}

{.mfi
.fframe 64
       add sp= -64, sp                            // Create new stack
       nop.f 0
       mov GR_SAVE_GP= gp                         // Save gp
}
;;

{.mmi
       stfd [ GR_Parameter_Y ]= FR_Y, 16          // STORE Parameter 2 on stack
       add GR_Parameter_X= 16, sp                 // Parameter 1 address
.save b0, GR_SAVE_B0
       mov GR_SAVE_B0= b0                         // Save b0
}
;;

.body
{.mib
       stfd [ GR_Parameter_X ]= FR_X              // STORE Parameter 1 on stack
       add GR_Parameter_RESULT= 0, GR_Parameter_Y // Parameter 3 address
       nop.b 0
}
{.mib
       stfd [ GR_Parameter_Y ]= FR_RESULT         // STORE Parameter 3 on stack
       add GR_Parameter_Y= -16, GR_Parameter_Y
       br.call.sptk b0= __libm_error_support#    // Call error handling function
}
;;

{.mmi
       add GR_Parameter_RESULT= 48, sp
       nop.m 0
       nop.i 0
}
;;

{.mmi
       ldfd f8= [ GR_Parameter_RESULT ]          // Get return result off stack
.restore sp
       add sp= 64, sp                            // Restore stack pointer
       mov b0= GR_SAVE_B0                        // Restore return address
}
;;

{.mib
       mov gp= GR_SAVE_GP                        // Restore gp
       mov ar.pfs= GR_SAVE_PFS                   // Restore ar.pfs
       br.ret.sptk b0                            // Return
}
;;


LOCAL_LIBM_END(__libm_error_region)

.type __libm_error_support#, @function
.global __libm_error_support#
