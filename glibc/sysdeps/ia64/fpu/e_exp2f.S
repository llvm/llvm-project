.file "exp2f.s"


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
// 08/25/00  Initial version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 09/05/02  Improved performance and accuracy
// 01/17/03  Fixed to call error support when x=128.0
// 03/31/05  Reformatted delimiters between data tables
//
// API
//==============================================================
// float exp2f(float)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// Let x= (K + fh + fl + r), where
// K is an integer, fh= 0.b1 b2 b3 b4 b5,
// fl= 2^{-5}* 0.b6 b7 b8 b8 b10 (fh, fl >= 0),
// and |r|<2^{-11}
// Th is a table that stores 2^fh (32 entries) rounded to
// double extended precision (only mantissa is stored)
// Tl is a table that stores 2^fl (32 entries) rounded to
// double extended precision (only mantissa is stored)
//
// 2^x is approximated as
// 2^K * Th [ f ] * Tl [ f ] * (1+c1*r+c2*r^2)

// Note: We use the following trick to speed up conversion from FP to integer:
//
// Let  x = K + r, where K is an integer, and  |r| <= 0.5
// Let N be the number of significand bits for the FP format used
//   ( N=64 for double-extended, N=53 for double)
//
// Then let y = 1.5 * 2^(N-1)  +  x    for RN mode
//          K = y -  1.5 * 2^(N-1)
//          r  = x - K
//
// If we want to obtain the integer part and the first m fractional bits of x,
// we can use the same trick, but with a constant of  1.5 * 2^(N-1-m):
//
// Let x = K + f + r
// f = 0.b_1 b_2 ... b_m
// |r| <= 2^(-m-1)
//
// Then let y = 1.5 * 2^(N-1-m)  +  x    for RN mode
//          (K+f) = y -  1.5 * 2^(N-1-m)
//          r  = x - K


// Special values
//==============================================================
// exp2(0)= 1
// exp2(+inf)= inf
// exp2(-inf)= 0
//

// Registers used
//==============================================================
// r2-r3, r14-r40
// f6-f15, f32-f45
// p6-p8, p12
//


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
GR_EMIN             = r25

GR_ROUNDVAL         = r26
GR_MASK             = r27
GR_KF0              = r28
GR_MASK_low         = r29
GR_COEFF_START      = r30

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

FR_KF0              = f12
FR_UF_LIMIT         = f15

FR_OF_LIMIT         = f32
FR_EXPMIN           = f33
FR_ROUNDVAL         = f34
FR_KF               = f35

FR_2_TO_K           = f36
FR_T_low            = f37
FR_T_high           = f38

FR_P12              = f41
FR_T_low_K          = f42
FR_T                = f44
FR_P                = f45


// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(poly_coeffs)

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
WEAK_LIBM_ENTRY(exp2f)


{.mfi
       alloc r32= ar.pfs, 1, 4, 4, 0
       // will continue only for non-zero normal/denormal numbers
       fclass.nm p12, p0= f8, 0x1b
       // GR_TBL_START= pointer to C_1...C_2 followed by T_table
       addl GR_TBL_START= @ltoff(poly_coeffs), gp
}
{.mlx
       mov GR_OF_LIMIT= 0xffff + 7               // Exponent of overflow limit
       movl GR_ROUNDVAL= 0x5a400000              // 1.5*2^(63-10) (SP)
}
;;

// Form special constant 1.5*2^(63-10) to give integer part and first 10
// fractional bits of x
{.mfi
       setf.s FR_ROUNDVAL= GR_ROUNDVAL           // Form special constant
       fcmp.lt.s1 p6, p8= f8, f0                 // X<0 ?
       nop.i 0
}
{.mfb
       ld8 GR_COEFF_START= [ GR_TBL_START ]      // Load pointer to coeff table
       nop.f 0
 (p12) br.cond.spnt SPECIAL_exp2                 // Branch if nan, inf, zero
}
;;

{.mlx
       setf.exp FR_OF_LIMIT= GR_OF_LIMIT         // Set overflow limit
       movl GR_UF_LIMIT= 0xc3160000              // (-2^7-22) = -150
}
;;

{.mfi
       ldfe FR_COEFF1= [ GR_COEFF_START ], 16    // load C_1
       fma.s0 f8= f8, f1, f0                     // normalize x
       nop.i 0
}
;;

{.mmi
       ldfe FR_COEFF2= [ GR_COEFF_START ], 16    // load C_2
       setf.s FR_UF_LIMIT= GR_UF_LIMIT           // Set underflow limit
       mov GR_EXP_CORR= 0xffff-126
}
;;

{.mfi
       nop.m 0
       fma.s1 FR_KF0= f8, f1, FR_ROUNDVAL        // y= x + 1.5*2^(63-10)
       nop.i 0
}
;;

{.mfi
       mov GR_MASK= 1023
       fms.s1 FR_KF= FR_KF0, f1, FR_ROUNDVAL     // (K+f)
       mov GR_MASK_low= 31
}
;;

{.mfi
       getf.sig GR_KF0= FR_KF0                   // (K+f)*2^10= round_to_int(y)
       fcmp.ge.s1 p12, p7= f8, FR_OF_LIMIT       // x >= overflow threshold ?
       add GR_LOG_TBL= 256, GR_COEFF_START       // Pointer to high T_table
}
;;

{.mmi
       and GR_F_low= GR_KF0, GR_MASK_low         // f_low
       and GR_F_high= GR_MASK, GR_KF0            // f_high*32
       shr GR_K= GR_KF0, 10                      // K
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
       fnma.s1 FR_R= FR_KF, f1, f8               // r= x - (K+f)
       shladd GR_Fh_ADDR= GR_Fh, 3, GR_LOG_TBL   // address of 2^{f_high}
}
{.mlx
       ldf8 FR_T_low= [ GR_Flow_ADDR ]           // load T_low= 2^{f_low}
       movl GR_EMIN= 0xc2fc0000                  // EMIN= -126
}
;;

{.mfi
       ldf8 FR_T_high= [ GR_Fh_ADDR ]            // load T_high= 2^{f_high}
 (p7)  fcmp.lt.s1 p12, p7= f8, FR_UF_LIMIT       // x<underflow threshold ?
       nop.i 0
}
;;

{.mfb
       setf.s FR_EXPMIN= GR_EMIN                 // FR_EXPMIN= EMIN
       fma.s1 FR_P12= FR_COEFF2, FR_R, FR_COEFF1 // P12= C_1+C_2*r
 (p12) br.cond.spnt OUT_RANGE_exp2
}
;;

{.mfi
       nop.m 0
       fma.s1 FR_T_low_K= FR_T_low, FR_2_TO_K, f0 // T= 2^{K-126}*T_low
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fma.s1 FR_P= FR_R, FR_P12, f0              // P= P12+r
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fma.s1 FR_T= FR_T_low_K, FR_T_high, f0     // T= T*T_high
       nop.i 0
}
;;

{.mfi
       nop.m 0
       fcmp.lt.s0 p6, p8= f8, FR_EXPMIN           // underflow (x<EMIN) ?
       nop.i 0
}
;;

{.mfb
       nop.m 0
       fma.s.s0 f8= FR_P, FR_T, FR_T              // result= T+T*P
 (p8)  br.ret.sptk b0                             // return
}
;;

{.mfb
 (p6)  mov GR_Parameter_TAG= 164
       nop.f 0
 (p6)  br.cond.sptk __libm_error_region
}
;;


SPECIAL_exp2:
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
 (p6)  mov f8= f0                                 // exp2(-Infinity)= 0
 (p6)  br.ret.spnt b0
}
;;

{.mfb
       nop.m 0
       nop.f 0
 (p7)  br.ret.spnt b0                             // exp2(+Infinity)= +Infinity
}
;;

{.mfb
       nop.m 0
 (p8)  mov f8= f1                                 // exp2(+/-0)= 1
 (p8)  br.ret.spnt b0
}
;;

{.mfb
       nop.m 0
       fma.s.s0 f8= f8, f1, f0                    // Remaining cases: NaNs
       br.ret.sptk b0
}
;;


OUT_RANGE_exp2:

// overflow: p8= 1

{.mii
 (p8)  mov GR_EXPMAX= 0x1fffe
       nop.i 0
       nop.i 0
}
;;

{.mmb
 (p8)  mov GR_Parameter_TAG= 163
 (p8)  setf.exp FR_R= GR_EXPMAX
       nop.b 999
}
;;

{.mfi
       nop.m 999
 (p8)  fma.s.s0 f8= FR_R, FR_R, f0                // Create overflow
       nop.i 999
}
// underflow: p6= 1
{.mii
 (p6)  mov GR_Parameter_TAG= 164
 (p6)  mov GR_EXPMAX= 1
       nop.i 0
}
;;

{.mmb
       nop.m 0
 (p6)  setf.exp FR_R= GR_EXPMAX
       nop.b 999
}
;;

{.mfb
       nop.m 999
 (p6)  fma.s.s0 f8= FR_R, FR_R, f0                // Create underflow
       nop.b 0
}
;;

WEAK_LIBM_END(exp2f)
libm_alias_float_other (__exp2, exp2)
#ifdef SHARED
.symver exp2f,exp2f@@GLIBC_2.27
.weak __exp2f_compat
.set __exp2f_compat,__exp2f
.symver __exp2f_compat,exp2f@GLIBC_2.2
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
       stfs [ GR_Parameter_Y ]= FR_Y, 16          // STORE Parameter 2 on stack
       add GR_Parameter_X= 16, sp                 // Parameter 1 address
.save b0, GR_SAVE_B0
       mov GR_SAVE_B0= b0                         // Save b0
}
;;

.body
{.mib
       stfs [ GR_Parameter_X ]= FR_X              // STORE Parameter 1 on stack
       add GR_Parameter_RESULT= 0, GR_Parameter_Y // Parameter 3 address
       nop.b 0
}
{.mib
       stfs [ GR_Parameter_Y ]= FR_RESULT         // STORE Parameter 3 on stack
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
       ldfs f8= [ GR_Parameter_RESULT ]          // Get return result off stack
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
