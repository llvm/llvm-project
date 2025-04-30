.file "atanf.s"


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


// History
//==============================================================
// 02/20/00 Initial version
// 08/17/00 Changed predicate register macro-usage to direct predicate
//          names due to an assembler bug.
// 02/06/02 Corrected .section statement
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align;
//          added missing bundling

//
// Assembly macros
//==============================================================

// integer registers used
EXP_Addr1          = r33
EXP_Addr2          = r34

// floating point registers used
atanf_coeff_R4     = f32
atanf_coeff_R5     = f33
atanf_coeff_R1     = f34
atanf_coeff_R2     = f35

atanf_coeff_R3     = f36
atanf_coeff_P1     = f37
atanf_coeff_Q6     = f38
atanf_coeff_Q7     = f39
atanf_coeff_Q8     = f40

atanf_coeff_Q9     = f41
atanf_coeff_Q4     = f42
atanf_coeff_Q5     = f43
atanf_coeff_Q2     = f44
atanf_coeff_Q3     = f45

atanf_coeff_P5     = f46
atanf_coeff_P6     = f47
atanf_coeff_Q0     = f48
atanf_coeff_Q1     = f49
atanf_coeff_P7     = f50

atanf_coeff_P8     = f51
atanf_coeff_P3     = f52
atanf_coeff_P4     = f53
atanf_coeff_P9     = f54
atanf_coeff_P10    = f55

atanf_coeff_P2     = f56
atanf_piby2        = f57
atanf_z            = f58
atanf_b            = f59
atanf_zsq          = f60

atanf_sgn_x        = f61
atanf_sgnx_piby2   = f62
atanf_abs_x        = f63
atanf_t            = f64
atanf_xcub         = f65

atanf_tsq          = f66
atanf_t4           = f67
atanf_x5           = f68
atanf_x6           = f69
atanf_x11          = f70

atanf_poly_p1      = f71
atanf_poly_p2      = f72
atanf_poly_p3      = f73
atanf_poly_p4      = f74
atanf_poly_p5      = f75

atanf_poly_q1      = f76
atanf_poly_q2      = f77
atanf_poly_q3      = f78
atanf_poly_q4      = f79
atanf_poly_q5      = f80

atanf_poly_q       = f81
atanf_poly_r1      = f81
atanf_poly_r2      = f82
atanf_poly_r3      = f83
atanf_bsq          = f84
atanf_z4           = f85

atanf_z5           = f86
atanf_z8           = f87
atanf_z13          = f88
atanf_poly_r2      = f89
atanf_poly_r1      = f90

atanf_z8_bsq       = f91
atanf_poly_r       = f92
atanf_z21_poly_r   = f93
atanf_answer       = f8


// predicate registers used
//atanf_pred_LE1     = p6
//atanf_pred_GT1     = p7


RODATA

.align 16

LOCAL_OBJECT_START(atanf_coeff_1_table)
data8 0x40c4c241be751ff2  // r4
data8 0x40e9f300c2f3070b  // r5
data8 0x409babffef772075  // r3
data8 0xbfd5555512191621  // p1
data8 0x3fc9997e7afbff4e  // p2  = q8
data8 0xbfd5555512191621  // p1  = q9
data8 0x3f97105b4160f86b  // p8  = q2
data8 0xbfa6e10ba401393f  // p7  = q3
data8 0x3f522e5d33bc9baa  // p10 = q0
data8 0xbf7deaadaa336451  // p9  = q1
data8 0xbfc2473c5145ee38  // p3
data8 0x3fbc4f512b1865f5  // p4
data8 0x3fc9997e7afbff4e  // p2
data8 0x3ff921fb54442d18  // pi/2
LOCAL_OBJECT_END(atanf_coeff_1_table)



LOCAL_OBJECT_START(atanf_coeff_2_table)
data8 0x4035000000004284  // r1
data8 0x406cdffff336a59b  // r2
data8 0x3fbc4f512b1865f5  // p4 = q6
data8 0xbfc2473c5145ee38  // p3 = q7
data8 0x3fb142a73d7c54e3  // p6 = q4
data8 0xbfb68eed6a8cfa32  // p5 = q5
data8 0xbfb68eed6a8cfa32  // p5
data8 0x3fb142a73d7c54e3  // p6
data8 0xbfa6e10ba401393f  // p7
data8 0x3f97105b4160f86b  // p8
data8 0xbf7deaadaa336451  // p9
data8 0x3f522e5d33bc9baa  // p10
LOCAL_OBJECT_END(atanf_coeff_2_table)



.section .text
GLOBAL_LIBM_ENTRY(atanf)

{     .mfi
     alloc      r32                  = ar.pfs,1,2,0,0
     frcpa.s1  atanf_z,p0   =    f1,f8
     addl      EXP_Addr2           =    @ltoff(atanf_coeff_2_table),gp
}
{     .mfi
     addl      EXP_Addr1 =    @ltoff(atanf_coeff_1_table),gp
     fma.s1    atanf_t   =    f8,f8,f0
     nop.i                 999;;
}


{     .mfi
     nop.m                      999
     fmerge.s  atanf_sgn_x    =    f8,f1
     nop.i                      999;;
}

{     .mfi
     ld8       EXP_Addr1      =    [EXP_Addr1]
     fmerge.s  atanf_abs_x    =    f1,f8
     nop.i                      999
}
{     .mfi
     ld8       EXP_Addr2 =    [EXP_Addr2]
     nop.f                 999
     nop.i                 999;;
}


{     .mfi
     nop.m                      999
     fclass.m  p8,p0   =    f8,0x7	// @zero
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fcmp.eq.unc.s0 p9,p10    =    f8,f1
     nop.i                      999;;
}

{     .mfi
     ldfpd     atanf_coeff_R4,atanf_coeff_R5 =    [EXP_Addr1],16
     fnma.s1   atanf_b                       =    f8,atanf_z,f1
     nop.i                                     999
}
{     .mfi
     ldfpd     atanf_coeff_R1,atanf_coeff_R2 =    [EXP_Addr2],16
     fma.s1    atanf_zsq                     =    atanf_z,atanf_z,f0
     nop.i                                     999;;
}


{     .mfi
     ldfpd     atanf_coeff_R3,atanf_coeff_P1 =    [EXP_Addr1],16
     fma.s1    atanf_xcub                    =    f8,atanf_t,f0
     nop.i                                     999
}
{     .mfi
     ldfpd     atanf_coeff_Q6,atanf_coeff_Q7 =    [EXP_Addr2],16
     fma.s1    atanf_tsq                     =    atanf_t,atanf_t,f0
     nop.i                                     999;;
}


{     .mfi
     ldfpd          atanf_coeff_Q8,atanf_coeff_Q9 =    [EXP_Addr1],16
//     fcmp.le.s1     atanf_pred_LE1,atanf_pred_GT1 =    atanf_abs_x,f1
     fcmp.le.s1     p6,p7 =    atanf_abs_x,f1
     nop.i                                          999
}
{     .mfi
     ldfpd     atanf_coeff_Q4,atanf_coeff_Q5 =    [EXP_Addr2],16
     nop.f                                     999
     nop.i                                     999;;
}


{     .mfi
     ldfpd     atanf_coeff_Q2,atanf_coeff_Q3 =    [EXP_Addr1],16
     fclass.m  p8,p0                  =    f8,0xe7	// @inf|@qnan|@snan|@zero
     nop.i                                     999
}
{     .mfi
     ldfpd     atanf_coeff_P5,atanf_coeff_P6 =    [EXP_Addr2],16
     nop.f                                     999
     nop.i                                     999;;
}


{     .mfi
     ldfpd     atanf_coeff_Q0,atanf_coeff_Q1 =    [EXP_Addr1],16
     nop.f                                     999
     nop.i                                     999
}
{     .mfi
     ldfpd     atanf_coeff_P7,atanf_coeff_P8 =    [EXP_Addr2],16
     nop.f                                     999
     nop.i                                     999;;
}


{     .mfi
     ldfpd     atanf_coeff_P3,atanf_coeff_P4 =    [EXP_Addr1],16
     fma.s1    atanf_bsq                     =    atanf_b,atanf_b,f0
     nop.i                                     999
}
{     .mfi
     ldfpd     atanf_coeff_P9,atanf_coeff_P10     =    [EXP_Addr2]
     fma.s1    atanf_z4                           =    atanf_zsq,atanf_zsq,f0
     nop.i                                          999;;
}


{     .mfi
     ldfpd     atanf_coeff_P2,atanf_piby2    =    [EXP_Addr1]
     fma.s1    atanf_x6                      =    atanf_t,atanf_tsq,f0
     nop.i                                     999
}
{     .mfi
     nop.m                 999
     fma.s1    atanf_t4  =    atanf_tsq,atanf_tsq,f0
     nop.i                 999;;
}


{     .mfb
     nop.m                 999
     fma.s1    atanf_x5  =    atanf_t,atanf_xcub,f0
(p8)  br.cond.spnt   ATANF_X_INF_NAN_ZERO
}
;;

{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_r1  =    atanf_b,atanf_coeff_R1,f1
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_r3  =    atanf_b,atanf_coeff_R5,atanf_coeff_R4
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_r2  =    atanf_b,atanf_coeff_R3,atanf_coeff_R2
     nop.i                      999
}
{     .mfi
     nop.m                 999
     fma.s1    atanf_z8  =    atanf_z4,atanf_z4,f0
     nop.i                 999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q2  =    atanf_t,atanf_coeff_Q5,atanf_coeff_Q4
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q3  =    atanf_t,atanf_coeff_Q7,atanf_coeff_Q6
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
     fma.s1    atanf_z5  =    atanf_z,atanf_z4,f0
     nop.i                 999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q1  =    atanf_t,atanf_coeff_Q9,atanf_coeff_Q8
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q4  =    atanf_t,atanf_coeff_Q1,atanf_coeff_Q0
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q5  =    atanf_t,atanf_coeff_Q3,atanf_coeff_Q2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p4  =    f8,atanf_coeff_P1,f0
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p5  =    atanf_t,atanf_coeff_P4,atanf_coeff_P3
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_r1  =    atanf_z8,atanf_poly_r1,f0
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_z8_bsq   =    atanf_z8,atanf_bsq,f0
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q2  =    atanf_tsq,atanf_poly_q3,atanf_poly_q2
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_r2  =    atanf_bsq,atanf_poly_r3,atanf_poly_r2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p2  =    atanf_t,atanf_coeff_P8,atanf_coeff_P7
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q1  =    atanf_poly_q1,f1,atanf_tsq
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
     fma.s1    atanf_z13 =    atanf_z5,atanf_z8,f0
     nop.i                 999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p1  =    atanf_t,atanf_coeff_P10,atanf_coeff_P9
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p4  =    atanf_t,atanf_poly_p4,f8
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q4  =    atanf_tsq,atanf_poly_q5,atanf_poly_q4
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p3  =    atanf_t,atanf_coeff_P6,atanf_coeff_P5
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p5  =    atanf_t,atanf_poly_p5,atanf_coeff_P2
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
     fma.s1    atanf_x11 =    atanf_x5,atanf_x6,f0
     nop.i                 999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_r   =    atanf_z8_bsq,atanf_poly_r2,atanf_poly_r1
     nop.i                      999;;
}


{     .mfi
     nop.m                           999
     fma.s0       atanf_sgnx_piby2    =    atanf_sgn_x,atanf_piby2,f0
     nop.i                           999
}
{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q2  =    atanf_t4,atanf_poly_q1,atanf_poly_q2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p1  =    atanf_tsq,atanf_poly_p1,atanf_poly_p2
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p4  =    atanf_x5,atanf_poly_p5,atanf_poly_p4
     nop.i                      999;;
}

{     .mfi
     nop.m                           999
     fma.s1    atanf_z21_poly_r    =    atanf_z13,atanf_poly_r,f0
     nop.i                           999;;
}

{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_q   =    atanf_t4,atanf_poly_q2,atanf_poly_q4
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fma.s1    atanf_poly_p1  =    atanf_tsq,atanf_poly_p1,atanf_poly_p3
     nop.i                      999;;
}

{     .mfi
                    nop.m                      999
//(atanf_pred_GT1)    fnma.s    atanf_answer   =    atanf_poly_q,atanf_z21_poly_r,atanf_sgnx_piby2
(p7)    fnma.s.s0    atanf_answer   =    atanf_poly_q,atanf_z21_poly_r,atanf_sgnx_piby2
                    nop.i                      999;;
}

{     .mfb
                    nop.m                      999
//(atanf_pred_LE1)    fma.s     atanf_answer   =    atanf_x11,atanf_poly_p1,atanf_poly_p4
(p6)    fma.s.s0     atanf_answer   =    atanf_x11,atanf_poly_p1,atanf_poly_p4
                    br.ret.sptk b0
}



ATANF_X_INF_NAN_ZERO:

{ .mfi
      nop.m 0
      fclass.m p8,p9 = f8,0x23  // @inf
      nop.i 0
}
;;
{ .mfi
      nop.m 0
(p8)  fmerge.s f8 = f8, atanf_piby2
      nop.i 0
}
;;
{ .mfb
      nop.m 0
      fnorm.s.s0 f8 = f8
      br.ret.sptk b0
}
;;

GLOBAL_LIBM_END(atanf)
libm_alias_float_other (atan, atan)
