.file "asinf.s"


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
// 02/02/00 Initial version
// 06/28/00 Improved speed
// 06/31/00 Changed register allocation because of some duplicate macros
//          moved nan exit bundle up to gain a cycle.
// 08/08/00 Improved speed by avoiding SIR flush.
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 08/17/00 Changed predicate register macro-usage to direct predicate
//          names due to an assembler bug.
// 10/17/00 Improved speed of x=0 and x=1 paths, set D flag if x denormal.
// 03/13/01 Corrected sign of imm1 value in dep instruction.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align


// Description
//=========================================
// The asinf function computes the arc sine of x in the range [-pi,+pi].
// A doman error occurs for arguments not in the range [-1,+1].
// asinf(+-0) returns +-0
// asinf(x) returns a Nan and raises the invalid exception for |x| >1

// The acosf function returns the arc cosine in the range [0, +pi] radians.
// A doman error occurs for arguments not in the range [-1,+1].
// acosf(1) returns +0
// acosf(x) returns a Nan and raises the invalid exception for |x| >1


// |x| <= sqrt(2)/2. get Ax and Bx

// poly_p1 = x p1
// poly_p3 = x2 p4 + p3
// poly_p1 = x2 (poly_p1) + x  = x2(x p1) + x
// poly_p2 = x2( poly_p3) + p2 = x2(x2 p4 + p3) + p2

// poly_Ax = x5(x2( poly_p3) + p2) + x2(x p1) + x
//         = x5(x2(x2 p4 + p3) + p2) + x2(x p1) + x

// poly_p7 = x2 p8 + p7
// poly_p5 = x2 p6 + p5

// poly_p7 = x4 p9 + (poly_p7)
// poly_p7 = x4 p9 + (x2 p8 + p7)
// poly_Bx = x4 (x4 p9 + (x2 p8 + p7)) + x2 p6 + p5

// answer1 = x11(x4 (x4 p9 + (x2 p8 + p7)) + x2 p6 + p5) + x5(x2(x2 p4 + p3) + p2) + x2(x p1) + x
//         = x19 p9 + x17 p8 + x15 p7 x13 p6 + x11 p5 + x9 p4 + x7 p3 + x5 p2 + x3 p1 + x



// |x| >  sqrt(2)/2

// Get z = sqrt(1-x2)

// Get polynomial in t = 1-x2

// t2      = t t
// t4      = t2 t2

// poly_p4 = t p5 + p4
// poly_p1 = t p1 + 1

// poly_p6 = t p7 + p6
// poly_p2 = t p3 + p2

// poly_p8 = t p9 + p8

// poly_p4 = t2 poly_p6 + poly_p4
//         = t2 (t p7 + p6) + (t p5 + p4)

// poly_p2 = t2 poly_p2 + poly_p1
//         = t2 (t p3 + p2) + (t p1 + 1)

// poly_p4 = t4 poly_p8 + poly_p4
//         = t4 (t p9 + p8) + (t2 (t p7 + p6) + (t p5 + p4))

// P(t)    = poly_p2 + t4 poly_p8
//         = t2 (t p3 + p2) + (t p1 + 1) + t4 (t4 (t p9 + p8) + (t2 (t p7 + p6) + (t p5 + p4)))
//         = t3 p3 + t2 p2 + t p1 + 1 + t9 p9 + t8 p8 + t7 p7 + t6 p6 + t5 p5 + t4 p4


//  answer2 = - sign(x) z P(t) + (sign(x) pi/2)
//


// Assembly macros
//=========================================

// predicate registers
//asinf_pred_LEsqrt2by2            = p7
//asinf_pred_GTsqrt2by2            = p8

// integer registers
ASINF_Addr1                      = r33
ASINF_Addr2                      = r34
ASINF_GR_1by2                    = r35

ASINF_GR_3by2                    = r36
ASINF_GR_5by2                    = r37

GR_SAVE_B0                    = r38
GR_SAVE_PFS                   = r39
GR_SAVE_GP                    = r40

GR_Parameter_X                = r41
GR_Parameter_Y                = r42
GR_Parameter_RESULT           = r43
GR_Parameter_TAG              = r44

// floating point registers

asinf_y                          = f32
asinf_abs_x                      = f33
asinf_x2                         = f34
asinf_sgn_x                      = f35

asinf_1by2                       = f36
asinf_3by2                       = f37
asinf_5by2                       = f38
asinf_coeff_P3                   = f39
asinf_coeff_P8                   = f40

asinf_coeff_P1                   = f41
asinf_coeff_P4                   = f42
asinf_coeff_P5                   = f43
asinf_coeff_P2                   = f44
asinf_coeff_P7                   = f45

asinf_coeff_P6                   = f46
asinf_coeff_P9                   = f47
asinf_x2                         = f48
asinf_x3                         = f49
asinf_x4                         = f50

asinf_x8                         = f51
asinf_x5                         = f52
asinf_const_piby2                = f53
asinf_const_sqrt2by2             = f54
asinf_x11                        = f55

asinf_poly_p1                    = f56
asinf_poly_p3                    = f57
asinf_sinf1                      = f58
asinf_poly_p2                    = f59
asinf_poly_Ax                    = f60

asinf_poly_p7                    = f61
asinf_poly_p5                    = f62
asinf_sgnx_t4                    = f63
asinf_poly_Bx                    = f64
asinf_t                          = f65

asinf_yby2                       = f66
asinf_B                          = f67
asinf_B2                         = f68
asinf_Az                         = f69
asinf_dz                         = f70

asinf_Sz                         = f71
asinf_d2z                        = f72
asinf_Fz                         = f73
asinf_z                          = f74
asinf_sgnx_z                     = f75

asinf_t2                         = f76
asinf_2poly_p4                   = f77
asinf_2poly_p6                   = f78
asinf_2poly_p1                   = f79
asinf_2poly_p2                   = f80

asinf_2poly_p8                   = f81
asinf_t4                         = f82
asinf_Pt                         = f83
asinf_sgnx_2poly_p2              = f84
asinf_sgn_x_piby2                = f85

asinf_poly_p7a                   = f86
asinf_2poly_p4a                  = f87
asinf_2poly_p4b                  = f88
asinf_2poly_p2a                  = f89
asinf_poly_p1a                   = f90





// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(asinf_coeff_1_table)
data8 0x3FC5555607DCF816 // P1
data8 0x3F9CF81AD9BAB2C6 // P4
data8 0x3FC59E0975074DF3 // P7
data8 0xBFA6F4CC2780AA1D // P6
data8 0x3FC2DD45292E93CB // P9
data8 0x3fe6a09e667f3bcd // sqrt(2)/2
LOCAL_OBJECT_END(asinf_coeff_1_table)

LOCAL_OBJECT_START(asinf_coeff_2_table)
data8 0x3FA6F108E31EFBA6 // P3
data8 0xBFCA31BF175D82A0 // P8
data8 0x3FA30C0337F6418B // P5
data8 0x3FB332C9266CB1F9 // P2
data8 0x3ff921fb54442d18 // pi_by_2
LOCAL_OBJECT_END(asinf_coeff_2_table)


.section .text
GLOBAL_LIBM_ENTRY(asinf)

// Load the addresses of the two tables.
// Then, load the coefficients and other constants.

{     .mfi
     alloc      r32            = ar.pfs,1,8,4,0
     fnma.s1   asinf_t        =    f8,f8,f1
     dep.z ASINF_GR_1by2 =    0x3f,24,8    // 0x3f000000
}
{     .mfi
     addl ASINF_Addr1    =    @ltoff(asinf_coeff_1_table),gp
     fma.s1    asinf_x2       =    f8,f8,f0
     addl      ASINF_Addr2    =    @ltoff(asinf_coeff_2_table),gp ;;
}


{     .mfi
     ld8       ASINF_Addr1    =    [ASINF_Addr1]
     fmerge.s  asinf_abs_x    =    f1,f8
     dep ASINF_GR_3by2 =    -1,r0,22,8     // 0x3fc00000
}
{     .mlx
     nop.m                      999
     movl      ASINF_GR_5by2  =    0x40200000;;
}



{     .mfi
     setf.s    asinf_1by2     =    ASINF_GR_1by2
     fmerge.s  asinf_sgn_x    =    f8,f1
     nop.i                      999
}
{     .mfi
     ld8       ASINF_Addr2    =    [ASINF_Addr2]
     nop.f 0
     nop.i                      999;;
}


{     .mfi
     setf.s    asinf_5by2     =    ASINF_GR_5by2
     fcmp.lt.s1 p11,p12 = f8,f0
     nop.i                      999;;
}

{ .mmf
     ldfpd     asinf_coeff_P1,asinf_coeff_P4 =    [ASINF_Addr1],16
     setf.s    asinf_3by2     =    ASINF_GR_3by2
     fclass.m.unc p8,p0      = f8, 0xc3 ;;	//@qnan | @snan
}


{     .mfi
     ldfpd     asinf_coeff_P7,asinf_coeff_P6 =    [ASINF_Addr1],16
     fma.s1    asinf_t2                      =    asinf_t,asinf_t,f0
     nop.i                                     999
}
{     .mfi
     ldfpd     asinf_coeff_P3,asinf_coeff_P8 =    [ASINF_Addr2],16
     fma.s1    asinf_x4                      =    asinf_x2,asinf_x2,f0
     nop.i                                     999;;
}


{     .mfi
     ldfpd     asinf_coeff_P9,asinf_const_sqrt2by2     =    [ASINF_Addr1]
     fclass.m.unc p10,p0      = f8, 0x07	//@zero
     nop.i                                     999
}
{     .mfi
     ldfpd     asinf_coeff_P5,asinf_coeff_P2 =    [ASINF_Addr2],16
     fma.s1    asinf_x3  =    f8,asinf_x2,f0
     nop.i                                     999;;
}


{     .mfi
     ldfd      asinf_const_piby2   =    [ASINF_Addr2]
     frsqrta.s1     asinf_B,p0                   =    asinf_t
     nop.i                                               999
}
{     .mfb
     nop.m                                               999
(p8) fma.s.s0 f8                = f8,f1,f0
(p8) br.ret.spnt   b0 ;;  // Exit if x=nan
}


{     .mfb
     nop.m                 999
     fcmp.eq.s1 p6,p0 = asinf_abs_x,f1
(p10) br.ret.spnt  b0 ;;     // Exit if x=0
}

{     .mfi
     nop.m                 999
     fcmp.gt.s1 p9,p0 = asinf_abs_x,f1
     nop.i                 999;;
}

{     .mfi
     nop.m                 999
     fma.s1    asinf_x8  =    asinf_x4,asinf_x4,f0
     nop.i                 999
}
{     .mfb
     nop.m                      999
     fma.s1    asinf_t4  =    asinf_t2,asinf_t2,f0
(p6) br.cond.spnt  ASINF_ABS_ONE ;;     // Branch if |x|=1
}

{     .mfi
     nop.m                 999
     fma.s1    asinf_x5  =    asinf_x2,asinf_x3,f0
     nop.i                 999
}
{     .mfb
(p9) mov            GR_Parameter_TAG = 62
     fma.s1    asinf_yby2     =    asinf_t,asinf_1by2,f0
(p9) br.cond.spnt  __libm_error_region ;;    // Branch if |x|>1
}


{     .mfi
     nop.m                 999
     fma.s1    asinf_Az  =    asinf_t,asinf_B,f0
     nop.i                 999
}
{     .mfi
     nop.m                 999
     fma.s1    asinf_B2  =    asinf_B,asinf_B,f0
     nop.i                 999;;
}

{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p1  =    f8,asinf_coeff_P1,f0
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_2poly_p1 =    asinf_coeff_P1,asinf_t,f1
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p3  =    asinf_coeff_P4,asinf_x2,asinf_coeff_P3
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_2poly_p6 =    asinf_coeff_P7,asinf_t,asinf_coeff_P6
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p7  =    asinf_x2,asinf_coeff_P8,asinf_coeff_P7
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_2poly_p2 =    asinf_coeff_P3,asinf_t,asinf_coeff_P2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p5  =    asinf_x2,asinf_coeff_P6,asinf_coeff_P5
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_2poly_p4 =    asinf_coeff_P5,asinf_t,asinf_coeff_P4
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
     fma.d.s1    asinf_x11 =    asinf_x8,asinf_x3,f0
     nop.i                 999
}
{     .mfi
     nop.m                 999
     fnma.s1   asinf_dz  =    asinf_B2,asinf_yby2,asinf_1by2
     nop.i                 999;;
}


{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p1a =    asinf_x2,asinf_poly_p1,f8
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_2poly_p8 =    asinf_coeff_P9,asinf_t,asinf_coeff_P8
     nop.i                      999;;
}


// Get the absolute value of x and determine the region in which x lies

{     .mfi
     nop.m                      999
     fcmp.le.s1     p7,p8 = asinf_abs_x,asinf_const_sqrt2by2
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p2  =    asinf_x2,asinf_poly_p3,asinf_coeff_P2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    asinf_poly_p7a =    asinf_x4,asinf_coeff_P9,asinf_poly_p7
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    asinf_2poly_p2a =    asinf_2poly_p2,asinf_t2,asinf_2poly_p1
     nop.i                      999;;
}


{     .mfi
     nop.m                                                         999
(p8) fma.s1    asinf_sgnx_t4  =    asinf_sgn_x,asinf_t4,f0
     nop.i                                                         999
}
{     .mfi
     nop.m                      999
(p8) fma.s1    asinf_2poly_p4a =    asinf_2poly_p6,asinf_t2,asinf_2poly_p4
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
(p8) fma.s1    asinf_Sz  =    asinf_5by2,asinf_dz,asinf_3by2
     nop.i                 999
}
{     .mfi
     nop.m                 999
(p8) fma.s1    asinf_d2z =    asinf_dz,asinf_dz,f0
     nop.i                 999;;
}


{     .mfi
     nop.m                           999
(p8) fma.s1   asinf_sgn_x_piby2 =    asinf_sgn_x,asinf_const_piby2,f0
     nop.i                           999
}
{     .mfi
     nop.m                      999
(p7) fma.d.s1    asinf_poly_Ax  =    asinf_x5,asinf_poly_p2,asinf_poly_p1a
     nop.i                 999;;
}

{     .mfi
     nop.m                      999
(p7) fma.d.s1    asinf_poly_Bx  =    asinf_x4,asinf_poly_p7a,asinf_poly_p5
     nop.i                      999
}
{     .mfi
     nop.m                           999
(p8) fma.s1    asinf_sgnx_2poly_p2 =    asinf_sgn_x,asinf_2poly_p2a,f0
     nop.i                           999;;
}

{     .mfi
     nop.m                      999
     fcmp.eq.s0 p6,p0 = f8,f0      // Only purpose is to set D if x denormal
     nop.i                      999
}
{     .mfi
     nop.m                      999
(p8) fma.s1    asinf_2poly_p4b =    asinf_2poly_p8,asinf_t4,asinf_2poly_p4a
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
(p8) fma.s1    asinf_Fz  =    asinf_d2z,asinf_Sz,asinf_dz
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
(p8) fma.d.s1  asinf_Pt  =    asinf_2poly_p4b,asinf_sgnx_t4,asinf_sgnx_2poly_p2
     nop.i                 999;;
}

{     .mfi
     nop.m                 999
(p8) fma.d.s1  asinf_z   =    asinf_Az,asinf_Fz,asinf_Az
     nop.i                 999;;
}

.pred.rel "mutex",p8,p7    //asinf_pred_GTsqrt2by2,asinf_pred_LEsqrt2by2
{     .mfi
                         nop.m            999
(p8)  fnma.s.s0     f8   =    asinf_z,asinf_Pt,asinf_sgn_x_piby2
                         nop.i            999
}

{     .mfb
                         nop.m            999
(p7)  fma.s.s0    f8    =    asinf_x11,asinf_poly_Bx,asinf_poly_Ax
                         br.ret.sptk b0 ;;
}

ASINF_ABS_ONE:
// Here for short exit if |x|=1
{     .mfb
     nop.m                      999
     fma.s.s0    f8 =    asinf_sgn_x,asinf_const_piby2,f0
     br.ret.sptk b0
}
;;

GLOBAL_LIBM_END(asinf)
libm_alias_float_other (asin, asin)

// Stack operations when calling error support.
//       (1)               (2)
//   sp   -> +          psp -> +
//           |                 |
//           |                 | <- GR_Y
//           |                 |
//           | <-GR_Y      Y2->|
//           |                 |
//           |                 | <- GR_X
//           |                 |
//  sp-64 -> +          sp ->  +
//    save ar.pfs          save b0
//    save gp


// Stack operations when calling error support.
//     (3) (call)              (4)
//  psp -> +                   sp -> +
//         |                         |
//    R3 ->| <- GR_RESULT            | -> f8
//         |                         |
//    Y2 ->| <- GR_Y                 |
//         |                         |
//    X1 ->|                         |
//         |                         |
//  sp ->  +                         +
//                              restore gp
//                              restore ar.pfs

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
                nop.f 999
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
        stfs [GR_Parameter_Y] = f1,16         // Store Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;

.body
{ .mfi
        nop.m 0
        frcpa.s0 f9,p0 = f0,f0
        nop.i 0
};;

{ .mib
        stfs [GR_Parameter_X] = f8            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfs [GR_Parameter_Y] = f9           // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
};;
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
