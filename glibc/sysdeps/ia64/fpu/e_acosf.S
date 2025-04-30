.file "acosf.s"


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
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 08/17/00 Changed predicate register macro-usage to direct predicate
//          names due to an assembler bug.
// 10/17/00 Improved speed of x=0 and x=1 paths, set D flag if x denormal.
// 03/13/01 Corrected sign of imm1 value in dep instruction.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 04/17/03 Moved mutex after label


// Description
//=========================================
// The acosf function computes the principle value of the arc sine of x.
// A doman error occurs for arguments not in the range [-1,+1].

// The acosf function returns the arc cosine in the range [0, +pi] radians.
// acos(1) returns +0
// acos(x) returns a Nan and raises the invalid exception for |x| >1

// |x| <= sqrt(2)/2. get Ax and Bx

// poly_p1 = x p1
// poly_p3 = x2 p4 + p3
// poly_p1 = x2 (poly_p1) + x  = x2(x p1) + x
// poly_p2 = x2( poly_p3) + p2 = x2(x2 p4 + p3) + p2

// poly_Ax = x5(x2( poly_p3) + p2) + x2(x p1) + x
//         = x5(x2(x2 p4 + p3) + p2) + x2(x p1) + x

// poly_p7 = x2 p8 + p7
// poly_p5 = x2 p6 + p5

// poly_p7 = x4 p9 + (x2 p8 + p7)
// poly_Bx = x4 (x4 p9 + (x2 p8 + p7)) + x2 p6 + p5

// sinf1 = x11(x4 (x4 p9 + (x2 p8 + p7)) + x2 p6 + p5) + x5(x2(x2 p4 + p3) + p2) + x2(x p1) + x
//       = x19 p9 + x17 p8 + x15 p7 x13 p6 + x11 p5 + x9 p4 + x7 p3 + x5 p2 + x3 p1 + x
// answer1 = pi/2 - sinf1



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


//  answer2 = sign(x) z P(t)       if x>0
//          = sign(x) z P(t) + pi  if x<0


//
// Assembly macros
//=========================================

// predicate registers
//acosf_pred_LEsqrt2by2            = p7
//acosf_pred_GTsqrt2by2            = p8

// integer registers
ACOSF_Addr1                      = r33
ACOSF_Addr2                      = r34
ACOSF_GR_1by2                    = r35

ACOSF_GR_3by2                    = r36
ACOSF_GR_5by2                    = r37

GR_SAVE_B0                    = r38
GR_SAVE_PFS                   = r39
GR_SAVE_GP                    = r40

GR_Parameter_X                = r41
GR_Parameter_Y                = r42
GR_Parameter_RESULT           = r43
GR_Parameter_TAG              = r44

// floating point registers

acosf_y                          = f32
acosf_abs_x                      = f33
acosf_x2                         = f34
acosf_sgn_x                      = f35

acosf_1by2                       = f36
acosf_3by2                       = f37
acosf_5by2                       = f38
acosf_coeff_P3                   = f39
acosf_coeff_P8                   = f40

acosf_coeff_P1                   = f41
acosf_coeff_P4                   = f42
acosf_coeff_P5                   = f43
acosf_coeff_P2                   = f44
acosf_coeff_P7                   = f45

acosf_coeff_P6                   = f46
acosf_coeff_P9                   = f47
acosf_x2                         = f48
acosf_x3                         = f49
acosf_x4                         = f50

acosf_x8                         = f51
acosf_x5                         = f52
acosf_const_piby2                = f53
acosf_const_sqrt2by2             = f54
acosf_x11                        = f55

acosf_poly_p1                    = f56
acosf_poly_p3                    = f57
acosf_sinf1                      = f58
acosf_poly_p2                    = f59
acosf_poly_Ax                    = f60

acosf_poly_p7                    = f61
acosf_poly_p5                    = f62
acosf_sgnx_t4                    = f63
acosf_poly_Bx                    = f64
acosf_t                          = f65

acosf_yby2                       = f66
acosf_B                          = f67
acosf_B2                         = f68
acosf_Az                         = f69
acosf_dz                         = f70

acosf_Sz                         = f71
acosf_d2z                        = f72
acosf_Fz                         = f73
acosf_z                          = f74
acosf_sgnx_z                     = f75

acosf_t2                         = f76
acosf_2poly_p4                   = f77
acosf_2poly_p6                   = f78
acosf_2poly_p1                   = f79
acosf_2poly_p2                   = f80

acosf_2poly_p8                   = f81
acosf_t4                         = f82
acosf_Pt                         = f83
acosf_sgnx_2poly_p2              = f84
acosf_sgn_x_piby2                = f85

acosf_poly_p7a                   = f86
acosf_2poly_p4a                  = f87
acosf_2poly_p4b                  = f88
acosf_2poly_p2a                  = f89
acosf_poly_p1a                   = f90





// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(acosf_coeff_1_table)
data8 0x3FC5555607DCF816 // P1
data8 0x3F9CF81AD9BAB2C6 // P4
data8 0x3FC59E0975074DF3 // P7
data8 0xBFA6F4CC2780AA1D // P6
data8 0x3FC2DD45292E93CB // P9
data8 0x3fe6a09e667f3bcd // sqrt(2)/2
LOCAL_OBJECT_END(acosf_coeff_1_table)

LOCAL_OBJECT_START(acosf_coeff_2_table)
data8 0x3FA6F108E31EFBA6 // P3
data8 0xBFCA31BF175D82A0 // P8
data8 0x3FA30C0337F6418B // P5
data8 0x3FB332C9266CB1F9 // P2
data8 0x3ff921fb54442d18 // pi_by_2
LOCAL_OBJECT_END(acosf_coeff_2_table)


.section .text
GLOBAL_LIBM_ENTRY(acosf)

// Load the addresses of the two tables.
// Then, load the coefficients and other constants.

{     .mfi
     alloc      r32            = ar.pfs,1,8,4,0
     fnma.s1   acosf_t        =    f8,f8,f1
     dep.z ACOSF_GR_1by2 =    0x3f,24,8    // 0x3f000000
}
{     .mfi
     addl ACOSF_Addr1    =    @ltoff(acosf_coeff_1_table),gp
     fma.s1    acosf_x2       =    f8,f8,f0
     addl      ACOSF_Addr2    =    @ltoff(acosf_coeff_2_table),gp ;;
}


{     .mfi
     ld8       ACOSF_Addr1    =    [ACOSF_Addr1]
     fmerge.s  acosf_abs_x    =    f1,f8
     dep ACOSF_GR_3by2 =    -1,r0,22,8     // 0x3fc00000
}
{     .mlx
     nop.m                      999
     movl      ACOSF_GR_5by2  =    0x40200000;;
}



{     .mfi
     setf.s    acosf_1by2     =    ACOSF_GR_1by2
     fmerge.s  acosf_sgn_x    =    f8,f1
     nop.i                      999
}
{     .mfi
     ld8       ACOSF_Addr2    =    [ACOSF_Addr2]
     nop.f 0
     nop.i                      999;;
}


{     .mfi
     setf.s    acosf_5by2     =    ACOSF_GR_5by2
     fcmp.lt.s1 p11,p12 = f8,f0
     nop.i                      999;;
}

{ .mmf
     ldfpd     acosf_coeff_P1,acosf_coeff_P4 =    [ACOSF_Addr1],16
     setf.s    acosf_3by2     =    ACOSF_GR_3by2
     fclass.m.unc p8,p0      = f8, 0xc3 ;;	//@qnan | @snan
}


{     .mfi
     ldfpd     acosf_coeff_P7,acosf_coeff_P6 =    [ACOSF_Addr1],16
     fma.s1    acosf_t2                      =    acosf_t,acosf_t,f0
     nop.i                                     999
}
{     .mfi
     ldfpd     acosf_coeff_P3,acosf_coeff_P8 =    [ACOSF_Addr2],16
     fma.s1    acosf_x4                      =    acosf_x2,acosf_x2,f0
     nop.i                                     999;;
}


{     .mfi
     ldfpd     acosf_coeff_P9,acosf_const_sqrt2by2     =    [ACOSF_Addr1]
     fclass.m.unc p10,p0      = f8, 0x07	//@zero
     nop.i                                     999
}
{     .mfi
     ldfpd     acosf_coeff_P5,acosf_coeff_P2 =    [ACOSF_Addr2],16
     fma.s1    acosf_x3  =    f8,acosf_x2,f0
     nop.i                                     999;;
}


{     .mfi
     ldfd      acosf_const_piby2   =    [ACOSF_Addr2]
     frsqrta.s1     acosf_B,p0                   =    acosf_t
     nop.i                                               999
}
{     .mfb
     nop.m                                               999
(p8) fma.s.s0 f8                = f8,f1,f0
(p8) br.ret.spnt   b0 ;;  // Exit if x=nan
}


{     .mfb
     nop.m                 999
     fcmp.eq.s1 p6,p0 = acosf_abs_x,f1
(p10) br.cond.spnt  ACOSF_ZERO ;;     // Branch if x=0
}

{     .mfi
     nop.m                 999
     fcmp.gt.s1 p9,p0 = acosf_abs_x,f1
     nop.i                 999;;
}

{     .mfi
     nop.m                 999
     fma.s1    acosf_x8  =    acosf_x4,acosf_x4,f0
     nop.i                 999
}
{     .mfb
     nop.m                      999
     fma.s1    acosf_t4  =    acosf_t2,acosf_t2,f0
(p6) br.cond.spnt  ACOSF_ABS_ONE ;;     // Branch if |x|=1
}

{     .mfi
     nop.m                 999
     fma.s1    acosf_x5  =    acosf_x2,acosf_x3,f0
     nop.i                 999
}
{     .mfb
(p9) mov            GR_Parameter_TAG = 59
     fma.s1    acosf_yby2     =    acosf_t,acosf_1by2,f0
(p9) br.cond.spnt  __libm_error_region ;;    // Branch if |x|>1
}


{     .mfi
     nop.m                 999
     fma.s1    acosf_Az  =    acosf_t,acosf_B,f0
     nop.i                 999
}
{     .mfi
     nop.m                 999
     fma.s1    acosf_B2  =    acosf_B,acosf_B,f0
     nop.i                 999;;
}

{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p1  =    f8,acosf_coeff_P1,f0
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_2poly_p1 =    acosf_coeff_P1,acosf_t,f1
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p3  =    acosf_coeff_P4,acosf_x2,acosf_coeff_P3
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_2poly_p6 =    acosf_coeff_P7,acosf_t,acosf_coeff_P6
     nop.i                      999;;
}

{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p7  =    acosf_x2,acosf_coeff_P8,acosf_coeff_P7
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_2poly_p2 =    acosf_coeff_P3,acosf_t,acosf_coeff_P2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p5  =    acosf_x2,acosf_coeff_P6,acosf_coeff_P5
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_2poly_p4 =    acosf_coeff_P5,acosf_t,acosf_coeff_P4
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
     fma.s1    acosf_x11 =    acosf_x8,acosf_x3,f0
     nop.i                 999
}
{     .mfi
     nop.m                 999
     fnma.s1   acosf_dz  =    acosf_B2,acosf_yby2,acosf_1by2
     nop.i                 999;;
}


{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p1a =    acosf_x2,acosf_poly_p1,f8
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_2poly_p8 =    acosf_coeff_P9,acosf_t,acosf_coeff_P8
     nop.i                      999;;
}


// Get the absolute value of x and determine the region in which x lies

{     .mfi
     nop.m                      999
     fcmp.le.s1     p7,p8 = acosf_abs_x,acosf_const_sqrt2by2
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p2  =    acosf_x2,acosf_poly_p3,acosf_coeff_P2
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
     fma.s1    acosf_poly_p7a =    acosf_x4,acosf_coeff_P9,acosf_poly_p7
     nop.i                      999
}
{     .mfi
     nop.m                      999
     fma.s1    acosf_2poly_p2a =    acosf_2poly_p2,acosf_t2,acosf_2poly_p1
     nop.i                      999;;
}


{     .mfi
     nop.m                                                         999
(p8) fma.s1    acosf_sgnx_t4  =    acosf_sgn_x,acosf_t4,f0
     nop.i                                                         999
}
{     .mfi
     nop.m                      999
(p8) fma.s1    acosf_2poly_p4a =    acosf_2poly_p6,acosf_t2,acosf_2poly_p4
     nop.i                      999;;
}


{     .mfi
     nop.m                 999
(p8) fma.s1    acosf_Sz  =    acosf_5by2,acosf_dz,acosf_3by2
     nop.i                 999
}
{     .mfi
     nop.m                 999
(p8) fma.s1    acosf_d2z =    acosf_dz,acosf_dz,f0
     nop.i                 999;;
}


{     .mfi
     nop.m                           999
(p8) fnma.d.s1   acosf_sgn_x_piby2 =    acosf_sgn_x,acosf_const_piby2,acosf_const_piby2
     nop.i                           999
}
{     .mfi
     nop.m                      999
(p7) fma.s1    acosf_poly_Ax  =    acosf_x5,acosf_poly_p2,acosf_poly_p1a
     nop.i                 999;;
}

{     .mfi
     nop.m                      999
(p7) fma.s1    acosf_poly_Bx  =    acosf_x4,acosf_poly_p7a,acosf_poly_p5
     nop.i                      999
}
{     .mfi
     nop.m                           999
(p8) fma.s1    acosf_sgnx_2poly_p2 =    acosf_sgn_x,acosf_2poly_p2a,f0
     nop.i                           999;;
}

{     .mfi
     nop.m                      999
     fcmp.eq.s0 p6,p0 = f8,f0      // Only purpose is to set D if x denormal
     nop.i                      999
}
{     .mfi
     nop.m                      999
(p8) fma.s1    acosf_2poly_p4b =    acosf_2poly_p8,acosf_t4,acosf_2poly_p4a
     nop.i                      999;;
}


{     .mfi
     nop.m                      999
(p8) fma.s1    acosf_Fz  =    acosf_d2z,acosf_Sz,acosf_dz
     nop.i                 999;;
}


{     .mfi
     nop.m                 999
(p8) fma.d.s1  acosf_Pt  =    acosf_2poly_p4b,acosf_sgnx_t4,acosf_sgnx_2poly_p2
     nop.i                 999;;
}

{     .mfi
     nop.m                 999
(p8) fma.d.s1  acosf_z   =    acosf_Az,acosf_Fz,acosf_Az
     nop.i                 999 ;;
}

{     .mfi
     nop.m                      999
(p7) fma.d.s1  acosf_sinf1    =    acosf_x11,acosf_poly_Bx,acosf_poly_Ax
     nop.i                      999;;
}

.pred.rel "mutex",p8,p7    //acosf_pred_GTsqrt2by2,acosf_pred_LEsqrt2by2
{     .mfi
      nop.m            999
(p8)  fma.s.s0     f8   =    acosf_z,acosf_Pt,acosf_sgn_x_piby2
      nop.i            999
}

{     .mfb
      nop.m            999
(p7)  fms.s.s0     f8   =    acosf_const_piby2,f1,acosf_sinf1
      br.ret.sptk b0 ;;
}

ACOSF_ZERO:
// Here if x=0
{     .mfb
      nop.m                 999
      fma.s.s0    f8 =    acosf_const_piby2,f1,f0  // acosf(0)=pi/2
      br.ret.sptk b0 ;;
}


ACOSF_ABS_ONE:
.pred.rel "mutex",p11,p12
// Here if |x|=1
{     .mfi
      nop.m                 999
(p11) fma.s.s0    f8 =    acosf_const_piby2,f1,acosf_const_piby2 // acosf(-1)=pi
      nop.i                 999
}
{     .mfb
      nop.m                 999
(p12) fma.s.s0    f8 =    f1,f0,f0 // acosf(1)=0
      br.ret.sptk b0 ;;
}

GLOBAL_LIBM_END(acosf)
libm_alias_float_other (acos, acos)


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
