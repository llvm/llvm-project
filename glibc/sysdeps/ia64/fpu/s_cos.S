.file "sincos.s"


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
// 02/02/00 Initial version
// 04/02/00 Unwind support added.
// 06/16/00 Updated tables to enforce symmetry
// 08/31/00 Saved 2 cycles in main path, and 9 in other paths.
// 09/20/00 The updated tables regressed to an old version, so reinstated them
// 10/18/00 Changed one table entry to ensure symmetry
// 01/03/01 Improved speed, fixed flag settings for small arguments.
// 02/18/02 Large arguments processing routine excluded
// 05/20/02 Cleaned up namespace and sf0 syntax
// 06/03/02 Insure inexact flag set for large arg result
// 09/05/02 Work range is widened by reduction strengthen (3 parts of Pi/16)
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 08/08/03 Improved performance
// 10/28/04 Saved sincos_r_sincos to avoid clobber by dynamic loader
// 03/31/05 Reformatted delimiters between data tables

// API
//==============================================================
// double sin( double x);
// double cos( double x);
//
// Overview of operation
//==============================================================
//
// Step 1
// ======
// Reduce x to region -1/2*pi/2^k ===== 0 ===== +1/2*pi/2^k  where k=4
//    divide x by pi/2^k.
//    Multiply by 2^k/pi.
//    nfloat = Round result to integer (round-to-nearest)
//
// r = x -  nfloat * pi/2^k
//    Do this as ((((x -  nfloat * HIGH(pi/2^k))) -
//                        nfloat * LOW(pi/2^k)) -
//                        nfloat * LOWEST(pi/2^k) for increased accuracy.
//    pi/2^k is stored as two numbers that when added make pi/2^k.
//       pi/2^k = HIGH(pi/2^k) + LOW(pi/2^k)
//    HIGH and LOW parts are rounded to zero values,
//    and LOWEST is rounded to nearest one.
//
// x = (nfloat * pi/2^k) + r
//    r is small enough that we can use a polynomial approximation
//    and is referred to as the reduced argument.
//
// Step 3
// ======
// Take the unreduced part and remove the multiples of 2pi.
// So nfloat = nfloat (with lower k+1 bits cleared) + lower k+1 bits
//
//    nfloat (with lower k+1 bits cleared) is a multiple of 2^(k+1)
//    N * 2^(k+1)
//    nfloat * pi/2^k = N * 2^(k+1) * pi/2^k + (lower k+1 bits) * pi/2^k
//    nfloat * pi/2^k = N * 2 * pi + (lower k+1 bits) * pi/2^k
//    nfloat * pi/2^k = N2pi + M * pi/2^k
//
//
// Sin(x) = Sin((nfloat * pi/2^k) + r)
//        = Sin(nfloat * pi/2^k) * Cos(r) + Cos(nfloat * pi/2^k) * Sin(r)
//
//          Sin(nfloat * pi/2^k) = Sin(N2pi + Mpi/2^k)
//                               = Sin(N2pi)Cos(Mpi/2^k) + Cos(N2pi)Sin(Mpi/2^k)
//                               = Sin(Mpi/2^k)
//
//          Cos(nfloat * pi/2^k) = Cos(N2pi + Mpi/2^k)
//                               = Cos(N2pi)Cos(Mpi/2^k) + Sin(N2pi)Sin(Mpi/2^k)
//                               = Cos(Mpi/2^k)
//
// Sin(x) = Sin(Mpi/2^k) Cos(r) + Cos(Mpi/2^k) Sin(r)
//
//
// Step 4
// ======
// 0 <= M < 2^(k+1)
// There are 2^(k+1) Sin entries in a table.
// There are 2^(k+1) Cos entries in a table.
//
// Get Sin(Mpi/2^k) and Cos(Mpi/2^k) by table lookup.
//
//
// Step 5
// ======
// Calculate Cos(r) and Sin(r) by polynomial approximation.
//
// Cos(r) = 1 + r^2 q1  + r^4 q2 + r^6 q3 + ... = Series for Cos
// Sin(r) = r + r^3 p1  + r^5 p2 + r^7 p3 + ... = Series for Sin
//
// and the coefficients q1, q2, ... and p1, p2, ... are stored in a table
//
//
// Calculate
// Sin(x) = Sin(Mpi/2^k) Cos(r) + Cos(Mpi/2^k) Sin(r)
//
// as follows
//
//    S[m] = Sin(Mpi/2^k) and C[m] = Cos(Mpi/2^k)
//    rsq = r*r
//
//
//    P = p1 + r^2p2 + r^4p3 + r^6p4
//    Q = q1 + r^2q2 + r^4q3 + r^6q4
//
//       rcub = r * rsq
//       Sin(r) = r + rcub * P
//              = r + r^3p1  + r^5p2 + r^7p3 + r^9p4 + ... = Sin(r)
//
//            The coefficients are not exactly these values, but almost.
//
//            p1 = -1/6  = -1/3!
//            p2 = 1/120 =  1/5!
//            p3 = -1/5040 = -1/7!
//            p4 = 1/362889 = 1/9!
//
//       P =  r + rcub * P
//
//    Answer = S[m] Cos(r) + [Cm] P
//
//       Cos(r) = 1 + rsq Q
//       Cos(r) = 1 + r^2 Q
//       Cos(r) = 1 + r^2 (q1 + r^2q2 + r^4q3 + r^6q4)
//       Cos(r) = 1 + r^2q1 + r^4q2 + r^6q3 + r^8q4 + ...
//
//       S[m] Cos(r) = S[m](1 + rsq Q)
//       S[m] Cos(r) = S[m] + Sm rsq Q
//       S[m] Cos(r) = S[m] + s_rsq Q
//       Q         = S[m] + s_rsq Q
//
// Then,
//
//    Answer = Q + C[m] P


// Registers used
//==============================================================
// general input registers:
// r14 -> r26
// r32 -> r35

// predicate registers used:
// p6 -> p11

// floating-point registers used
// f9 -> f15
// f32 -> f61

// Assembly macros
//==============================================================
sincos_NORM_f8                 = f9
sincos_W                       = f10
sincos_int_Nfloat              = f11
sincos_Nfloat                  = f12

sincos_r                       = f13
sincos_rsq                     = f14
sincos_rcub                    = f15
sincos_save_tmp                = f15

sincos_Inv_Pi_by_16            = f32
sincos_Pi_by_16_1              = f33
sincos_Pi_by_16_2              = f34

sincos_Inv_Pi_by_64            = f35

sincos_Pi_by_16_3              = f36

sincos_r_exact                 = f37

sincos_Sm                      = f38
sincos_Cm                      = f39

sincos_P1                      = f40
sincos_Q1                      = f41
sincos_P2                      = f42
sincos_Q2                      = f43
sincos_P3                      = f44
sincos_Q3                      = f45
sincos_P4                      = f46
sincos_Q4                      = f47

sincos_P_temp1                 = f48
sincos_P_temp2                 = f49

sincos_Q_temp1                 = f50
sincos_Q_temp2                 = f51

sincos_P                       = f52
sincos_Q                       = f53

sincos_srsq                    = f54

sincos_SIG_INV_PI_BY_16_2TO61  = f55
sincos_RSHF_2TO61              = f56
sincos_RSHF                    = f57
sincos_2TOM61                  = f58
sincos_NFLOAT                  = f59
sincos_W_2TO61_RSH             = f60

fp_tmp                         = f61

/////////////////////////////////////////////////////////////

sincos_GR_sig_inv_pi_by_16     = r14
sincos_GR_rshf_2to61           = r15
sincos_GR_rshf                 = r16
sincos_GR_exp_2tom61           = r17
sincos_GR_n                    = r18
sincos_GR_m                    = r19
sincos_GR_32m                  = r19
sincos_GR_all_ones             = r19
sincos_AD_1                    = r20
sincos_AD_2                    = r21
sincos_exp_limit               = r22
sincos_r_signexp               = r23
sincos_r_17_ones               = r24
sincos_r_sincos                = r25
sincos_r_exp                   = r26

GR_SAVE_PFS                    = r33
GR_SAVE_B0                     = r34
GR_SAVE_GP                     = r35
GR_SAVE_r_sincos               = r36


RODATA

// Pi/16 parts
.align 16
LOCAL_OBJECT_START(double_sincos_pi)
   data8 0xC90FDAA22168C234, 0x00003FFC // pi/16 1st part
   data8 0xC4C6628B80DC1CD1, 0x00003FBC // pi/16 2nd part
   data8 0xA4093822299F31D0, 0x00003F7A // pi/16 3rd part
LOCAL_OBJECT_END(double_sincos_pi)

// Coefficients for polynomials
LOCAL_OBJECT_START(double_sincos_pq_k4)
   data8 0x3EC71C963717C63A // P4
   data8 0x3EF9FFBA8F191AE6 // Q4
   data8 0xBF2A01A00F4E11A8 // P3
   data8 0xBF56C16C05AC77BF // Q3
   data8 0x3F8111111110F167 // P2
   data8 0x3FA555555554DD45 // Q2
   data8 0xBFC5555555555555 // P1
   data8 0xBFDFFFFFFFFFFFFC // Q1
LOCAL_OBJECT_END(double_sincos_pq_k4)

// Sincos table (S[m], C[m])
LOCAL_OBJECT_START(double_sin_cos_beta_k4)

data8 0x0000000000000000 , 0x00000000 // sin( 0 pi/16)  S0
data8 0x8000000000000000 , 0x00003fff // cos( 0 pi/16)  C0
//
data8 0xc7c5c1e34d3055b3 , 0x00003ffc // sin( 1 pi/16)  S1
data8 0xfb14be7fbae58157 , 0x00003ffe // cos( 1 pi/16)  C1
//
data8 0xc3ef1535754b168e , 0x00003ffd // sin( 2 pi/16)  S2
data8 0xec835e79946a3146 , 0x00003ffe // cos( 2 pi/16)  C2
//
data8 0x8e39d9cd73464364 , 0x00003ffe // sin( 3 pi/16)  S3
data8 0xd4db3148750d181a , 0x00003ffe // cos( 3 pi/16)  C3
//
data8 0xb504f333f9de6484 , 0x00003ffe // sin( 4 pi/16)  S4
data8 0xb504f333f9de6484 , 0x00003ffe // cos( 4 pi/16)  C4
//
data8 0xd4db3148750d181a , 0x00003ffe // sin( 5 pi/16)  C3
data8 0x8e39d9cd73464364 , 0x00003ffe // cos( 5 pi/16)  S3
//
data8 0xec835e79946a3146 , 0x00003ffe // sin( 6 pi/16)  C2
data8 0xc3ef1535754b168e , 0x00003ffd // cos( 6 pi/16)  S2
//
data8 0xfb14be7fbae58157 , 0x00003ffe // sin( 7 pi/16)  C1
data8 0xc7c5c1e34d3055b3 , 0x00003ffc // cos( 7 pi/16)  S1
//
data8 0x8000000000000000 , 0x00003fff // sin( 8 pi/16)  C0
data8 0x0000000000000000 , 0x00000000 // cos( 8 pi/16)  S0
//
data8 0xfb14be7fbae58157 , 0x00003ffe // sin( 9 pi/16)  C1
data8 0xc7c5c1e34d3055b3 , 0x0000bffc // cos( 9 pi/16)  -S1
//
data8 0xec835e79946a3146 , 0x00003ffe // sin(10 pi/16)  C2
data8 0xc3ef1535754b168e , 0x0000bffd // cos(10 pi/16)  -S2
//
data8 0xd4db3148750d181a , 0x00003ffe // sin(11 pi/16)  C3
data8 0x8e39d9cd73464364 , 0x0000bffe // cos(11 pi/16)  -S3
//
data8 0xb504f333f9de6484 , 0x00003ffe // sin(12 pi/16)  S4
data8 0xb504f333f9de6484 , 0x0000bffe // cos(12 pi/16)  -S4
//
data8 0x8e39d9cd73464364 , 0x00003ffe // sin(13 pi/16) S3
data8 0xd4db3148750d181a , 0x0000bffe // cos(13 pi/16) -C3
//
data8 0xc3ef1535754b168e , 0x00003ffd // sin(14 pi/16) S2
data8 0xec835e79946a3146 , 0x0000bffe // cos(14 pi/16) -C2
//
data8 0xc7c5c1e34d3055b3 , 0x00003ffc // sin(15 pi/16) S1
data8 0xfb14be7fbae58157 , 0x0000bffe // cos(15 pi/16) -C1
//
data8 0x0000000000000000 , 0x00000000 // sin(16 pi/16) S0
data8 0x8000000000000000 , 0x0000bfff // cos(16 pi/16) -C0
//
data8 0xc7c5c1e34d3055b3 , 0x0000bffc // sin(17 pi/16) -S1
data8 0xfb14be7fbae58157 , 0x0000bffe // cos(17 pi/16) -C1
//
data8 0xc3ef1535754b168e , 0x0000bffd // sin(18 pi/16) -S2
data8 0xec835e79946a3146 , 0x0000bffe // cos(18 pi/16) -C2
//
data8 0x8e39d9cd73464364 , 0x0000bffe // sin(19 pi/16) -S3
data8 0xd4db3148750d181a , 0x0000bffe // cos(19 pi/16) -C3
//
data8 0xb504f333f9de6484 , 0x0000bffe // sin(20 pi/16) -S4
data8 0xb504f333f9de6484 , 0x0000bffe // cos(20 pi/16) -S4
//
data8 0xd4db3148750d181a , 0x0000bffe // sin(21 pi/16) -C3
data8 0x8e39d9cd73464364 , 0x0000bffe // cos(21 pi/16) -S3
//
data8 0xec835e79946a3146 , 0x0000bffe // sin(22 pi/16) -C2
data8 0xc3ef1535754b168e , 0x0000bffd // cos(22 pi/16) -S2
//
data8 0xfb14be7fbae58157 , 0x0000bffe // sin(23 pi/16) -C1
data8 0xc7c5c1e34d3055b3 , 0x0000bffc // cos(23 pi/16) -S1
//
data8 0x8000000000000000 , 0x0000bfff // sin(24 pi/16) -C0
data8 0x0000000000000000 , 0x00000000 // cos(24 pi/16) S0
//
data8 0xfb14be7fbae58157 , 0x0000bffe // sin(25 pi/16) -C1
data8 0xc7c5c1e34d3055b3 , 0x00003ffc // cos(25 pi/16) S1
//
data8 0xec835e79946a3146 , 0x0000bffe // sin(26 pi/16) -C2
data8 0xc3ef1535754b168e , 0x00003ffd // cos(26 pi/16) S2
//
data8 0xd4db3148750d181a , 0x0000bffe // sin(27 pi/16) -C3
data8 0x8e39d9cd73464364 , 0x00003ffe // cos(27 pi/16) S3
//
data8 0xb504f333f9de6484 , 0x0000bffe // sin(28 pi/16) -S4
data8 0xb504f333f9de6484 , 0x00003ffe // cos(28 pi/16) S4
//
data8 0x8e39d9cd73464364 , 0x0000bffe // sin(29 pi/16) -S3
data8 0xd4db3148750d181a , 0x00003ffe // cos(29 pi/16) C3
//
data8 0xc3ef1535754b168e , 0x0000bffd // sin(30 pi/16) -S2
data8 0xec835e79946a3146 , 0x00003ffe // cos(30 pi/16) C2
//
data8 0xc7c5c1e34d3055b3 , 0x0000bffc // sin(31 pi/16) -S1
data8 0xfb14be7fbae58157 , 0x00003ffe // cos(31 pi/16) C1
//
data8 0x0000000000000000 , 0x00000000 // sin(32 pi/16) S0
data8 0x8000000000000000 , 0x00003fff // cos(32 pi/16) C0
LOCAL_OBJECT_END(double_sin_cos_beta_k4)

.section .text

////////////////////////////////////////////////////////
// There are two entry points: sin and cos


// If from sin, p8 is true
// If from cos, p9 is true

GLOBAL_IEEE754_ENTRY(sin)

{ .mlx
      getf.exp      sincos_r_signexp    = f8
      movl sincos_GR_sig_inv_pi_by_16   = 0xA2F9836E4E44152A // signd of 16/pi
}
{ .mlx
      addl          sincos_AD_1         = @ltoff(double_sincos_pi), gp
      movl sincos_GR_rshf_2to61         = 0x47b8000000000000 // 1.1 2^(63+63-2)
}
;;

{ .mfi
      ld8           sincos_AD_1         = [sincos_AD_1]
      fnorm.s0      sincos_NORM_f8      = f8  // Normalize argument
      cmp.eq        p8,p9               = r0, r0 // set p8 (clear p9) for sin
}
{ .mib
      mov           sincos_GR_exp_2tom61  = 0xffff-61 // exponent of scale 2^-61
      mov           sincos_r_sincos       = 0x0 // sincos_r_sincos = 0 for sin
      br.cond.sptk  _SINCOS_COMMON  // go to common part
}
;;

GLOBAL_IEEE754_END(sin)
libm_alias_double_other (__sin, sin)

GLOBAL_IEEE754_ENTRY(cos)

{ .mlx
      getf.exp      sincos_r_signexp    = f8
      movl sincos_GR_sig_inv_pi_by_16   = 0xA2F9836E4E44152A // signd of 16/pi
}
{ .mlx
      addl          sincos_AD_1         = @ltoff(double_sincos_pi), gp
      movl sincos_GR_rshf_2to61         = 0x47b8000000000000 // 1.1 2^(63+63-2)
}
;;

{ .mfi
      ld8           sincos_AD_1         = [sincos_AD_1]
      fnorm.s1      sincos_NORM_f8      = f8 // Normalize argument
      cmp.eq        p9,p8               = r0, r0 // set p9 (clear p8) for cos
}
{ .mib
      mov           sincos_GR_exp_2tom61  = 0xffff-61 // exp of scale 2^-61
      mov           sincos_r_sincos       = 0x8 // sincos_r_sincos = 8 for cos
      nop.b         999
}
;;

////////////////////////////////////////////////////////
// All entry points end up here.
// If from sin, sincos_r_sincos is 0 and p8 is true
// If from cos, sincos_r_sincos is 8 = 2^(k-1) and p9 is true
// We add sincos_r_sincos to N

///////////// Common sin and cos part //////////////////
_SINCOS_COMMON:


// Form two constants we need
//  16/pi * 2^-2 * 2^63, scaled by 2^61 since we just loaded the significand
//  1.1000...000 * 2^(63+63-2) to right shift int(W) into the low significand
{ .mfi
      setf.sig      sincos_SIG_INV_PI_BY_16_2TO61 = sincos_GR_sig_inv_pi_by_16
      fclass.m      p6,p0                         = f8, 0xe7 // if x = 0,inf,nan
      mov           sincos_exp_limit              = 0x1001a
}
{ .mlx
      setf.d        sincos_RSHF_2TO61   = sincos_GR_rshf_2to61
      movl          sincos_GR_rshf      = 0x43e8000000000000 // 1.1 2^63
}                                                            // Right shift
;;

// Form another constant
//  2^-61 for scaling Nfloat
// 0x1001a is register_bias + 27.
// So if f8 >= 2^27, go to large argument routines
{ .mfi
      alloc         r32                 = ar.pfs, 1, 4, 0, 0
      fclass.m      p11,p0              = f8, 0x0b // Test for x=unorm
      mov           sincos_GR_all_ones  = -1 // For "inexect" constant create
}
{ .mib
      setf.exp      sincos_2TOM61       = sincos_GR_exp_2tom61
      nop.i         999
(p6)  br.cond.spnt  _SINCOS_SPECIAL_ARGS
}
;;

// Load the two pieces of pi/16
// Form another constant
//  1.1000...000 * 2^63, the right shift constant
{ .mmb
      ldfe          sincos_Pi_by_16_1   = [sincos_AD_1],16
      setf.d        sincos_RSHF         = sincos_GR_rshf
(p11) br.cond.spnt  _SINCOS_UNORM       // Branch if x=unorm
}
;;

_SINCOS_COMMON2:
// Return here if x=unorm
// Create constant used to set inexact
{ .mmi
      ldfe          sincos_Pi_by_16_2   = [sincos_AD_1],16
      setf.sig      fp_tmp              = sincos_GR_all_ones
      nop.i         999
};;

// Select exponent (17 lsb)
{ .mfi
      ldfe          sincos_Pi_by_16_3   = [sincos_AD_1],16
      nop.f         999
      dep.z         sincos_r_exp        = sincos_r_signexp, 0, 17
};;

// Polynomial coefficients (Q4, P4, Q3, P3, Q2, Q1, P2, P1) loading
// p10 is true if we must call routines to handle larger arguments
// p10 is true if f8 exp is >= 0x1001a (2^27)
{ .mmb
      ldfpd         sincos_P4,sincos_Q4 = [sincos_AD_1],16
      cmp.ge        p10,p0              = sincos_r_exp,sincos_exp_limit
(p10) br.cond.spnt  _SINCOS_LARGE_ARGS // Go to "large args" routine
};;

// sincos_W          = x * sincos_Inv_Pi_by_16
// Multiply x by scaled 16/pi and add large const to shift integer part of W to
//   rightmost bits of significand
{ .mfi
      ldfpd         sincos_P3,sincos_Q3 = [sincos_AD_1],16
      fma.s1 sincos_W_2TO61_RSH = sincos_NORM_f8,sincos_SIG_INV_PI_BY_16_2TO61,sincos_RSHF_2TO61
      nop.i         999
};;

// get N = (int)sincos_int_Nfloat
// sincos_NFLOAT = Round_Int_Nearest(sincos_W)
// This is done by scaling back by 2^-61 and subtracting the shift constant
{ .mmf
      getf.sig      sincos_GR_n         = sincos_W_2TO61_RSH
      ldfpd         sincos_P2,sincos_Q2 = [sincos_AD_1],16
      fms.s1 sincos_NFLOAT = sincos_W_2TO61_RSH,sincos_2TOM61,sincos_RSHF
};;

// sincos_r          = -sincos_Nfloat * sincos_Pi_by_16_1 + x
{ .mfi
      ldfpd         sincos_P1,sincos_Q1 = [sincos_AD_1],16
      fnma.s1 sincos_r = sincos_NFLOAT, sincos_Pi_by_16_1, sincos_NORM_f8
      nop.i         999
};;

// Add 2^(k-1) (which is in sincos_r_sincos) to N
{ .mmi
      add           sincos_GR_n         = sincos_GR_n, sincos_r_sincos
;;
// Get M (least k+1 bits of N)
      and           sincos_GR_m         = 0x1f,sincos_GR_n
      nop.i         999
};;

// sincos_r          = sincos_r -sincos_Nfloat * sincos_Pi_by_16_2
{ .mfi
      nop.m         999
      fnma.s1 sincos_r = sincos_NFLOAT, sincos_Pi_by_16_2,  sincos_r
      shl           sincos_GR_32m       = sincos_GR_m,5
};;

// Add 32*M to address of sin_cos_beta table
// For sin denorm. - set uflow
{ .mfi
      add           sincos_AD_2         = sincos_GR_32m, sincos_AD_1
(p8)  fclass.m.unc  p10,p0              = f8,0x0b
      nop.i         999
};;

// Load Sin and Cos table value using obtained index m  (sincosf_AD_2)
{ .mfi
      ldfe          sincos_Sm           = [sincos_AD_2],16
      nop.f         999
      nop.i         999
};;

// get rsq = r*r
{ .mfi
      ldfe          sincos_Cm           = [sincos_AD_2]
      fma.s1        sincos_rsq          = sincos_r, sincos_r,   f0 // r^2 = r*r
      nop.i         999
}
{ .mfi
      nop.m         999
      fmpy.s0       fp_tmp              = fp_tmp,fp_tmp // forces inexact flag
      nop.i         999
};;

// sincos_r_exact = sincos_r -sincos_Nfloat * sincos_Pi_by_16_3
{ .mfi
      nop.m         999
      fnma.s1 sincos_r_exact = sincos_NFLOAT, sincos_Pi_by_16_3, sincos_r
      nop.i         999
};;

// Polynomials calculation
// P_1 = P4*r^2 + P3
// Q_2 = Q4*r^2 + Q3
{ .mfi
      nop.m         999
      fma.s1        sincos_P_temp1      = sincos_rsq, sincos_P4, sincos_P3
      nop.i         999
}
{ .mfi
      nop.m         999
      fma.s1        sincos_Q_temp1      = sincos_rsq, sincos_Q4, sincos_Q3
      nop.i         999
};;

// get rcube = r^3 and S[m]*r^2
{ .mfi
      nop.m         999
      fmpy.s1       sincos_srsq         = sincos_Sm,sincos_rsq
      nop.i         999
}
{ .mfi
      nop.m         999
      fmpy.s1       sincos_rcub         = sincos_r_exact, sincos_rsq
      nop.i         999
};;

// Polynomials calculation
// Q_2 = Q_1*r^2 + Q2
// P_1 = P_1*r^2 + P2
{ .mfi
      nop.m         999
      fma.s1        sincos_Q_temp2      = sincos_rsq, sincos_Q_temp1, sincos_Q2
      nop.i         999
}
{ .mfi
      nop.m         999
      fma.s1        sincos_P_temp2      = sincos_rsq, sincos_P_temp1, sincos_P2
      nop.i         999
};;

// Polynomials calculation
// Q = Q_2*r^2 + Q1
// P = P_2*r^2 + P1
{ .mfi
      nop.m         999
      fma.s1        sincos_Q            = sincos_rsq, sincos_Q_temp2, sincos_Q1
      nop.i         999
}
{ .mfi
      nop.m         999
      fma.s1        sincos_P            = sincos_rsq, sincos_P_temp2, sincos_P1
      nop.i         999
};;

// Get final P and Q
// Q = Q*S[m]*r^2 + S[m]
// P = P*r^3 + r
{ .mfi
      nop.m         999
      fma.s1        sincos_Q            = sincos_srsq,sincos_Q, sincos_Sm
      nop.i         999
}
{ .mfi
      nop.m         999
      fma.s1        sincos_P            = sincos_rcub,sincos_P, sincos_r_exact
      nop.i         999
};;

// If sin(denormal), force underflow to be set
{ .mfi
      nop.m         999
(p10) fmpy.d.s0     fp_tmp              = sincos_NORM_f8,sincos_NORM_f8
      nop.i         999
};;

// Final calculation
// result = C[m]*P + Q
{ .mfb
      nop.m         999
      fma.d.s0      f8                  = sincos_Cm, sincos_P, sincos_Q
      br.ret.sptk   b0  // Exit for common path
};;

////////// x = 0/Inf/NaN path //////////////////
_SINCOS_SPECIAL_ARGS:
.pred.rel "mutex",p8,p9
// sin(+/-0) = +/-0
// sin(Inf)  = NaN
// sin(NaN)  = NaN
{ .mfi
      nop.m         999
(p8)  fma.d.s0      f8                  = f8, f0, f0 // sin(+/-0,NaN,Inf)
      nop.i         999
}
// cos(+/-0) = 1.0
// cos(Inf)  = NaN
// cos(NaN)  = NaN
{ .mfb
      nop.m         999
(p9)  fma.d.s0      f8                  = f8, f0, f1 // cos(+/-0,NaN,Inf)
      br.ret.sptk   b0 // Exit for x = 0/Inf/NaN path
};;

_SINCOS_UNORM:
// Here if x=unorm
{ .mfb
      getf.exp      sincos_r_signexp    = sincos_NORM_f8 // Get signexp of x
      fcmp.eq.s0    p11,p0              = f8, f0  // Dummy op to set denorm flag
      br.cond.sptk  _SINCOS_COMMON2     // Return to main path
};;

GLOBAL_IEEE754_END(cos)
libm_alias_double_other (__cos, cos)

//////////// x >= 2^27 - large arguments routine call ////////////
LOCAL_LIBM_ENTRY(__libm_callout_sincos)
_SINCOS_LARGE_ARGS:
.prologue
{ .mfi
      mov           GR_SAVE_r_sincos    = sincos_r_sincos // Save sin or cos
      nop.f         999
.save ar.pfs,GR_SAVE_PFS
      mov           GR_SAVE_PFS         = ar.pfs
}
;;

{ .mfi
      mov           GR_SAVE_GP          = gp
      nop.f         999
.save b0, GR_SAVE_B0
      mov           GR_SAVE_B0          = b0
}

.body
{ .mbb
      setf.sig      sincos_save_tmp     = sincos_GR_all_ones// inexact set
      nop.b         999
(p8)  br.call.sptk.many b0              = __libm_sin_large# // sin(large_X)

};;

{ .mbb
      cmp.ne        p9,p0               = GR_SAVE_r_sincos, r0 // set p9 if cos
      nop.b         999
(p9)  br.call.sptk.many b0              = __libm_cos_large# // cos(large_X)
};;

{ .mfi
      mov           gp                  = GR_SAVE_GP
      fma.d.s0      f8                  = f8, f1, f0 // Round result to double
      mov           b0                  = GR_SAVE_B0
}
// Force inexact set
{ .mfi
      nop.m         999
      fmpy.s0       sincos_save_tmp     = sincos_save_tmp, sincos_save_tmp
      nop.i         999
};;

{ .mib
      nop.m         999
      mov           ar.pfs              = GR_SAVE_PFS
      br.ret.sptk   b0 // Exit for large arguments routine call
};;

LOCAL_LIBM_END(__libm_callout_sincos)

.type    __libm_sin_large#,@function
.global  __libm_sin_large#
.type    __libm_cos_large#,@function
.global  __libm_cos_large#
