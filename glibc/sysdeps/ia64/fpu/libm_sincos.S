.file "libm_sincos.s"


// Copyright (c) 2002 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2002 by the Intel Numerics Group, Intel Corporation
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
// 02/01/02 Initial version
// 02/18/02 Large arguments processing routine is excluded.
//          External interface entry points are added
// 03/13/02 Corrected restore of predicate registers
// 03/19/02 Added stack unwind around call to __libm_cis_large
// 09/05/02 Work range is widened by reduction strengthen (3 parts of Pi/16)
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 08/08/03 Improved performance
// 02/11/04 cis is moved to the separate file.
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// 1) void sincos(double, double*s, double*c)
// 2) __libm_sincos - internal LIBM function, that accepts
//    argument in f8 and returns cosine through f8, sine through f9
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
//    Answer = S[m] Cos(r) + C[m] P
//
//       Cos(r) = 1 + rsq Q
//       Cos(r) = 1 + r^2 Q
//       Cos(r) = 1 + r^2 (q1 + r^2q2 + r^4q3 + r^6q4)
//       Cos(r) = 1 + r^2q1 + r^4q2 + r^6q3 + r^8q4 + ...
//
//       S[m] Cos(r) = S[m](1 + rsq Q)
//       S[m] Cos(r) = S[m] + S[m] rsq Q
//       S[m] Cos(r) = S[m] + s_rsq Q
//       Q           = S[m] + s_rsq Q
//
// Then,
//
//    Answer = Q + C[m] P

// Registers used
//==============================================================
// general input registers:
// r14 -> r39

// predicate registers used:
// p6 -> p14
//
// floating-point registers used
// f9 -> f15
// f32 -> f67

// Assembly macros
//==============================================================

cis_Arg                     = f8

cis_Sin_res                 = f9
cis_Cos_res                 = f8

cis_NORM_f8                 = f10
cis_W                       = f11
cis_int_Nfloat              = f12
cis_Nfloat                  = f13

cis_r                       = f14
cis_rsq                     = f15
cis_rcub                    = f32

cis_Inv_Pi_by_16            = f33
cis_Pi_by_16_hi             = f34
cis_Pi_by_16_lo             = f35

cis_Inv_Pi_by_64            = f36
cis_Pi_by_16_lowest         = f37
cis_r_exact                 = f38


cis_P1                      = f39
cis_Q1                      = f40
cis_P2                      = f41
cis_Q2                      = f42
cis_P3                      = f43
cis_Q3                      = f44
cis_P4                      = f45
cis_Q4                      = f46

cis_P_temp1                 = f47
cis_P_temp2                 = f48

cis_Q_temp1                 = f49
cis_Q_temp2                 = f50

cis_P                       = f51

cis_SIG_INV_PI_BY_16_2TO61  = f52
cis_RSHF_2TO61              = f53
cis_RSHF                    = f54
cis_2TOM61                  = f55
cis_NFLOAT                  = f56
cis_W_2TO61_RSH             = f57

cis_tmp                     = f58

cis_Sm_sin                  = f59
cis_Cm_sin                  = f60

cis_Sm_cos                  = f61
cis_Cm_cos                  = f62

cis_srsq_sin                = f63
cis_srsq_cos                = f64

cis_Q_sin                   = f65
cis_Q_cos                   = f66
cis_Q                       = f67

/////////////////////////////////////////////////////////////

cis_pResSin                 = r33
cis_pResCos                 = r34

cis_GR_sig_inv_pi_by_16     = r14
cis_GR_rshf_2to61           = r15
cis_GR_rshf                 = r16
cis_GR_exp_2tom61           = r17
cis_GR_n                    = r18
cis_GR_n_sin                = r19
cis_exp_limit               = r20
cis_r_signexp               = r21
cis_AD_1                    = r22
cis_r_sincos                = r23
cis_r_exp                   = r24
cis_r_17_ones               = r25
cis_GR_m_sin                = r26
cis_GR_32m_sin              = r26
cis_GR_n_cos                = r27
cis_GR_m_cos                = r28
cis_GR_32m_cos              = r28
cis_AD_2_sin                = r29
cis_AD_2_cos                = r30
cis_gr_tmp                  = r31

GR_SAVE_B0                  = r35
GR_SAVE_GP                  = r36
rB0_SAVED                   = r37
GR_SAVE_PFS                 = r38
GR_SAVE_PR                  = r39

RODATA

.align 16
// Pi/16 parts
LOCAL_OBJECT_START(double_cis_pi)
   data8 0xC90FDAA22168C234, 0x00003FFC // pi/16 1st part
   data8 0xC4C6628B80DC1CD1, 0x00003FBC // pi/16 2nd part
   data8 0xA4093822299F31D0, 0x00003F7A // pi/16 3rd part
LOCAL_OBJECT_END(double_cis_pi)

// Coefficients for polynomials
LOCAL_OBJECT_START(double_cis_pq_k4)
   data8 0x3EC71C963717C63A // P4
   data8 0x3EF9FFBA8F191AE6 // Q4
   data8 0xBF2A01A00F4E11A8 // P3
   data8 0xBF56C16C05AC77BF // Q3
   data8 0x3F8111111110F167 // P2
   data8 0x3FA555555554DD45 // Q2
   data8 0xBFC5555555555555 // P1
   data8 0xBFDFFFFFFFFFFFFC // Q1
LOCAL_OBJECT_END(double_cis_pq_k4)

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

GLOBAL_IEEE754_ENTRY(sincos)
// cis_GR_sig_inv_pi_by_16 = significand of 16/pi
{ .mlx
      getf.exp      cis_r_signexp       = cis_Arg
      movl          cis_GR_sig_inv_pi_by_16 = 0xA2F9836E4E44152A

}
// cis_GR_rshf_2to61 = 1.1000 2^(63+63-2)
{ .mlx
      addl          cis_AD_1                = @ltoff(double_cis_pi), gp
      movl          cis_GR_rshf_2to61       = 0x47b8000000000000
};;

{ .mfi
      ld8           cis_AD_1            = [cis_AD_1]
      fnorm.s1      cis_NORM_f8         = cis_Arg
      cmp.eq        p13, p14            = r0, r0 // p13 set for sincos
}
// cis_GR_exp_2tom61 = exponent of scaling factor 2^-61
{ .mib
      mov           cis_GR_exp_2tom61   = 0xffff-61
      nop.i         0
      br.cond.sptk  _CIS_COMMON
};;
GLOBAL_IEEE754_END(sincos)
libm_alias_double_other (__sincos, sincos)

GLOBAL_LIBM_ENTRY(__libm_sincos)
// cis_GR_sig_inv_pi_by_16 = significand of 16/pi
{ .mlx
      getf.exp      cis_r_signexp       = cis_Arg
      movl          cis_GR_sig_inv_pi_by_16 = 0xA2F9836E4E44152A
}
// cis_GR_rshf_2to61 = 1.1000 2^(63+63-2)
{ .mlx
      addl          cis_AD_1            = @ltoff(double_cis_pi), gp
      movl          cis_GR_rshf_2to61   = 0x47b8000000000000
};;

// p14 set for __libm_sincos and cis
{ .mfi
      ld8           cis_AD_1            = [cis_AD_1]
      fnorm.s1      cis_NORM_f8         = cis_Arg
      cmp.eq        p14, p13            = r0, r0
}
// cis_GR_exp_2tom61 = exponent of scaling factor 2^-61
{ .mib
      mov           cis_GR_exp_2tom61   = 0xffff-61
      nop.i         0
      nop.b         0
};;

_CIS_COMMON:
//  Form two constants we need
//  16/pi * 2^-2 * 2^63, scaled by 2^61 since we just loaded the significand
//  1.1000...000 * 2^(63+63-2) to right shift int(W) into the low significand
//  fcmp used to set denormal, and invalid on snans
{ .mfi
      setf.sig      cis_SIG_INV_PI_BY_16_2TO61 = cis_GR_sig_inv_pi_by_16
      fclass.m      p6,p0                      = cis_Arg, 0xe7 // if x=0,inf,nan
      addl          cis_gr_tmp                 = -1, r0
}
// 1.1000 2^63 for right shift
{ .mlx
      setf.d        cis_RSHF_2TO61             = cis_GR_rshf_2to61
      movl          cis_GR_rshf                = 0x43e8000000000000
};;

//  Form another constant
//  2^-61 for scaling Nfloat
//  0x1001a is register_bias + 27.
//  So if f8 >= 2^27, go to large arguments routine
{ .mfi
      alloc         GR_SAVE_PFS         = ar.pfs, 3, 5, 0, 0
      fclass.m      p11,p0              = cis_Arg, 0x0b // Test for x=unorm
      mov           cis_exp_limit       = 0x1001a
}
{ .mib
      setf.exp      cis_2TOM61          = cis_GR_exp_2tom61
      nop.i         0
(p6)  br.cond.spnt  _CIS_SPECIAL_ARGS
};;

//  Load the two pieces of pi/16
//  Form another constant
//  1.1000...000 * 2^63, the right shift constant
{ .mmb
      ldfe          cis_Pi_by_16_hi     = [cis_AD_1],16
      setf.d        cis_RSHF            = cis_GR_rshf
(p11) br.cond.spnt  _CIS_UNORM          // Branch if x=unorm
};;

_CIS_COMMON2:
// Return here if x=unorm
// Create constant inexact set
{ .mmi
      ldfe          cis_Pi_by_16_lo     = [cis_AD_1],16
      setf.sig      cis_tmp             = cis_gr_tmp
      nop.i         0
};;

// Select exponent (17 lsb)
{ .mfi
      ldfe          cis_Pi_by_16_lowest = [cis_AD_1],16
      nop.f         0
      dep.z         cis_r_exp           = cis_r_signexp, 0, 17
};;

// Start loading P, Q coefficients
// p10 is true if we must call routines to handle larger arguments
// p10 is true if f8 exp is > 0x1001a
{ .mmb
      ldfpd         cis_P4,cis_Q4       = [cis_AD_1],16
      cmp.ge        p10, p0             = cis_r_exp, cis_exp_limit
(p10) br.cond.spnt  _CIS_LARGE_ARGS // go to |x| >= 2^27 path
};;

// cis_W = x * cis_Inv_Pi_by_16
// Multiply x by scaled 16/pi and add large const to shift integer part of W to
// rightmost bits of significand
{ .mfi
      ldfpd         cis_P3,cis_Q3       = [cis_AD_1],16
      fma.s1 cis_W_2TO61_RSH = cis_NORM_f8,cis_SIG_INV_PI_BY_16_2TO61,cis_RSHF_2TO61
      nop.i  0
};;

// get N = (int)cis_int_Nfloat
// cis_NFLOAT = Round_Int_Nearest(cis_W)
{ .mmf
      getf.sig      cis_GR_n            = cis_W_2TO61_RSH
      ldfpd  cis_P2,cis_Q2   = [cis_AD_1],16
      fms.s1        cis_NFLOAT          = cis_W_2TO61_RSH,cis_2TOM61,cis_RSHF
};;

// cis_r = -cis_Nfloat * cis_Pi_by_16_hi + x
{ .mfi
      ldfpd         cis_P1,cis_Q1       = [cis_AD_1], 16
      fnma.s1       cis_r               = cis_NFLOAT,cis_Pi_by_16_hi,cis_NORM_f8
      nop.i         0
};;

// Add 2^(k-1) (which is in cis_r_sincos) to N
{ .mmi
      add           cis_GR_n_cos        = 0x8, cis_GR_n
;;
//Get M (least k+1 bits of N)
      and           cis_GR_m_sin        = 0x1f,cis_GR_n
      and           cis_GR_m_cos        = 0x1f,cis_GR_n_cos
};;

{ .mmi
      nop.m         0
      nop.m         0
      shl           cis_GR_32m_sin      = cis_GR_m_sin,5
};;

// Add 32*M to address of sin_cos_beta table
// cis_r =  cis_r -cis_Nfloat * cis_Pi_by_16_lo
{ .mfi
      add           cis_AD_2_sin        = cis_GR_32m_sin, cis_AD_1
      fnma.s1       cis_r               = cis_NFLOAT, cis_Pi_by_16_lo,  cis_r
      shl           cis_GR_32m_cos      = cis_GR_m_cos,5
};;

// Add 32*M to address of sin_cos_beta table
{ .mmf
      ldfe          cis_Sm_sin          = [cis_AD_2_sin],16
      add           cis_AD_2_cos        = cis_GR_32m_cos, cis_AD_1
      fclass.m.unc  p10,p0              = cis_Arg,0x0b  // den. input - uflow
};;

{ .mfi
      ldfe          cis_Sm_cos          = [cis_AD_2_cos], 16
      nop.i         0
};;

{ .mfi
      ldfe          cis_Cm_sin          = [cis_AD_2_sin]
      fma.s1        cis_rsq             = cis_r, cis_r,   f0  // get r^2
      nop.i         0
}
// fmpy forces inexact flag
{ .mfi
      nop.m         0
      fmpy.s0       cis_tmp             = cis_tmp,cis_tmp
      nop.i         0
};;

{ .mfi
      nop.m         0
      fnma.s1       cis_r_exact         = cis_NFLOAT, cis_Pi_by_16_lowest, cis_r
      nop.i         0
};;

{ .mfi
      ldfe          cis_Cm_cos          = [cis_AD_2_cos]
      fma.s1        cis_P_temp1         = cis_rsq, cis_P4, cis_P3
      nop.i         0
}

{ .mfi
      nop.m         0
      fma.s1        cis_Q_temp1         = cis_rsq, cis_Q4, cis_Q3
      nop.i         0
};;

{ .mfi
      nop.m         0
      fmpy.s1       cis_srsq_sin        = cis_Sm_sin, cis_rsq
      nop.i         0
}
{ .mfi
      nop.m         0
      fmpy.s1       cis_srsq_cos        = cis_Sm_cos,cis_rsq
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.s1        cis_Q_temp2         = cis_rsq, cis_Q_temp1, cis_Q2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        cis_P_temp2         = cis_rsq, cis_P_temp1, cis_P2
      nop.i         0
};;

{ .mfi
      nop.m         0
      fmpy.s1       cis_rcub            = cis_r_exact, cis_rsq // get r^3
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.s1        cis_Q               = cis_rsq, cis_Q_temp2, cis_Q1
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        cis_P               = cis_rsq, cis_P_temp2, cis_P1
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.s1        cis_Q_sin           = cis_srsq_sin,cis_Q, cis_Sm_sin
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        cis_Q_cos           = cis_srsq_cos,cis_Q, cis_Sm_cos
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.s1        cis_P               = cis_rcub,cis_P, cis_r_exact // final P
      nop.i         0
};;

// If den. arg, force underflow to be set
{ .mfi
      nop.m         0
(p10) fmpy.d.s0     cis_tmp             = cis_Arg,cis_Arg
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.d.s0      cis_Sin_res         = cis_Cm_sin,cis_P,cis_Q_sin//Final sin
      nop.i         0
}
{ .mfb
      nop.m         0
      fma.d.s0      cis_Cos_res         = cis_Cm_cos,cis_P,cis_Q_cos//Final cos
(p14) br.ret.sptk   b0  // common exit for __libm_sincos and cis main path
};;

{ .mmb
      stfd          [cis_pResSin]       = cis_Sin_res
      stfd          [cis_pResCos]       = cis_Cos_res
      br.ret.sptk   b0 // common exit for sincos main path
};;

_CIS_SPECIAL_ARGS:
// sin(+/-0) = +/-0
// sin(Inf)  = NaN
// sin(NaN)  = NaN
{ .mfi
      nop.m         999
      fma.d.s0      cis_Sin_res          = cis_Arg, f0, f0 // sinf(+/-0,NaN,Inf)
      nop.i         999
};;
// cos(+/-0) = 1.0
// cos(Inf)  = NaN
// cos(NaN)  = NaN
{ .mfb
      nop.m         999
      fma.d.s0      cis_Cos_res          = cis_Arg, f0, f1 // cosf(+/-0,NaN,Inf)
(p14) br.ret.sptk   b0 //spec exit for __libm_sincos and cis main path
};;

{ .mmb
      stfd          [cis_pResSin]       = cis_Sin_res
      stfd          [cis_pResCos]       = cis_Cos_res
      br.ret.sptk   b0 // common exit for sincos main path
};;

_CIS_UNORM:
// Here if x=unorm
{ .mfb
      getf.exp      cis_r_signexp       = cis_NORM_f8 // Get signexp of x
      fcmp.eq.s0    p11,p0              = cis_Arg, f0 // Dummy op to set denorm
      br.cond.sptk  _CIS_COMMON2        // Return to main path
};;

GLOBAL_LIBM_END(__libm_sincos)

////  |x| > 2^27 path  ///////
.proc _CIS_LARGE_ARGS
_CIS_LARGE_ARGS:
.prologue
{ .mfi
      nop.m         0
      nop.f         0
.save ar.pfs, GR_SAVE_PFS
      mov           GR_SAVE_PFS         = ar.pfs
}
;;

{ .mfi
      mov           GR_SAVE_GP          = gp
      nop.f         0
.save b0, GR_SAVE_B0
      mov           GR_SAVE_B0          = b0
};;

.body
// Call of huge arguments sincos
{ .mib
      nop.m         0
      mov           GR_SAVE_PR          = pr
      br.call.sptk  b0                  = __libm_sincos_large
};;

{ .mfi
      mov           gp                  = GR_SAVE_GP
      nop.f         0
      mov           pr                  = GR_SAVE_PR, 0x1fffe
}
;;

{ .mfi
      nop.m         0
      nop.f         0
      mov           b0                  = GR_SAVE_B0
}
;;

{ .mfi
      nop.m         0
      fma.d.s0      cis_Cos_res         = cis_Cos_res, f1, f0
      mov           ar.pfs              = GR_SAVE_PFS
}
{ .mfb
      nop.m         0
      fma.d.s0      cis_Sin_res         = cis_Sin_res, f1, f0
(p14) br.ret.sptk   b0  // exit for |x| > 2^27 path (__libm_sincos and cis)
};;

{ .mmb
      stfd          [cis_pResSin]       = cis_Sin_res
      stfd          [cis_pResCos]       = cis_Cos_res
      br.ret.sptk   b0 // exit for sincos |x| > 2^27 path
};;
.endp _CIS_LARGE_ARGS

.type __libm_sincos_large#,@function
.global __libm_sincos_large#
