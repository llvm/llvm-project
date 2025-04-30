.file "sincosf.s"


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
// 09/05/02 Single precision version is made using double precision one as base
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// float sinf( float x);
// float cosf( float x);
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
//    Do this as (x -  nfloat * HIGH(pi/2^k)) - nfloat * LOW(pi/2^k)

//    for increased accuracy.
//    pi/2^k is stored as two numbers that when added make pi/2^k.
//       pi/2^k = HIGH(pi/2^k) + LOW(pi/2^k)
//    HIGH part is rounded to zero, LOW - to nearest
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
// Cos(r) = 1 + r^2 q1  + r^4 q2  = Series for Cos
// Sin(r) = r + r^3 p1  + r^5 p2  = Series for Sin
//
// and the coefficients q1, q2 and p1, p2 are stored in a table
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
//    P = P1 + r^2*P2
//    Q = Q1 + r^2*Q2
//
//       rcub = r * rsq
//       Sin(r) = r + rcub * P
//              = r + r^3p1  + r^5p2 = Sin(r)
//
//            The coefficients are not exactly these values, but almost.
//
//            p1 = -1/6  = -1/3!
//            p2 = 1/120 =  1/5!
//            p3 = -1/5040 = -1/7!
//            p4 = 1/362889 = 1/9!
//
//       P =  r + r^3 * P
//
//    Answer = S[m] Cos(r) + C[m] P
//
//       Cos(r) = 1 + rsq Q
//       Cos(r) = 1 + r^2 Q
//       Cos(r) = 1 + r^2 (q1 + r^2q2)
//       Cos(r) = 1 + r^2q1 + r^4q2
//
//       S[m] Cos(r) = S[m](1 + rsq Q)
//       S[m] Cos(r) = S[m] + S[m] rsq Q
//       S[m] Cos(r) = S[m] + s_rsq Q
//       Q         = S[m] + s_rsq Q
//
// Then,
//
//    Answer = Q + C[m] P


// Registers used
//==============================================================
// general input registers:
// r14 -> r19
// r32 -> r45

// predicate registers used:
// p6 -> p14

// floating-point registers used
// f9 -> f15
// f32 -> f61

// Assembly macros
//==============================================================
sincosf_NORM_f8                 = f9
sincosf_W                       = f10
sincosf_int_Nfloat              = f11
sincosf_Nfloat                  = f12

sincosf_r                       = f13
sincosf_rsq                     = f14
sincosf_rcub                    = f15
sincosf_save_tmp                = f15

sincosf_Inv_Pi_by_16            = f32
sincosf_Pi_by_16_1              = f33
sincosf_Pi_by_16_2              = f34

sincosf_Inv_Pi_by_64            = f35

sincosf_Pi_by_16_3              = f36

sincosf_r_exact                 = f37

sincosf_Sm                      = f38
sincosf_Cm                      = f39

sincosf_P1                      = f40
sincosf_Q1                      = f41
sincosf_P2                      = f42
sincosf_Q2                      = f43
sincosf_P3                      = f44
sincosf_Q3                      = f45
sincosf_P4                      = f46
sincosf_Q4                      = f47

sincosf_P_temp1                 = f48
sincosf_P_temp2                 = f49

sincosf_Q_temp1                 = f50
sincosf_Q_temp2                 = f51

sincosf_P                       = f52
sincosf_Q                       = f53

sincosf_srsq                    = f54

sincosf_SIG_INV_PI_BY_16_2TO61  = f55
sincosf_RSHF_2TO61              = f56
sincosf_RSHF                    = f57
sincosf_2TOM61                  = f58
sincosf_NFLOAT                  = f59
sincosf_W_2TO61_RSH             = f60

fp_tmp                          = f61

/////////////////////////////////////////////////////////////

sincosf_AD_1                    = r33
sincosf_AD_2                    = r34
sincosf_exp_limit               = r35
sincosf_r_signexp               = r36
sincosf_AD_beta_table           = r37
sincosf_r_sincos                = r38

sincosf_r_exp                   = r39
sincosf_r_17_ones               = r40

sincosf_GR_sig_inv_pi_by_16     = r14
sincosf_GR_rshf_2to61           = r15
sincosf_GR_rshf                 = r16
sincosf_GR_exp_2tom61           = r17
sincosf_GR_n                    = r18
sincosf_GR_m                    = r19
sincosf_GR_32m                  = r19
sincosf_GR_all_ones             = r19

gr_tmp                          = r41
GR_SAVE_PFS                     = r41
GR_SAVE_B0                      = r42
GR_SAVE_GP                      = r43

RODATA
.align 16

// Pi/16 parts
LOCAL_OBJECT_START(double_sincosf_pi)
   data8 0xC90FDAA22168C234, 0x00003FFC // pi/16 1st part
   data8 0xC4C6628B80DC1CD1, 0x00003FBC // pi/16 2nd part
LOCAL_OBJECT_END(double_sincosf_pi)

// Coefficients for polynomials
LOCAL_OBJECT_START(double_sincosf_pq_k4)
   data8 0x3F810FABB668E9A2 // P2
   data8 0x3FA552E3D6DE75C9 // Q2
   data8 0xBFC555554447BC7F // P1
   data8 0xBFDFFFFFC447610A // Q1
LOCAL_OBJECT_END(double_sincosf_pq_k4)

// Sincos table (S[m], C[m])
LOCAL_OBJECT_START(double_sin_cos_beta_k4)
    data8 0x0000000000000000 // sin ( 0 Pi / 16 )
    data8 0x3FF0000000000000 // cos ( 0 Pi / 16 )
//
    data8 0x3FC8F8B83C69A60B // sin ( 1 Pi / 16 )
    data8 0x3FEF6297CFF75CB0 // cos ( 1 Pi / 16 )
//
    data8 0x3FD87DE2A6AEA963 // sin ( 2 Pi / 16 )
    data8 0x3FED906BCF328D46 // cos ( 2 Pi / 16 )
//
    data8 0x3FE1C73B39AE68C8 // sin ( 3 Pi / 16 )
    data8 0x3FEA9B66290EA1A3 // cos ( 3 Pi / 16 )
//
    data8 0x3FE6A09E667F3BCD // sin ( 4 Pi / 16 )
    data8 0x3FE6A09E667F3BCD // cos ( 4 Pi / 16 )
//
    data8 0x3FEA9B66290EA1A3 // sin ( 5 Pi / 16 )
    data8 0x3FE1C73B39AE68C8 // cos ( 5 Pi / 16 )
//
    data8 0x3FED906BCF328D46 // sin ( 6 Pi / 16 )
    data8 0x3FD87DE2A6AEA963 // cos ( 6 Pi / 16 )
//
    data8 0x3FEF6297CFF75CB0 // sin ( 7 Pi / 16 )
    data8 0x3FC8F8B83C69A60B // cos ( 7 Pi / 16 )
//
    data8 0x3FF0000000000000 // sin ( 8 Pi / 16 )
    data8 0x0000000000000000 // cos ( 8 Pi / 16 )
//
    data8 0x3FEF6297CFF75CB0 // sin ( 9 Pi / 16 )
    data8 0xBFC8F8B83C69A60B // cos ( 9 Pi / 16 )
//
    data8 0x3FED906BCF328D46 // sin ( 10 Pi / 16 )
    data8 0xBFD87DE2A6AEA963 // cos ( 10 Pi / 16 )
//
    data8 0x3FEA9B66290EA1A3 // sin ( 11 Pi / 16 )
    data8 0xBFE1C73B39AE68C8 // cos ( 11 Pi / 16 )
//
    data8 0x3FE6A09E667F3BCD // sin ( 12 Pi / 16 )
    data8 0xBFE6A09E667F3BCD // cos ( 12 Pi / 16 )
//
    data8 0x3FE1C73B39AE68C8 // sin ( 13 Pi / 16 )
    data8 0xBFEA9B66290EA1A3 // cos ( 13 Pi / 16 )
//
    data8 0x3FD87DE2A6AEA963 // sin ( 14 Pi / 16 )
    data8 0xBFED906BCF328D46 // cos ( 14 Pi / 16 )
//
    data8 0x3FC8F8B83C69A60B // sin ( 15 Pi / 16 )
    data8 0xBFEF6297CFF75CB0 // cos ( 15 Pi / 16 )
//
    data8 0x0000000000000000 // sin ( 16 Pi / 16 )
    data8 0xBFF0000000000000 // cos ( 16 Pi / 16 )
//
    data8 0xBFC8F8B83C69A60B // sin ( 17 Pi / 16 )
    data8 0xBFEF6297CFF75CB0 // cos ( 17 Pi / 16 )
//
    data8 0xBFD87DE2A6AEA963 // sin ( 18 Pi / 16 )
    data8 0xBFED906BCF328D46 // cos ( 18 Pi / 16 )
//
    data8 0xBFE1C73B39AE68C8 // sin ( 19 Pi / 16 )
    data8 0xBFEA9B66290EA1A3 // cos ( 19 Pi / 16 )
//
    data8 0xBFE6A09E667F3BCD // sin ( 20 Pi / 16 )
    data8 0xBFE6A09E667F3BCD // cos ( 20 Pi / 16 )
//
    data8 0xBFEA9B66290EA1A3 // sin ( 21 Pi / 16 )
    data8 0xBFE1C73B39AE68C8 // cos ( 21 Pi / 16 )
//
    data8 0xBFED906BCF328D46 // sin ( 22 Pi / 16 )
    data8 0xBFD87DE2A6AEA963 // cos ( 22 Pi / 16 )
//
    data8 0xBFEF6297CFF75CB0 // sin ( 23 Pi / 16 )
    data8 0xBFC8F8B83C69A60B // cos ( 23 Pi / 16 )
//
    data8 0xBFF0000000000000 // sin ( 24 Pi / 16 )
    data8 0x0000000000000000 // cos ( 24 Pi / 16 )
//
    data8 0xBFEF6297CFF75CB0 // sin ( 25 Pi / 16 )
    data8 0x3FC8F8B83C69A60B // cos ( 25 Pi / 16 )
//
    data8 0xBFED906BCF328D46 // sin ( 26 Pi / 16 )
    data8 0x3FD87DE2A6AEA963 // cos ( 26 Pi / 16 )
//
    data8 0xBFEA9B66290EA1A3 // sin ( 27 Pi / 16 )
    data8 0x3FE1C73B39AE68C8 // cos ( 27 Pi / 16 )
//
    data8 0xBFE6A09E667F3BCD // sin ( 28 Pi / 16 )
    data8 0x3FE6A09E667F3BCD // cos ( 28 Pi / 16 )
//
    data8 0xBFE1C73B39AE68C8 // sin ( 29 Pi / 16 )
    data8 0x3FEA9B66290EA1A3 // cos ( 29 Pi / 16 )
//
    data8 0xBFD87DE2A6AEA963 // sin ( 30 Pi / 16 )
    data8 0x3FED906BCF328D46 // cos ( 30 Pi / 16 )
//
    data8 0xBFC8F8B83C69A60B // sin ( 31 Pi / 16 )
    data8 0x3FEF6297CFF75CB0 // cos ( 31 Pi / 16 )
//
    data8 0x0000000000000000 // sin ( 32 Pi / 16 )
    data8 0x3FF0000000000000 // cos ( 32 Pi / 16 )
LOCAL_OBJECT_END(double_sin_cos_beta_k4)

.section .text

////////////////////////////////////////////////////////
// There are two entry points: sin and cos
// If from sin, p8 is true
// If from cos, p9 is true

GLOBAL_IEEE754_ENTRY(sinf)

{ .mlx
      alloc         r32                 = ar.pfs,1,13,0,0
      movl  sincosf_GR_sig_inv_pi_by_16 = 0xA2F9836E4E44152A //signd of 16/pi
}
{ .mlx
      addl         sincosf_AD_1         = @ltoff(double_sincosf_pi), gp
      movl  sincosf_GR_rshf_2to61       = 0x47b8000000000000 // 1.1 2^(63+63-2)
};;

{ .mfi
      ld8           sincosf_AD_1        = [sincosf_AD_1]
      fnorm.s1      sincosf_NORM_f8     = f8     // Normalize argument
      cmp.eq        p8,p9               = r0, r0 // set p8 (clear p9) for sin
}
{ .mib
      mov           sincosf_GR_exp_2tom61 = 0xffff-61 // exponent of scale 2^-61
      mov           sincosf_r_sincos      = 0x0       // 0 for sin
      br.cond.sptk  _SINCOSF_COMMON                 // go to common part
};;

GLOBAL_IEEE754_END(sinf)
libm_alias_float_other (__sin, sin)

GLOBAL_IEEE754_ENTRY(cosf)

{ .mlx
      alloc         r32                 = ar.pfs,1,13,0,0
      movl  sincosf_GR_sig_inv_pi_by_16 = 0xA2F9836E4E44152A //signd of 16/pi
}
{ .mlx
      addl          sincosf_AD_1        = @ltoff(double_sincosf_pi), gp
      movl  sincosf_GR_rshf_2to61       = 0x47b8000000000000 // 1.1 2^(63+63-2)
};;

{ .mfi
      ld8           sincosf_AD_1        = [sincosf_AD_1]
      fnorm.s1      sincosf_NORM_f8     = f8        // Normalize argument
      cmp.eq        p9,p8               = r0, r0    // set p9 (clear p8) for cos
}
{ .mib
      mov           sincosf_GR_exp_2tom61 = 0xffff-61 // exponent of scale 2^-61
      mov           sincosf_r_sincos      = 0x8       // 8 for cos
      nop.b         999
};;

////////////////////////////////////////////////////////
// All entry points end up here.
// If from sin, sincosf_r_sincos is 0 and p8 is true
// If from cos, sincosf_r_sincos is 8 = 2^(k-1) and p9 is true
// We add sincosf_r_sincos to N

///////////// Common sin and cos part //////////////////
_SINCOSF_COMMON:

//  Form two constants we need
//  16/pi * 2^-2 * 2^63, scaled by 2^61 since we just loaded the significand
//  1.1000...000 * 2^(63+63-2) to right shift int(W) into the low significand
//  fcmp used to set denormal, and invalid on snans
{ .mfi
      setf.sig      sincosf_SIG_INV_PI_BY_16_2TO61 = sincosf_GR_sig_inv_pi_by_16
      fclass.m      p6,p0                          = f8, 0xe7 // if x=0,inf,nan
      mov           sincosf_exp_limit              = 0x10017
}
{ .mlx
      setf.d        sincosf_RSHF_2TO61  = sincosf_GR_rshf_2to61
      movl          sincosf_GR_rshf     = 0x43e8000000000000 // 1.1000 2^63
};;                                                          // Right shift

//  Form another constant
//  2^-61 for scaling Nfloat
//  0x10017 is register_bias + 24.
//  So if f8 >= 2^24, go to large argument routines
{ .mmi
      getf.exp      sincosf_r_signexp   = f8
      setf.exp      sincosf_2TOM61      = sincosf_GR_exp_2tom61
      addl          gr_tmp              = -1,r0 // For "inexect" constant create
};;

// Load the two pieces of pi/16
// Form another constant
//  1.1000...000 * 2^63, the right shift constant
{ .mmb
      ldfe          sincosf_Pi_by_16_1  = [sincosf_AD_1],16
      setf.d        sincosf_RSHF        = sincosf_GR_rshf
(p6)  br.cond.spnt  _SINCOSF_SPECIAL_ARGS
};;

// Getting argument's exp for "large arguments" filtering
{ .mmi
      ldfe          sincosf_Pi_by_16_2  = [sincosf_AD_1],16
      setf.sig      fp_tmp              = gr_tmp // constant for inexact set
      nop.i         999
};;

// Polynomial coefficients (Q2, Q1, P2, P1) loading
{ .mmi
      ldfpd         sincosf_P2,sincosf_Q2 = [sincosf_AD_1],16
      nop.m         999
      nop.i         999
};;

// Select exponent (17 lsb)
{ .mmi
      ldfpd         sincosf_P1,sincosf_Q1 = [sincosf_AD_1],16
      nop.m         999
      dep.z         sincosf_r_exp         = sincosf_r_signexp, 0, 17
};;

// p10 is true if we must call routines to handle larger arguments
// p10 is true if f8 exp is >= 0x10017 (2^24)
{ .mfb
      cmp.ge        p10,p0              = sincosf_r_exp,sincosf_exp_limit
      nop.f         999
(p10) br.cond.spnt  _SINCOSF_LARGE_ARGS // Go to "large args" routine
};;

// sincosf_W          = x * sincosf_Inv_Pi_by_16
// Multiply x by scaled 16/pi and add large const to shift integer part of W to
//   rightmost bits of significand
{ .mfi
      nop.m         999
      fma.s1 sincosf_W_2TO61_RSH = sincosf_NORM_f8, sincosf_SIG_INV_PI_BY_16_2TO61, sincosf_RSHF_2TO61
      nop.i         999
};;

// sincosf_NFLOAT = Round_Int_Nearest(sincosf_W)
// This is done by scaling back by 2^-61 and subtracting the shift constant
{ .mfi
      nop.m         999
      fms.s1 sincosf_NFLOAT = sincosf_W_2TO61_RSH,sincosf_2TOM61,sincosf_RSHF
      nop.i         999
};;

// get N = (int)sincosf_int_Nfloat
{ .mfi
      getf.sig      sincosf_GR_n        = sincosf_W_2TO61_RSH // integer N value
      nop.f         999
      nop.i         999
};;

// Add 2^(k-1) (which is in sincosf_r_sincos=8) to N
// sincosf_r          = -sincosf_Nfloat * sincosf_Pi_by_16_1 + x
{ .mfi
      add           sincosf_GR_n        = sincosf_GR_n, sincosf_r_sincos
      fnma.s1 sincosf_r = sincosf_NFLOAT, sincosf_Pi_by_16_1, sincosf_NORM_f8
      nop.i         999
};;

// Get M (least k+1 bits of N)
{ .mmi
      and           sincosf_GR_m        = 0x1f,sincosf_GR_n // Put mask 0x1F  -
      nop.m         999                                     // - select k+1 bits
      nop.i         999
};;

// Add 16*M to address of sin_cos_beta table
{ .mfi
      shladd        sincosf_AD_2        = sincosf_GR_32m, 4, sincosf_AD_1
(p8)  fclass.m.unc  p10,p0              = f8,0x0b  // If sin denormal input -
      nop.i         999
};;

// Load Sin and Cos table value using obtained index m  (sincosf_AD_2)
{ .mfi
      ldfd          sincosf_Sm          = [sincosf_AD_2],8 // Sin value S[m]
(p9)  fclass.m.unc  p11,p0              = f8,0x0b  // If cos denormal input -
      nop.i         999                            // - set denormal
};;

// sincosf_r          = sincosf_r -sincosf_Nfloat * sincosf_Pi_by_16_2
{ .mfi
      ldfd          sincosf_Cm          = [sincosf_AD_2] // Cos table value C[m]
      fnma.s1  sincosf_r_exact = sincosf_NFLOAT, sincosf_Pi_by_16_2, sincosf_r
      nop.i         999
}
// get rsq = r*r
{ .mfi
      nop.m         999
      fma.s1        sincosf_rsq         = sincosf_r, sincosf_r,  f0 // r^2 = r*r
      nop.i         999
};;

{ .mfi
      nop.m         999
      fmpy.s0       fp_tmp              = fp_tmp, fp_tmp // forces inexact flag
      nop.i         999
};;

// Polynomials calculation
// Q = Q2*r^2 + Q1
// P = P2*r^2 + P1
{ .mfi
      nop.m         999
      fma.s1        sincosf_Q           = sincosf_rsq, sincosf_Q2, sincosf_Q1
      nop.i         999
}
{ .mfi
      nop.m         999
      fma.s1        sincosf_P           = sincosf_rsq, sincosf_P2, sincosf_P1
      nop.i         999
};;

// get rcube and S[m]*r^2
{ .mfi
      nop.m         999
      fmpy.s1       sincosf_srsq        = sincosf_Sm,sincosf_rsq // r^2*S[m]
      nop.i         999
}
{ .mfi
      nop.m         999
      fmpy.s1       sincosf_rcub        = sincosf_r_exact, sincosf_rsq
      nop.i         999
};;

// Get final P and Q
// Q = Q*S[m]*r^2 + S[m]
// P = P*r^3 + r
{ .mfi
      nop.m         999
      fma.s1        sincosf_Q           = sincosf_srsq,sincosf_Q, sincosf_Sm
      nop.i         999
}
{ .mfi
      nop.m         999
      fma.s1        sincosf_P           = sincosf_rcub,sincosf_P,sincosf_r_exact
      nop.i         999
};;

// If sinf(denormal) - force underflow to be set
.pred.rel "mutex",p10,p11
{ .mfi
      nop.m         999
(p10) fmpy.s.s0     fp_tmp              = f8,f8 // forces underflow flag
      nop.i         999                         // for denormal sine args
}
// If cosf(denormal) - force denormal to be set
{ .mfi
      nop.m         999
(p11) fma.s.s0     fp_tmp              = f8, f1, f8 // forces denormal flag
      nop.i         999                              // for denormal cosine args
};;


// Final calculation
// result = C[m]*P + Q
{ .mfb
      nop.m         999
      fma.s.s0      f8                  = sincosf_Cm, sincosf_P, sincosf_Q
      br.ret.sptk   b0 // Exit for common path
};;

////////// x = 0/Inf/NaN path //////////////////
_SINCOSF_SPECIAL_ARGS:
.pred.rel "mutex",p8,p9
// sinf(+/-0) = +/-0
// sinf(Inf)  = NaN
// sinf(NaN)  = NaN
{ .mfi
      nop.m         999
(p8)  fma.s.s0      f8                  = f8, f0, f0 // sinf(+/-0,NaN,Inf)
      nop.i         999
}
// cosf(+/-0) = 1.0
// cosf(Inf)  = NaN
// cosf(NaN)  = NaN
{ .mfb
      nop.m         999
(p9)  fma.s.s0      f8                  = f8, f0, f1 // cosf(+/-0,NaN,Inf)
      br.ret.sptk   b0 // Exit for x = 0/Inf/NaN path
};;

GLOBAL_IEEE754_END(cosf)
libm_alias_float_other (__cos, cos)

//////////// x >= 2^24 - large arguments routine call ////////////
LOCAL_LIBM_ENTRY(__libm_callout_sincosf)
_SINCOSF_LARGE_ARGS:
.prologue
{ .mfi
      mov           sincosf_GR_all_ones = -1 // 0xffffffff
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
      setf.sig      sincosf_save_tmp    = sincosf_GR_all_ones  // inexact set
      nop.b         999
(p8)  br.call.sptk.many b0              = __libm_sin_large# // sinf(large_X)
};;

{ .mbb
      cmp.ne        p9,p0               = sincosf_r_sincos, r0 // set p9 if cos
      nop.b         999
(p9)  br.call.sptk.many b0              = __libm_cos_large# // cosf(large_X)
};;

{ .mfi
      mov           gp                  = GR_SAVE_GP
      fma.s.s0      f8                  = f8, f1, f0 // Round result to single
      mov           b0                  = GR_SAVE_B0
}
{ .mfi // force inexact set
      nop.m         999
      fmpy.s0       sincosf_save_tmp    = sincosf_save_tmp, sincosf_save_tmp
      nop.i         999
};;

{ .mib
      nop.m         999
      mov           ar.pfs              = GR_SAVE_PFS
      br.ret.sptk   b0 // Exit for large arguments routine call
};;
LOCAL_LIBM_END(__libm_callout_sincosf)

.type    __libm_sin_large#, @function
.global  __libm_sin_large#
.type    __libm_cos_large#, @function
.global  __libm_cos_large#
