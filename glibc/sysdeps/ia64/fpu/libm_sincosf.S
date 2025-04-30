.file "libm_sincosf.s"


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
// 02/26/02 Added temporary return of results in r8, r9
// 03/13/02 Corrected restore of predicate registers
// 03/19/02 Added stack unwind around call to __libm_cisf_large
// 09/05/02 Work range is widened by reduction strengthen (2 parts of Pi/16)
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 02/11/04 cisf is moved to the separate file.
// 03/31/05 Reformatted delimiters between data tables

// API
//==============================================================
// 1) void sincosf(float, float*s, float*c)
// 2) __libm_sincosf - internal LIBM function, that accepts
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
//    Do this as (x -  nfloat * HIGH(pi/2^k)) - nfloat * LOW(pi/2^k) for increased accuracy.
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
// Cos(r) = 1 + r^2 q1  + r^4 q2 = Series for Cos
// Sin(r) = r + r^3 p1  + r^5 p2 = Series for Sin
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
//    P = p1 + r^2p2
//    Q = q1 + r^2q2
//
//       rcub = r * rsq
//       Sin(r) = r + rcub * P
//              = r + r^3p1  + r^5p2 = Sin(r)
//
//       P =  r + rcub * P
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
//       Q           = S[m] + s_rsq Q
//
// Then,
//
//    Answer = Q + C[m] P


// Registers used
//==============================================================
// general input registers:
// r14 -> r19
// r32 -> r49

// predicate registers used:
// p6 -> p14

// floating-point registers used
// f9 -> f15
// f32 -> f100

// Assembly macros
//==============================================================

cisf_Arg                     = f8

cisf_Sin_res                 = f9
cisf_Cos_res                 = f8


cisf_NORM_f8                 = f10
cisf_W                       = f11
cisf_int_Nfloat              = f12
cisf_Nfloat                  = f13

cisf_r                       = f14
cisf_r_exact                 = f68
cisf_rsq                     = f15
cisf_rcub                    = f32

cisf_Inv_Pi_by_16            = f33
cisf_Pi_by_16_hi             = f34
cisf_Pi_by_16_lo             = f35

cisf_Inv_Pi_by_64            = f36
cisf_Pi_by_64_hi             = f37
cisf_Pi_by_64_lo             = f38


cisf_P1                      = f39
cisf_Q1                      = f40
cisf_P2                      = f41
cisf_Q2                      = f42
cisf_P3                      = f43
cisf_Q3                      = f44
cisf_P4                      = f45
cisf_Q4                      = f46

cisf_P_temp1                 = f47
cisf_P_temp2                 = f48

cisf_Q_temp1                 = f49
cisf_Q_temp2                 = f50

cisf_P                       = f51

cisf_SIG_INV_PI_BY_16_2TO61  = f52
cisf_RSHF_2TO61              = f53
cisf_RSHF                    = f54
cisf_2TOM61                  = f55
cisf_NFLOAT                  = f56
cisf_W_2TO61_RSH             = f57

cisf_tmp                     = f58

cisf_Sm_sin                  = f59
cisf_Cm_sin                  = f60

cisf_Sm_cos                  = f61
cisf_Cm_cos                  = f62

cisf_srsq_sin                = f63
cisf_srsq_cos                = f64

cisf_Q_sin                   = f65
cisf_Q_cos                   = f66
cisf_Q                       = f67

/////////////////////////////////////////////////////////////

cisf_pResSin                 = r33
cisf_pResCos                 = r34

cisf_exp_limit               = r35
cisf_r_signexp               = r36
cisf_AD_beta_table           = r37
cisf_r_sincos                = r38

cisf_r_exp                   = r39
cisf_r_17_ones               = r40

cisf_GR_sig_inv_pi_by_16     = r14
cisf_GR_rshf_2to61           = r15
cisf_GR_rshf                 = r16
cisf_GR_exp_2tom61           = r17
cisf_GR_n                    = r18

cisf_GR_n_sin                = r19
cisf_GR_m_sin                = r41
cisf_GR_32m_sin              = r41

cisf_GR_n_cos                = r42
cisf_GR_m_cos                = r43
cisf_GR_32m_cos              = r43

cisf_AD_2_sin                = r44
cisf_AD_2_cos                = r45

cisf_gr_tmp                  = r46
GR_SAVE_B0                   = r47
GR_SAVE_GP                   = r48
rB0_SAVED                    = r49
GR_SAVE_PFS                  = r50
GR_SAVE_PR                   = r51
cisf_AD_1                    = r52

RODATA

.align 16
// Pi/16 parts
LOCAL_OBJECT_START(double_cisf_pi)
   data8 0xC90FDAA22168C234, 0x00003FFC // pi/16 1st part
   data8 0xC4C6628B80DC1CD1, 0x00003FBC // pi/16 2nd part
LOCAL_OBJECT_END(double_cisf_pi)

// Coefficients for polynomials
LOCAL_OBJECT_START(double_cisf_pq_k4)
   data8 0x3F810FABB668E9A2 // P2
   data8 0x3FA552E3D6DE75C9 // Q2
   data8 0xBFC555554447BC7F // P1
   data8 0xBFDFFFFFC447610A // Q1
LOCAL_OBJECT_END(double_cisf_pq_k4)

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

GLOBAL_IEEE754_ENTRY(sincosf)
// cis_GR_sig_inv_pi_by_16 = significand of 16/pi
{ .mlx
      alloc         GR_SAVE_PFS              = ar.pfs, 0, 21, 0, 0
      movl          cisf_GR_sig_inv_pi_by_16 = 0xA2F9836E4E44152A // 16/pi signd

}
// cis_GR_rshf_2to61 = 1.1000 2^(63+63-2)
{ .mlx
      addl          cisf_AD_1           = @ltoff(double_cisf_pi), gp
      movl          cisf_GR_rshf_2to61  = 0x47b8000000000000 // 1.1 2^(63+63-2)
};;

{ .mfi
      ld8           cisf_AD_1           = [cisf_AD_1]
      fnorm.s1      cisf_NORM_f8        = cisf_Arg
      cmp.eq        p13, p14            = r0, r0 // p13 set for sincos
}
// cis_GR_exp_2tom61 = exponent of scaling factor 2^-61
{ .mib
      mov           cisf_GR_exp_2tom61  = 0xffff-61
      nop.i         0
      br.cond.sptk  _CISF_COMMON
};;
GLOBAL_IEEE754_END(sincosf)
libm_alias_float_other (__sincos, sincos)

GLOBAL_LIBM_ENTRY(__libm_sincosf)
{ .mlx
// cisf_GR_sig_inv_pi_by_16 = significand of 16/pi
      alloc         GR_SAVE_PFS              = ar.pfs,0,21,0,0
      movl          cisf_GR_sig_inv_pi_by_16 = 0xA2F9836E4E44152A
}
// cisf_GR_rshf_2to61 = 1.1000 2^(63+63-2)
{ .mlx
      addl          cisf_AD_1           = @ltoff(double_cisf_pi), gp
      movl          cisf_GR_rshf_2to61  = 0x47b8000000000000
};;

// p14 set for __libm_sincos and cis
{ .mfi
      ld8           cisf_AD_1           = [cisf_AD_1]
      fnorm.s1      cisf_NORM_f8        = cisf_Arg
      cmp.eq        p14, p13            = r0, r0
}
// cisf_GR_exp_2tom61 = exponent of scaling factor 2^-61
{ .mib
      mov           cisf_GR_exp_2tom61  = 0xffff-61
      nop.i         0
      nop.b         0
};;

_CISF_COMMON:
//  Form two constants we need
//  16/pi * 2^-2 * 2^63, scaled by 2^61 since we just loaded the significand
//  1.1000...000 * 2^(63+63-2) to right shift int(W) into the low significand
//  fcmp used to set denormal, and invalid on snans
{ .mfi
      setf.sig      cisf_SIG_INV_PI_BY_16_2TO61 = cisf_GR_sig_inv_pi_by_16
      fclass.m      p6,p0                       = cisf_Arg, 0xe7//if x=0,inf,nan
      addl          cisf_gr_tmp                 = -1, r0
}
// cisf_GR_rshf = 1.1000 2^63 for right shift
{ .mlx
      setf.d        cisf_RSHF_2TO61     = cisf_GR_rshf_2to61
      movl          cisf_GR_rshf        = 0x43e8000000000000
};;

//  Form another constant
//  2^-61 for scaling Nfloat
//  0x10017 is register_bias + 24.
//  So if f8 >= 2^24, go to large args routine
{ .mmi
      getf.exp      cisf_r_signexp      = cisf_Arg
      setf.exp      cisf_2TOM61         = cisf_GR_exp_2tom61
      mov           cisf_exp_limit      = 0x10017
};;

// Load the two pieces of pi/16
// Form another constant
//  1.1000...000 * 2^63, the right shift constant
{ .mmb
      ldfe          cisf_Pi_by_16_hi    = [cisf_AD_1],16
      setf.d        cisf_RSHF           = cisf_GR_rshf
(p6)  br.cond.spnt  _CISF_SPECIAL_ARGS
};;

{ .mmi
      ldfe          cisf_Pi_by_16_lo    = [cisf_AD_1],16
      setf.sig      cisf_tmp            = cisf_gr_tmp //constant for inexact set
      nop.i         0
};;

// Start loading P, Q coefficients
{ .mmi
      ldfpd         cisf_P2,cisf_Q2     = [cisf_AD_1],16
      nop.m         0
      dep.z         cisf_r_exp          = cisf_r_signexp, 0, 17
};;

// p10 is true if we must call routines to handle larger arguments
// p10 is true if f8 exp is >= 0x10017
{ .mmb
      ldfpd         cisf_P1,cisf_Q1     = [cisf_AD_1], 16
      cmp.ge        p10, p0             = cisf_r_exp, cisf_exp_limit
(p10) br.cond.spnt  _CISF_LARGE_ARGS    // go to |x| >= 2^24 path
};;

// cisf_W          = x * cisf_Inv_Pi_by_16
// Multiply x by scaled 16/pi and add large const to shift integer part of W to
//   rightmost bits of significand
{ .mfi
      nop.m  0
      fma.s1 cisf_W_2TO61_RSH = cisf_NORM_f8,cisf_SIG_INV_PI_BY_16_2TO61,cisf_RSHF_2TO61
      nop.i  0
};;

// cisf_NFLOAT = Round_Int_Nearest(cisf_W)
{ .mfi
      nop.m         0
      fms.s1        cisf_NFLOAT         = cisf_W_2TO61_RSH,cisf_2TOM61,cisf_RSHF
      nop.i         0
};;

// N = (int)cisf_int_Nfloat
{ .mfi
      getf.sig      cisf_GR_n           = cisf_W_2TO61_RSH
      nop.f         0
      nop.i         0
};;

// Add 2^(k-1) (which is in cisf_r_sincos) to N
// cisf_r = -cisf_Nfloat * cisf_Pi_by_16_hi + x
// cisf_r = cisf_r -cisf_Nfloat * cisf_Pi_by_16_lo
{ .mfi
      add     cisf_GR_n_cos = 0x8, cisf_GR_n
      fnma.s1 cisf_r        = cisf_NFLOAT, cisf_Pi_by_16_hi, cisf_NORM_f8
      nop.i   0
};;

//Get M (least k+1 bits of N)
{ .mmi
      and           cisf_GR_m_sin       = 0x1f,cisf_GR_n
      and           cisf_GR_m_cos       = 0x1f,cisf_GR_n_cos
      nop.i         0
};;

{ .mmi
      shladd        cisf_AD_2_cos       = cisf_GR_m_cos,4, cisf_AD_1
      shladd        cisf_AD_2_sin       = cisf_GR_m_sin,4, cisf_AD_1
      nop.i         0
};;

// den. input to set uflow
{ .mmf
      ldfpd         cisf_Sm_sin, cisf_Cm_sin = [cisf_AD_2_sin]
      ldfpd         cisf_Sm_cos, cisf_Cm_cos = [cisf_AD_2_cos]
      fclass.m.unc  p10,p0                   = cisf_Arg,0x0b
};;

{ .mfi
      nop.m         0
      fma.s1        cisf_rsq            = cisf_r, cisf_r,   f0  // get r^2
      nop.i         0
}
{ .mfi
      nop.m         0
      fmpy.s0       cisf_tmp            = cisf_tmp,cisf_tmp // inexact flag
      nop.i         0
};;

{ .mmf
      nop.m         0
      nop.m         0
      fnma.s1       cisf_r_exact        = cisf_NFLOAT, cisf_Pi_by_16_lo, cisf_r
};;

{ .mfi
      nop.m         0
      fma.s1        cisf_P              = cisf_rsq, cisf_P2, cisf_P1
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        cisf_Q              = cisf_rsq, cisf_Q2, cisf_Q1
      nop.i         0
};;

{ .mfi
      nop.m         0
      fmpy.s1       cisf_rcub           = cisf_r_exact, cisf_rsq // get r^3
      nop.i         0
};;

{ .mfi
      nop.m         0
      fmpy.s1       cisf_srsq_sin       = cisf_Sm_sin,cisf_rsq
      nop.i         0
}
{ .mfi
      nop.m         0
      fmpy.s1       cisf_srsq_cos       = cisf_Sm_cos,cisf_rsq
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.s1        cisf_P              = cisf_rcub,cisf_P,cisf_r_exact
      nop.i         0
};;

{ .mfi
      nop.m         0
      fma.s1        cisf_Q_sin          = cisf_srsq_sin,cisf_Q, cisf_Sm_sin
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        cisf_Q_cos          = cisf_srsq_cos,cisf_Q, cisf_Sm_cos
      nop.i         0
};;

// If den. arg, force underflow to be set
{ .mfi
      nop.m         0
(p10) fmpy.s.s0     cisf_tmp            = cisf_Arg,cisf_Arg
      nop.i         0
};;

//Final sin
{ .mfi
      nop.m         0
      fma.s.s0      cisf_Sin_res        = cisf_Cm_sin, cisf_P, cisf_Q_sin
      nop.i         0
}
//Final cos
{ .mfb
      nop.m         0
      fma.s.s0      cisf_Cos_res    = cisf_Cm_cos, cisf_P, cisf_Q_cos
(p14) br.cond.sptk  _CISF_RETURN //com. exit for __libm_sincos and cis main path
};;

{ .mmb
      stfs          [cisf_pResSin]      = cisf_Sin_res
      stfs          [cisf_pResCos]      = cisf_Cos_res
      br.ret.sptk   b0 // common exit for sincos main path
};;

_CISF_SPECIAL_ARGS:
// sinf(+/-0) = +/-0
// sinf(Inf)  = NaN
// sinf(NaN)  = NaN
{ .mfi
      nop.m         999
      fma.s.s0      cisf_Sin_res        = cisf_Arg, f0, f0 // sinf(+/-0,NaN,Inf)
      nop.i         999
};;

// cosf(+/-0) = 1.0
// cosf(Inf)  = NaN
// cosf(NaN)  = NaN
{ .mfb
      nop.m         999
      fma.s.s0      cisf_Cos_res        = cisf_Arg, f0, f1 // cosf(+/-0,NaN,Inf)
(p14) br.cond.sptk  _CISF_RETURN //spec exit for __libm_sincos and cis main path
};;

{ .mmb
      stfs          [cisf_pResSin]      = cisf_Sin_res
      stfs          [cisf_pResCos]      = cisf_Cos_res
      br.ret.sptk   b0 // special exit for sincos main path
};;

 // exit for sincos
 // NOTE! r8 and r9 used only because of compiler issue
 // connected with float point complex function arguments pass
 // After fix of this issue this operations can be deleted
_CISF_RETURN:
{ .mmb
      getf.s        r8                  = cisf_Cos_res
      getf.s        r9                  = cisf_Sin_res
      br.ret.sptk   b0 // exit for sincos
};;
GLOBAL_LIBM_END(__libm_sincosf)

////  |x| > 2^24 path  ///////
.proc _CISF_LARGE_ARGS
_CISF_LARGE_ARGS:
.prologue
{ .mfi
      nop.m         0
      nop.f         0
.save ar.pfs, GR_SAVE_PFS
      mov           GR_SAVE_PFS         = ar.pfs
};;

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
      fma.s.s0      cisf_Cos_res        = cisf_Cos_res, f1, f0
      mov           ar.pfs              = GR_SAVE_PFS
}
// exit for |x| > 2^24 path (__libm_sincos and cis)
{ .mfb
      nop.m         0
      fma.s.s0      cisf_Sin_res        = cisf_Sin_res, f1, f0
(p14) br.cond.sptk  _CISF_RETURN
};;

{ .mmb
      stfs          [cisf_pResSin]      = cisf_Sin_res
      stfs          [cisf_pResCos]      = cisf_Cos_res
      br.ret.sptk   b0 // exit for sincos |x| > 2^24 path
};;

.endp _CISF_LARGE_ARGS

.type   __libm_sincos_large#,@function
.global __libm_sincos_large#
