.file "tancotf.s"


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
// 04/04/00 Unwind support added
// 12/27/00 Improved speed
// 02/21/01 Updated to call tanl
// 05/30/02 Improved speed, added cotf.
// 11/25/02 Added explicit completer on fnorm
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 04/17/03 Eliminated redundant stop bits
// 03/31/05 Reformatted delimiters between data tables
//
// APIs
//==============================================================
// float tanf(float)
// float cotf(float)
//
// Algorithm Description for tanf
//==============================================================
// The tanf function computes the principle value of the tangent of x,
// where x is radian argument.
//
// There are 5 paths:
// 1. x = +/-0.0
//    Return tanf(x) = +/-0.0
//
// 2. x = [S,Q]NaN
//    Return tanf(x) = QNaN
//
// 3. x = +/-Inf
//    Return tanf(x) = QNaN
//
// 4. x = r + (Pi/2)*N, N = RoundInt(x*(2/Pi)), N is even, |r|<Pi/4
//    Return tanf(x) = P19(r) = A1*r + A3*r^3 + A5*r^5 + ... + A19*r^19 =
//    = r*(A1 + A3*t + A5*t^2 + ... + A19*t^9) = r*P9(t), where t = r^2
//
// 5. x = r + (Pi/2)*N, N = RoundInt(x*(2/Pi)), N is odd, |r|<Pi/4
//    Return tanf(x) = -1/r + P11(r) = -1/r + B1*r + B3*r^3 + ... + B11*r^11 =
//    = -1/r + r*(B1 + B3*t + B5*t^2 + ... + B11*t^5) = -1/r + r*P11(t),
//    where t = r^2
//
// Algorithm Description for cotf
//==============================================================
// The cotf function computes the principle value of the cotangent of x,
// where x is radian argument.
//
// There are 5 paths:
// 1. x = +/-0.0
//    Return cotf(x) = +/-Inf and error handling is called
//
// 2. x = [S,Q]NaN
//    Return cotf(x) = QNaN
//
// 3. x = +/-Inf
//    Return cotf(x) = QNaN
//
// 4. x = r + (Pi/2)*N, N = RoundInt(x*(2/Pi)), N is odd, |r|<Pi/4
//    Return cotf(x) = P19(-r) = A1*(-r) + A3*(-r^3) + ... + A19*(-r^19) =
//    = -r*(A1 + A3*t + A5*t^2 + ... + A19*t^9) = -r*P9(t), where t = r^2
//
// 5. x = r + (Pi/2)*N, N = RoundInt(x*(2/Pi)), N is even, |r|<Pi/4
//    Return cotf(x) = 1/r + P11(-r) = 1/r + B1*(-r) + ... + B11*(-r^11) =
//    = 1/r - r*(B1 + B3*t + B5*t^2 + ... + B11*t^5) = 1/r - r*P11(t),
//    where t = r^2
//
//    We set p10 and clear p11 if computing tanf, vice versa for cotf.
//
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f32 -> f80
//
// General registers used:
// r14 -> r23, r32 -> r39
//
// Predicate registers used:
// p6 -> p13
//
// Assembly macros
//==============================================================
// integer registers
rExp                        = r14
rSignMask                   = r15
rRshf                       = r16
rScFctrExp                  = r17
rIntN                       = r18
rSigRcpPiby2                = r19
rScRshf                     = r20
rCoeffA                     = r21
rCoeffB                     = r22
rExpCut                     = r23

GR_SAVE_B0                  = r33
GR_SAVE_PFS                 = r34
GR_SAVE_GP                  = r35
GR_Parameter_X              = r36
GR_Parameter_Y              = r37
GR_Parameter_RESULT         = r38
GR_Parameter_Tag            = r39

//==============================================================
// floating point registers
fScRcpPiby2                 = f32
fScRshf                     = f33
fNormArg                    = f34
fScFctr                     = f35
fRshf                       = f36
fShiftedN                   = f37
fN                          = f38
fR                          = f39
fA01                        = f40
fA03                        = f41
fA05                        = f42
fA07                        = f43
fA09                        = f44
fA11                        = f45
fA13                        = f46
fA15                        = f47
fA17                        = f48
fA19                        = f49
fB01                        = f50
fB03                        = f51
fB05                        = f52
fB07                        = f53
fB09                        = f54
fB11                        = f55
fA03_01                     = f56
fA07_05                     = f57
fA11_09                     = f58
fA15_13                     = f59
fA19_17                     = f60
fA11_05                     = f61
fA19_13                     = f62
fA19_05                     = f63
fRbyA03_01                  = f64
fB03_01                     = f65
fB07_05                     = f66
fB11_09                     = f67
fB11_05                     = f68
fRbyB03_01                  = f69
fRbyB11_01                  = f70
fRp2                        = f71
fRp4                        = f72
fRp8                        = f73
fRp5                        = f74
fY0                         = f75
fY1                         = f76
fD                          = f77
fDp2                        = f78
fInvR                       = f79
fPiby2                      = f80
//==============================================================


RODATA
.align 16

LOCAL_OBJECT_START(coeff_A)
data8 0x3FF0000000000000 // A1  = 1.00000000000000000000e+00
data8 0x3FD5555556BCE758 // A3  = 3.33333334641442641606e-01
data8 0x3FC111105C2DAE48 // A5  = 1.33333249100689099175e-01
data8 0x3FABA1F876341060 // A7  = 5.39701122561673229739e-02
data8 0x3F965FB86D12A38D // A9  = 2.18495194027670719750e-02
data8 0x3F8265F62415F9D6 // A11 = 8.98353860497717439465e-03
data8 0x3F69E3AE64CCF58D // A13 = 3.16032468108912746342e-03
data8 0x3F63920D09D0E6F6 // A15 = 2.38897844840557235331e-03
LOCAL_OBJECT_END(coeff_A)

LOCAL_OBJECT_START(coeff_B)
data8 0xC90FDAA22168C235, 0x3FFF // pi/2
data8 0x3FD55555555358DB // B1  = 3.33333333326107426583e-01
data8 0x3F96C16C252F643F // B3  = 2.22222230621336129239e-02
data8 0x3F61566243AB3C60 // B5  = 2.11638633968606896785e-03
data8 0x3F2BC1169BD4438B // B7  = 2.11748132564551094391e-04
data8 0x3EF611B4CEA056A1 // B9  = 2.10467959860990200942e-05
data8 0x3EC600F9E32194BF // B11 = 2.62305891234274186608e-06
data8 0xBF42BA7BCC177616 // A17 =-5.71546981685324877205e-04
data8 0x3F4F2614BC6D3BB8 // A19 = 9.50584530849832782542e-04
LOCAL_OBJECT_END(coeff_B)


.section .text

LOCAL_LIBM_ENTRY(cotf)

{ .mlx
      getf.exp  rExp        = f8                    // ***** Get 2ˆ17 * s + E
      movl      rSigRcpPiby2= 0xA2F9836E4E44152A    // significand of 2/Pi
}
{ .mlx
      addl      rCoeffA     = @ltoff(coeff_A), gp
      movl      rScRshf     = 0x47e8000000000000    // 1.5*2^(63+63+1)
}
;;

{ .mfi
      alloc     r32         = ar.pfs, 0, 4, 4, 0
      fclass.m  p9, p0      = f8, 0xc3              // Test for x=nan
      cmp.eq    p11, p10    = r0, r0                // if p11=1 we compute cotf
}
{ .mib
      ld8       rCoeffA     = [rCoeffA]
      mov       rExpCut     = 0x10009               // cutoff for exponent
      br.cond.sptk Common_Path
}
;;

LOCAL_LIBM_END(cotf)


GLOBAL_IEEE754_ENTRY(tanf)

{ .mlx
      getf.exp  rExp        = f8                    // ***** Get 2ˆ17 * s + E
      movl      rSigRcpPiby2= 0xA2F9836E4E44152A    // significand of 2/Pi
}
{ .mlx
      addl      rCoeffA     = @ltoff(coeff_A), gp
      movl      rScRshf     = 0x47e8000000000000    // 1.5*2^(63+63+1)
}
;;

{ .mfi
      alloc     r32         = ar.pfs, 0, 4, 4, 0
      fclass.m  p9, p0      = f8, 0xc3              // Test for x=nan
      cmp.eq    p10, p11    = r0, r0                // if p10=1 we compute tandf
}
{ .mib
      ld8       rCoeffA     = [rCoeffA]
      mov       rExpCut     = 0x10009               // cutoff for exponent
      nop.b     0
}
;;

// Below is common path for both tandf and cotdf
Common_Path:
{ .mfi
      setf.sig  fScRcpPiby2 = rSigRcpPiby2          // 2^(63+1)*(2/Pi)
      fclass.m  p8, p0      = f8, 0x23              // Test for x=inf
      mov       rSignMask   = 0x1ffff               // mask for sign bit
}
{ .mlx
      setf.d    fScRshf     = rScRshf               // 1.5*2^(63+63+1)
      movl      rRshf       = 0x43e8000000000000    // 1.5 2^63 for right shift
}
;;

{ .mfi
      and       rSignMask   = rSignMask, rExp       // clear sign bit
(p10) fclass.m.unc p7, p0   = f8, 0x07              // Test for x=0 (for tanf)
      mov       rScFctrExp  = 0xffff-64             // exp of scaling factor
}
{ .mfb
      adds      rCoeffB     = coeff_B - coeff_A, rCoeffA
(p9)  fma.s.s0  f8          = f8, f1, f8            // Set qnan if x=nan
(p9)  br.ret.spnt b0                                // Exit for x=nan
}
;;

{ .mfi
      cmp.ge    p6, p0      = rSignMask, rExpCut    // p6 = (E => 0x10009)
(p8)  frcpa.s0  f8, p0      = f0, f0                // Set qnan indef if x=inf
      mov GR_Parameter_Tag  = 227                   // (cotf)
}
{ .mbb
      ldfe      fPiby2      = [rCoeffB], 16
(p8)  br.ret.spnt b0                                // Exit for x=inf
(p6)  br.cond.spnt Huge_Argument                    // Branch if |x|>=2^10
}
;;

{ .mfi
      nop.m     0
(p11) fclass.m.unc p6, p0   = f8, 0x07              // Test for x=0 (for cotf)
      nop.i     0
}
{ .mfb
      nop.m     0
      fnorm.s0  fNormArg    = f8
(p7)  br.ret.spnt b0                                // Exit for x=0 (for tanf)
}
;;

{ .mmf
      ldfpd     fA01, fA03  = [rCoeffA], 16
      ldfpd     fB01, fB03  = [rCoeffB], 16
      fmerge.s  f10         = f8, f8                // Save input for error call
}
;;

{ .mmf
      setf.exp  fScFctr     = rScFctrExp            // get as real
      setf.d    fRshf       = rRshf                 // get right shifter as real
(p6)  frcpa.s0  f8, p0      = f1, f8                // cotf(+-0) = +-Inf
}
;;

{ .mmb
      ldfpd     fA05, fA07  = [rCoeffA], 16
      ldfpd     fB05, fB07  = [rCoeffB], 16
(p6)  br.cond.spnt __libm_error_region    // call error support if cotf(+-0)
}
;;

{ .mmi
      ldfpd     fA09, fA11  = [rCoeffA], 16
      ldfpd     fB09, fB11  = [rCoeffB], 16
      nop.i     0
}
;;

{ .mfi
      nop.m     0
      fma.s1    fShiftedN = fNormArg,fScRcpPiby2,fScRshf // x*2^70*(2/Pi)+ScRshf
      nop.i     0
}
;;

{ .mfi
      nop.m     0
      fms.s1    fN          = fShiftedN, fScFctr, fRshf // N = Y*2^(-70) - Rshf
      nop.i     0
}
;;

.pred.rel "mutex", p10, p11
{ .mfi
      getf.sig  rIntN       = fShiftedN             // get N as integer
(p10) fnma.s1   fR          = fN, fPiby2, fNormArg  // R = x - (Pi/2)*N (tanf)
      nop.i     0
}
{ .mfi
      nop.m     0
(p11) fms.s1    fR          = fN, fPiby2, fNormArg  // R = (Pi/2)*N - x (cotf)
      nop.i     0
}
;;

{ .mmi
      ldfpd     fA13, fA15  = [rCoeffA], 16
      ldfpd     fA17, fA19  = [rCoeffB], 16
      nop.i     0
}
;;

Return_From_Huges:
{ .mfi
      nop.m     0
      fma.s1    fRp2        = fR, fR, f0            // R^2
(p11) add       rIntN       = 0x1, rIntN            // N = N + 1 (cotf)
}
;;

{ .mfi
      nop.m     0
      frcpa.s1  fY0, p0     = f1, fR                // Y0 ~ 1/R
      tbit.z    p8, p9      = rIntN, 0              // p8=1 if N is even
}
;;

// Below are mixed polynomial calculations (mixed for even and odd N)
{ .mfi
      nop.m     0
(p9)  fma.s1    fB03_01     = fRp2, fB03, fB01      // R^2*B3 + B1
      nop.i     0
}
{ .mfi
      nop.m     0
      fma.s1    fRp4        = fRp2, fRp2, f0        // R^4
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p8)  fma.s1    fA15_13     = fRp2, fA15, fA13      // R^2*A15 + A13
      nop.i     0
}
{ .mfi
      nop.m     0
(p8)  fma.s1    fA19_17     = fRp2, fA19, fA17      // R^2*A19 + A17
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p8)  fma.s1    fA07_05     = fRp2, fA07, fA05      // R^2*A7 + A5
      nop.i     0
}
{ .mfi
      nop.m     0
(p8)  fma.s1    fA11_09     = fRp2, fA11, fA09      // R^2*A11 + A9
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p9)  fma.s1    fB07_05     = fRp2, fB07, fB05      // R^2*B7 + B5
      nop.i     0
}
{ .mfi
      nop.m     0
(p9)  fma.s1    fB11_09     = fRp2, fB11, fB09      // R^2*B11 + B9
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p9)  fnma.s1   fD          = fR, fY0, f1           // D = 1 - R*Y0
      nop.i     0
}
{ .mfi
      nop.m     0
(p8)  fma.s1    fA03_01     = fRp2, fA03, fA01      // R^2*A3 + A1
      nop.i     0
}
;;

{ .mfi
      nop.m     0
      fma.s1    fRp8        = fRp4, fRp4, f0        // R^8
      nop.i     0
}
{ .mfi
      nop.m     0
      fma.s1    fRp5        = fR, fRp4, f0          // R^5
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p8)  fma.s1    fA11_05     = fRp4, fA11_09, fA07_05 // R^4*(R^2*A11 + A9) + ...
      nop.i     0
}
{ .mfi
      nop.m     0
(p8)  fma.s1    fA19_13     = fRp4, fA19_17, fA15_13 // R^4*(R^2*A19 + A17) + ..
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p9)  fma.s1    fB11_05     = fRp4, fB11_09, fB07_05 // R^4*(R^2*B11 + B9) + ...
      nop.i     0
}
{ .mfi
      nop.m     0
(p9)  fma.s1    fRbyB03_01  = fR, fB03_01, f0       // R*(R^2*B3 + B1)
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p9)  fma.s1    fY1         = fY0, fD, fY0          // Y1 = Y0*D + Y0
      nop.i     0
}
{ .mfi
      nop.m     0
(p9)  fma.s1    fDp2        = fD, fD, f0            // D^2
      nop.i     0
}
;;

{ .mfi
      nop.m     0
   // R^8*(R^6*A19 + R^4*A17 + R^2*A15 + A13) + R^6*A11 + R^4*A9 + R^2*A7 + A5
(p8)  fma.d.s1  fA19_05     = fRp8, fA19_13, fA11_05
      nop.i     0
}
{ .mfi
      nop.m     0
(p8)  fma.d.s1  fRbyA03_01  = fR, fA03_01, f0       // R*(R^2*A3 + A1)
      nop.i     0
}
;;

{ .mfi
      nop.m     0
(p9)  fma.d.s1  fInvR       = fY1, fDp2, fY1        // 1/R = Y1*D^2 + Y1
      nop.i     0
}
{ .mfi
      nop.m     0
   // R^5*(R^6*B11 + R^4*B9 + R^2*B7 + B5) + R^3*B3 + R*B1
(p9)  fma.d.s1  fRbyB11_01  = fRp5, fB11_05, fRbyB03_01
      nop.i     0
}
;;

.pred.rel "mutex", p8, p9
{ .mfi
      nop.m     0
   // Result = R^5*(R^14*A19 + R^12*A17 + R^10*A15 + ...) + R^3*A3 + R*A1
(p8)  fma.s.s0  f8          = fRp5, fA19_05, fRbyA03_01
      nop.i 0
}
{ .mfb
      nop.m     0
   // Result = -1/R + R^11*B11 + R^9*B9 + R^7*B7 + R^5*B5 + R^3*B3 + R*B1
(p9)  fnma.s.s0 f8          = f1, fInvR, fRbyB11_01
      br.ret.sptk b0                                // exit for main path
}
;;

GLOBAL_IEEE754_END(tanf)
libm_alias_float_other (__tan, tan)


LOCAL_LIBM_ENTRY(__libm_callout)
Huge_Argument:
.prologue

{ .mfi
      nop.m 0
      fmerge.s f9 = f0,f0
.save ar.pfs,GR_SAVE_PFS
      mov  GR_SAVE_PFS=ar.pfs
}
;;

{ .mfi
      mov GR_SAVE_GP=gp
      nop.f 0
.save b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0
}

.body
{ .mmb
      nop.m 999
      nop.m 999
(p10) br.cond.sptk.many  call_tanl ;;
}

// Here if we should call cotl (p10=0, p11=1)
{ .mmb
      nop.m 999
      nop.m 999
      br.call.sptk.many  b0=__libm_cotl# ;;
}

{ .mfi
      mov gp        = GR_SAVE_GP
      fnorm.s.s0 f8 = f8
      mov b0        = GR_SAVE_B0
}
;;

{ .mib
      nop.m 999
      mov ar.pfs    = GR_SAVE_PFS
      br.ret.sptk     b0
;;
}

// Here if we should call tanl (p10=1, p11=0)
call_tanl:
{ .mmb
      nop.m 999
      nop.m 999
      br.call.sptk.many  b0=__libm_tanl# ;;
}

{ .mfi
      mov gp        = GR_SAVE_GP
      fnorm.s.s0 f8 = f8
      mov b0        = GR_SAVE_B0
}
;;

{ .mib
      nop.m 999
      mov ar.pfs    = GR_SAVE_PFS
      br.ret.sptk     b0
;;
}

LOCAL_LIBM_END(__libm_callout)

.type __libm_tanl#,@function
.global __libm_tanl#
.type __libm_cotl#,@function
.global __libm_cotl#


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

// (1)
{ .mfi
      add           GR_Parameter_Y=-32,sp        // Parameter 2 value
      nop.f         0
.save   ar.pfs,GR_SAVE_PFS
      mov           GR_SAVE_PFS=ar.pfs           // Save ar.pfs
}
{ .mfi
.fframe 64
      add sp=-64,sp                              // Create new stack
      nop.f 0
      mov GR_SAVE_GP=gp                          // Save gp
};;

// (2)
{ .mmi
      stfs [GR_Parameter_Y] = f1,16              // STORE Parameter 2 on stack
      add GR_Parameter_X = 16,sp                 // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                          // Save b0
};;

.body
// (3)
{ .mib
      stfs [GR_Parameter_X] = f10                // STORE Parameter 1 on stack
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
      nop.b 0
}
{ .mib
      stfs [GR_Parameter_Y] = f8                 // STORE Parameter 3 on stack
      add   GR_Parameter_Y = -16,GR_Parameter_Y
      br.call.sptk b0=__libm_error_support#      // Call error handling function
};;
{ .mmi
      nop.m 0
      nop.m 0
      add   GR_Parameter_RESULT = 48,sp
};;

// (4)
{ .mmi
      ldfs  f8 = [GR_Parameter_RESULT]           // Get return result off stack
.restore sp
      add   sp = 64,sp                           // Restore stack pointer
      mov   b0 = GR_SAVE_B0                      // Restore return address
};;
{ .mib
      mov   gp = GR_SAVE_GP                      // Restore gp
      mov   ar.pfs = GR_SAVE_PFS                 // Restore ar.pfs
      br.ret.sptk     b0                         // Return
};;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
