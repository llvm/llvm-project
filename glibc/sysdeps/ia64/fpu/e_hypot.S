.file "hypot.s"


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
//
//*********************************************************************
//
// History:
// 02/02/00 hand-optimized
// 04/04/00 Unwind support added
// 06/20/00 new version
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 04/17/03 Added missing mutex directive
//
//*********************************************************************
//                           ___________
// Function:   hypot(x,y) = |(x^2 + y^2) = for double precision values
//             x and y
//             Also provides cabs functionality.
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8  (Input and Return Value)
//                              f9  (Input)
//                              f6 -f15, f32-f34
//
//    General Purpose Registers:
//      r2,r3,r29 (Scratch)
//      r32-r36 (Locals)
//      r37-r40 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6 - p10
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    All faults and exceptions should be raised correctly.
//    Overflow can occur.
//    hypot(Infinity and anything) = +Infinity
//    hypot(QNaN and anything) = QNaN
//    hypot(SNaN and anything ) = QNaN
//
//*********************************************************************
//
// Implementation:
//  x2 = x * x   in double-extended
//  y2 = y * y   in double-extended
//  temp = x2 + y2   in double-extended
//  sqrt(temp) rounded to double
//
//*********************************************************************

GR_SAVE_PFS         = r33
GR_SAVE_B0          = r34
GR_SAVE_GP          = r35
GR_Parameter_X      = r36
GR_Parameter_Y      = r37
GR_Parameter_RESULT = r38
GR_Parameter_TAG    = r39

FR_X                = f32
FR_Y                = f33
FR_RESULT           = f8

.section .text

LOCAL_LIBM_ENTRY(cabs)
LOCAL_LIBM_END(cabs)

GLOBAL_IEEE754_ENTRY(hypot)

{.mfi
  alloc r32= ar.pfs,0,4,4,0
  // Compute x*x
  fma.s1 f10=f8,f8,f0
  // r2=bias-1
  mov r2=0xfffe
}
{.mfi
  // 63/8
  mov r3=0x40fc //0000
  // y*y
  fma.s1 f11=f9,f9,f0
  // r29=429/16
  mov r29=0x41d68;; //000
}

{ .mfi
     nop.m 0
//   Check if x is an Inf - if so return Inf even
//   if y is a NaN (C9X)
     fclass.m.unc p7, p6 = f8, 0x023
     shl r3=r3,16
}
{.mfi
	nop.m 0
  // if possible overflow, copy f8 to f32
  // set Denormal, if necessary
  // (p8)
  fma.d.s0 f32=f8,f1,f0
  nop.i 0;;
}
{ .mfi
     nop.m 0
//   Check if y is an Inf - if so return Inf even
//   if x is a NaN (C9X)
     fclass.m.unc p8, p9 = f9, 0x023
     shl r29=r29,12
}
{ .mfb
	 // f7=0.5
	 setf.exp f7=r2
//   For x=inf, multiply y by 1 to raise invalid on y an SNaN
//   (p7) fma.s0 f9=f9,f1,f0
     // copy f9 to f33; set Denormal, if necessary
	 fma.d.s0 f33=f9,f1,f0
     nop.b 0;;
}
{.mfb
  // f13=63/8
  setf.s f13=r3
  // is y Zero ?
  (p6) fclass.m p6,p0=f9,0x7
  nop.b 0
}
{.mlx
  nop.m 0
  movl r2=0x408c0000;;
}

{.mfi
  // f34=429/16
  setf.s f34=r29
  // is x Zero ?
  (p9) fclass.m p9,p0=f8,0x7
  // 231/16
  mov r3=0x4167;; //0000
}
{.mfi
  nop.m 0
  // a=x2+y2
  fma.s1 f12=f10,f1,f11
  nop.i 0;;
}
{.mfi
  nop.m 0
  // y not NaN ?
  (p9) fclass.m p8,p0=f9,0x3f
  shl r3=r3,16
}
{.mfi
  nop.m 0
  // f6=2
  fma.s1 f6=f1,f1,f1
  nop.i 0;;
}


{.mfi
  nop.m 0
  // x not NaN ?
  (p6) fclass.m p7,p0=f8,0x3f
  nop.i 0;;
}
{.mfi
  // f9=35/8
  setf.s f9=r2
  nop.f 0
  // 2*emax-2
  mov r2=0x107fb;;
}

.pred.rel "mutex",p7,p8
{.mfb
  nop.m 0
  // if f8=Infinity or f9=Zero, return |f8|
  (p7) fmerge.s f8=f0,f32
  (p7) br.ret.spnt b0
}
{.mfb
  nop.m 0
  // if f9=Infinity or f8=Zero, return |f9|
  (p8) fmerge.s f8=f0,f33
  (p8) br.ret.spnt b0;;
}


{.mfi
  // f10 =231/16
  setf.s f10=r3
  // z0=frsqrta(a)
  frsqrta.s1 f8,p6=f12
  nop.i 0;;
}

{ .mfi
	 nop.m 0
//   Identify Natvals, Infs, NaNs, and Zeros
//   and return result
     fclass.m.unc p7, p0 = f12, 0x1E7
     nop.i 0;;
}
{.mfb
  // get exponent of x^2+y^2
  getf.exp r3=f12
  // if special case, set f8
  (p7) mov f8=f12
  (p7) br.ret.spnt b0;;
}


{.mfi
  nop.m 0
  // S0=a*z0
  (p6) fma.s1 f14=f12,f8,f0
  nop.i 0
}
{.mfi
  nop.m 0
  // H0=0.5*z0
  (p6) fma.s1 f15=f8,f7,f0
  nop.i 0;;
}


{.mfi
  nop.m 0
  // f6=5/2
  fma.s1 f6=f7,f1,f6
  nop.i 0
}
{.mfi
  nop.m 0
  // f11=3/2
  fma.s1 f11=f7,f1,f1
  nop.i 0;;
}

{.mfi
  nop.m 0
  // d=0.5-S0*H0
  (p6) fnma.s1 f7=f14,f15,f7
  nop.i 0;;
}

{.mfi
  nop.m 0
  // P67=231/16+429/16*d
  (p6) fma.s1 f10=f34,f7,f10
  nop.i 0
}
{.mfi
  nop.m 0
  // P45=63/8*d+35/8
  (p6) fma.s1 f9=f13,f7,f9
  nop.i 0;;
}
{.mfi
  nop.m 0
  // P23=5/2*d+3/2
  (p6) fma.s1 f11=f6,f7,f11
  nop.i 0
}
{.mfi
  nop.m 0
  // d2=d*d
  (p6) fma.s1 f13=f7,f7,f0
  nop.i 0;;
}

{.mfi
  nop.m 0
  // P47=d2*P67+P45
  (p6) fma.s1 f10=f10,f13,f9
  nop.i 0
}
{.mfi
  nop.m 0
  // P13=d*P23+1
  (p6) fma.s1 f11=f11,f7,f1
  nop.i 0;;
}
{.mfi
  nop.m 0
  // d3=d2*d
  (p6) fma.s1 f13=f13,f7,f0
  nop.i 0;;
}

{.mfi
  nop.m 0
  // T0=d*S0
  (p6) fma.s1 f15=f7,f14,f0
  nop.i 0
}
{.mfi
  // Is x^2 + y^2 well less than the overflow
  // threshold?
  (p6) cmp.lt.unc p7, p8 =  r3,r2
  // P=P13+d3*P47
  (p6) fma.s1 f10=f13,f10,f11
  nop.i 0;;
}

{.mfb
  nop.m 0
  // S=P*T0+S0
  fma.d.s0 f8=f10,f15,f14
  // No overflow in this case
  (p7) br.ret.sptk b0;;
}

{ .mfi
     nop.m 0
(p8) fsetc.s2 0x7F,0x42
     // Possible overflow path, must detect by
     // Setting widest range exponent with prevailing
     // rounding mode.
     nop.i 0 ;;
}


{ .mfi
   // bias+0x400 (bias+EMAX+1)
   (p8) mov r2=0x103ff
   // S=P*T0+S0
   (p8) fma.d.s2 f12=f10,f15,f14
   nop.i 0 ;;
}
{ .mfi
(p8) setf.exp f11 = r2
(p8) fsetc.s2 0x7F,0x40
//   Restore Original Mode in S2
     nop.i 0 ;;
}
{ .mfi
     nop.m 0
(p8) fcmp.lt.unc.s1 p9, p10 =  f12, f11
     nop.i 0 ;;
}
{ .mib
     nop.m 0
     mov GR_Parameter_TAG = 46
	 // No overflow
(p9) br.ret.sptk b0;;
}
GLOBAL_IEEE754_END(hypot)
libm_alias_double_other (__hypot, hypot)

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 0
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
        stfd [GR_Parameter_Y] = FR_Y,16         // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfd [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT      // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfd  f8 = [GR_Parameter_RESULT]       // Get return result off stack
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
        mov   b0 = GR_SAVE_B0                  // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
};;
LOCAL_LIBM_END(__libm_error_region#)
.type   __libm_error_support#,@function
.global __libm_error_support#
