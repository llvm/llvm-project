.file "modff.s"


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
// History
//==============================================================
// 02/02/00 Initial version
// 04/04/00 Improved speed, corrected result for NaN input
// 12/22/00 Fixed so inexact flag is never set, and invalid is not set for
//            qnans nor for inputs larger than 2^63.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// float modff(float x, float *iptr)
// break a floating point x number into fraction and an exponent
//
// input  floating point f8, address in r33
// output floating point f8 (x fraction), and *iptr (x integral part)
//
// OVERVIEW
//==============================================================

// NO FRACTIONAL PART: HUGE
// If
// for double-extended
// If the true exponent is greater than or equal 63
//      1003e ==> 1003e -ffff = 3f = 63(dec)
// for double
// If the true exponent is greater than or equal 52
//                10033 -ffff = 34 = 52(dec)
// for single
// If the true exponent is greater than or equal 23
//                10016 -ffff = 17 = 23(dec)
// then
// we are already an integer (p9 true)

// NO INTEGER PART:    SMALL
//     Is f8 exponent less than register bias (that is, is it
//     less than 1). If it is, get the right sign of
//     zero and store this in iptr.

// CALCULATION: NOT HUGE, NOT SMALL
// To get the integer part
// Take the floating-point  input and truncate
//   then convert  this integer to fp  Call it  MODF_INTEGER_PART

// Subtract  MODF_INTEGER_PART from MODF_NORM_F8 to get fraction part
// Then put fraction part in f8
//      put integer  part MODF_INTEGER_PART into *iptr

// Registers used
//==============================================================

// predicate registers used:
// p6 - p13

//                      0xFFFF           0x10016
// -----------------------+-----------------+-------------
//              SMALL     |      NORMAL     | HUGE
//    p11 --------------->|<----- p12 ----->| <-------------- p9
//    p10 --------------------------------->|
//    p13 --------------------------------------------------->|
//

// floating-point registers used:
MODF_NORM_F8               = f9
MODF_FRACTION_PART         = f10
MODF_INTEGER_PART          = f11
MODF_INT_INTEGER_PART      = f12


// general registers used
modf_signexp    = r14
modf_GR_no_frac = r15
modf_GR_FFFF    = r16
modf_17_ones    = r17
modf_exp        = r18
// r33 = iptr


.section .text
GLOBAL_LIBM_ENTRY(modff)

// Main path is p9, p11, p8 FALSE and p12 TRUE

// Assume input is normalized and get signexp
// Normalize input just in case
// Form exponent bias
{ .mfi
      getf.exp  modf_signexp = f8
      fnorm.s0          MODF_NORM_F8  = f8
      addl           modf_GR_FFFF  = 0xffff, r0
}
// Get integer part of input
// Form exponent mask
{ .mfi
      nop.m 999
      fcvt.fx.trunc.s1  MODF_INT_INTEGER_PART   = f8
      mov  modf_17_ones     = 0x1ffff ;;
}

// Is x nan or inf?
// qnan snan inf norm     unorm 0 -+
// 1    1    1   0        0     0 11 = 0xe3 NAN_INF
// Form biased exponent where input only has an integer part
{ .mfi
      nop.m 999
      fclass.m.unc p6,p13 = f8, 0xe3
      addl modf_GR_no_frac = 0x10016, r0 ;;
}

// Mask to get exponent
// Is x unnorm?
// qnan snan inf norm     unorm 0 -+
// 0    0    0   0        1     0 11 = 0x0b UNORM
// Set p13 to indicate calculation path, else p6 if nan or inf
{ .mfi
      and       modf_exp = modf_17_ones, modf_signexp
      fclass.m.unc p8,p0 = f8, 0x0b
      nop.i 999 ;;
}

// p11 <== SMALL, no integer part, fraction is everyting
// p9  <== HUGE,  no fraction part, integer is everything
// p12 <== NORMAL, fraction part and integer part
{ .mii
(p13) cmp.lt.unc p11,p10 = modf_exp, modf_GR_FFFF
      nop.i 999
      nop.i 999 ;;
}

// Is x inf? p6 if inf, p7 if nan
{ .mfb
(p10) cmp.ge.unc p9,p12  = modf_exp, modf_GR_no_frac
(p6)  fclass.m.unc p6,p7 = f8, 0x23
(p8)  br.cond.spnt MODF_DENORM ;;
}

MODF_COMMON:
// For HUGE set fraction to signed 0
{ .mfi
      nop.m 999
(p9)  fmerge.s f8 = f8,f0
      nop.i 999
}
// For HUGE set integer part to normalized input
{ .mfi
      nop.m 999
(p9)  fnorm.s.s0 MODF_INTEGER_PART = MODF_NORM_F8
      nop.i 999 ;;
}

// For SMALL set fraction to normalized input, integer part to signed 0
{ .mfi
      nop.m 999
(p11) fmerge.s MODF_INTEGER_PART = f8,f0
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fnorm.s.s0 f8 = MODF_NORM_F8
      nop.i 999 ;;
}

// For NORMAL float the integer part
{ .mfi
      nop.m 999
(p12) fcvt.xf    MODF_INTEGER_PART = MODF_INT_INTEGER_PART
      nop.i 999 ;;
}

// If x inf set integer part to INF, fraction to signed 0
{ .mfi
(p6)  stfs [r33] = MODF_NORM_F8
(p6)  fmerge.s  f8 = f8,f0
      nop.i 999 ;;
}

// If x nan set integer and fraction parts to NaN (quietized)
{ .mfi
(p7)  stfs [r33] = MODF_NORM_F8
(p7)  fmerge.s  f8 = MODF_NORM_F8, MODF_NORM_F8
      nop.i 999 ;;
}

{ .mmi
(p9)  stfs [r33] = MODF_INTEGER_PART
      nop.m 999
      nop.i 999 ;;
}

// For NORMAL compute fraction part
{ .mfi
(p11) stfs [r33] = MODF_INTEGER_PART
(p12) fms.s.s0   f8 = MODF_NORM_F8,f1, MODF_INTEGER_PART
      nop.i 999 ;;
}

// For NORMAL test if fraction part is zero; if so append correct sign
{ .mfi
      nop.m 999
(p12) fcmp.eq.unc.s0 p7,p0 = MODF_NORM_F8, MODF_INTEGER_PART
      nop.i 999 ;;
}

{ .mfi
(p12) stfs [r33] = MODF_INTEGER_PART
      nop.f 999
      nop.i 999 ;;
}

// For NORMAL if fraction part is zero append sign of input
{ .mfb
      nop.m 999
(p7)  fmerge.s f8 = MODF_NORM_F8, f0
      br.ret.sptk    b0 ;;
}

MODF_DENORM:
// If x unorm get signexp from normalized input
// If x unorm get integer part from normalized input
{ .mfi
      getf.exp  modf_signexp = MODF_NORM_F8
      fcvt.fx.trunc.s1  MODF_INT_INTEGER_PART   = MODF_NORM_F8
      nop.i 999 ;;
}

// If x unorm mask to get exponent
{ .mmi
      and       modf_exp = modf_17_ones, modf_signexp ;;
      cmp.lt.unc p11,p10 = modf_exp, modf_GR_FFFF
      nop.i 999 ;;
}

{ .mfb
(p10) cmp.ge.unc p9,p12  = modf_exp, modf_GR_no_frac
      nop.f 999
      br.cond.spnt MODF_COMMON ;;
}

GLOBAL_LIBM_END(modff)
libm_alias_float_other (modf, modf)
