.file "round.s"


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
// 10/25/00 Initial version
// 06/14/01 Changed cmp to an equivalent form
// 05/20/02 Cleaned up namespace and sf0 syntax
// 01/20/03 Improved performance and reduced code size
// 04/18/03 Eliminate possible WAW dependency warning
// 09/03/03 Improved performance
//==============================================================

// API
//==============================================================
// double round(double x)
//==============================================================

// general input registers:
// r14 - r18

rSignexp   = r14
rExp       = r15
rExpMask   = r16
rBigexp    = r17
rExpHalf   = r18

// floating-point registers:
// f8 - f13

fXtruncInt = f9
fNormX     = f10
fHalf      = f11
fInc       = f12
fRem       = f13

// predicate registers used:
// p6 - p10

// Overview of operation
//==============================================================
// double round(double x)
// Return an integer value (represented as a double) that is x
// rounded to nearest integer, halfway cases rounded away from
// zero.
//  if x>0   result = trunc(x+0.5)
//  if x<0   result = trunc(x-0.5)
//
//==============================================================

// double_extended
// if the exponent is > 1003e => 3F(true) = 63(decimal)
// we have a significand of 64 bits 1.63-bits.
// If we multiply by 2^63, we no longer have a fractional part
// So input is an integer value already.

// double
// if the exponent is >= 10033 => 34(true) = 52(decimal)
// 34 + 3ff = 433
// we have a significand of 53 bits 1.52-bits. (implicit 1)
// If we multiply by 2^52, we no longer have a fractional part
// So input is an integer value already.

// single
// if the exponent is > 10016 => 17(true) = 23(decimal)
// we have a significand of 24 bits 1.23-bits. (implicit 1)
// If we multiply by 2^23, we no longer have a fractional part
// So input is an integer value already.


.section .text
GLOBAL_LIBM_ENTRY(round)

{ .mfi
      getf.exp         rSignexp  = f8        // Get signexp, recompute if unorm
      fcvt.fx.trunc.s1 fXtruncInt  = f8      // Convert to int in significand
      addl             rBigexp = 0x10033, r0 // Set exponent at which is integer
}
{ .mfi
      mov              rExpHalf    = 0x0FFFE // Form sign and exponent of 0.5
      fnorm.s1         fNormX  = f8          // Normalize input
      mov              rExpMask    = 0x1FFFF // Form exponent mask
}
;;

{ .mfi
      setf.exp         fHalf = rExpHalf      // Form 0.5
      fclass.m         p7,p0 = f8, 0x0b      // Test x unorm
      nop.i            0
}
;;

{ .mfb
      nop.m            0
      fclass.m         p6,p0 = f8, 0x1e3     // Test x natval, nan, inf
(p7)  br.cond.spnt     ROUND_UNORM           // Branch if x unorm
}
;;

ROUND_COMMON:
// Return here from ROUND_UNORM
{ .mfb
      nop.m            0
      fcmp.lt.s1       p8,p9 = f8, f0        // Test if x < 0
(p6)  br.cond.spnt     ROUND_SPECIAL         // Exit if x natval, nan, inf
}
;;

{ .mfi
      nop.m            0
      fcvt.xf          f8 = fXtruncInt        // Pre-Result if 0.5 <= |x| < 2^52
      nop.i            0
}
;;

{ .mfi
      and              rExp = rSignexp, rExpMask // Get biased exponent
      fmerge.s         fInc = fNormX, f1      // Form increment if |rem| >= 0.5
      nop.i            0
}
;;

{ .mmi
      cmp.lt           p6,p0 = rExp, rExpHalf // Is |x| < 0.5?
      cmp.ge           p7,p0 = rExp, rBigexp  // Is |x| >= 2^52?
      cmp.lt           p10,p0 = rExp, rExpHalf // Is |x| < 0.5?
}
;;

// We must correct result if |x| < 0.5, or |x| >= 2^52
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m            0
(p6)  fmerge.s         f8 = fNormX, f0        // If |x| < 0.5, result sgn(x)*0
      nop.i            0
}
{ .mfb
(p7)  cmp.eq           p10,p0 = r0, r0        // Also turn on p10 if |x| >= 2^52
(p7)  fma.d.s0         f8 = fNormX, f1, f0    // If |x| >= 2^52, result x
(p10) br.ret.spnt      b0                     // Exit |x| < 0.5 or |x| >= 2^52
}
;;

// Here if 0.5 <= |x| < 2^52
{ .mfi
      nop.m            0
(p9)  fms.s1           fRem = fNormX, f1, f8  // Get remainder = x - trunc(x)
      nop.i            0
}
{ .mfi
      nop.m            0
(p8)  fms.s1           fRem = f8, f1, fNormX  // Get remainder = trunc(x) - x
      nop.i            0
}
;;

{ .mfi
      nop.m            0
      fcmp.ge.s1       p9,p0 = fRem, fHalf    // Test |rem| >= 0.5
      nop.i            0
}
;;

// If x < 0 and remainder <= -0.5, then subtract 1 from result
// If x > 0 and remainder >= +0.5, then add 1 to result
{ .mfb
      nop.m            0
(p9)  fma.d.s0         f8 = f8, f1, fInc
      br.ret.sptk      b0
}
;;


ROUND_SPECIAL:
// Here if x natval, nan, inf
{ .mfb
      nop.m            0
      fma.d.s0         f8 = f8, f1, f0
      br.ret.sptk      b0
}
;;

ROUND_UNORM:
// Here if x unorm
{ .mfi
      getf.exp         rSignexp  = fNormX     // Get signexp, recompute if unorm
      fcmp.eq.s0       p7,p0 = f8, f0         // Dummy op to set denormal flag
      nop.i            0
}
{ .mfb
      nop.m            0
      fcvt.fx.trunc.s1 fXtruncInt  = fNormX   // Convert to int in significand
      br.cond.sptk     ROUND_COMMON           // Return to main path
}
;;

GLOBAL_LIBM_END(round)
libm_alias_double_other (round, round)
