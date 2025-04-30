.file "ceilf.s"


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
// 06/13/00 Improved speed
// 06/27/00 Eliminated incorrect invalid flag setting
// 05/20/02 Cleaned up namespace and sf0 syntax
// 01/28/03 Improved performance
//==============================================================

// API
//==============================================================
// float ceilf(float x)
//==============================================================

// general input registers:
// r14 - r19

rSignexp   = r14
rExp       = r15
rExpMask   = r16
rBigexp    = r17
rM1        = r18
rSignexpM1 = r19

// floating-point registers:
// f8 - f13

fXInt      = f9
fNormX     = f10
fTmp       = f11
fAdj       = f12
fPreResult = f13

// predicate registers used:
// p6 - p10

// Overview of operation
//==============================================================
// float ceilf(float x)
// Return an integer value (represented as a float) that is the smallest
// value not less than x
// This is x rounded toward +infinity to an integral value.
// Inexact is set if x != ceilf(x)
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
GLOBAL_LIBM_ENTRY(ceilf)

{ .mfi
      getf.exp         rSignexp  = f8        // Get signexp, recompute if unorm
      fclass.m         p7,p0 = f8, 0x0b      // Test x unorm
      addl             rBigexp = 0x10016, r0 // Set exponent at which is integer
}
{ .mfi
      mov              rM1 = -1              // Set all ones
      fcvt.fx.trunc.s1 fXInt  = f8           // Convert to int in significand
      mov              rExpMask    = 0x1FFFF // Form exponent mask
}
;;

{ .mfi
      mov              rSignexpM1  = 0x2FFFF // Form signexp of -1
      fcmp.lt.s1       p8,p9 = f8, f0        // Test x < 0
      nop.i            0
}
{ .mfb
      setf.sig         fTmp = rM1            // Make const for setting inexact
      fnorm.s1         fNormX  = f8          // Normalize input
(p7)  br.cond.spnt     CEIL_UNORM            // Branch if x unorm
}
;;

CEIL_COMMON:
// Return here from CEIL_UNORM
{ .mfi
      nop.m            0
      fclass.m         p6,p0 = f8, 0x1e7     // Test x natval, nan, inf, 0
      nop.i            0
}
;;

.pred.rel "mutex",p8,p9
{ .mfi
      nop.m            0
(p8)  fma.s1           fAdj = f0, f0, f0     // If x < 0, adjustment is 0
      nop.i            0
}
{ .mfi
      nop.m            0
(p9)  fma.s1           fAdj = f1, f1, f0     // If x > 0, adjustment is +1
      nop.i            0
}
;;

{ .mfi
      nop.m            0
      fcvt.xf          fPreResult = fXInt    // trunc(x)
      nop.i            0
}
{ .mfb
      nop.m            0
(p6)  fma.s.s0         f8 = f8, f1, f0       // Result if x natval, nan, inf, 0
(p6)  br.ret.spnt      b0                    // Exit if x natval, nan, inf, 0
}
;;

{ .mmi
      and              rExp = rSignexp, rExpMask // Get biased exponent
;;
      cmp.ge           p7,p6 = rExp, rBigexp  // Is |x| >= 2^23?
(p8)  cmp.lt.unc       p10,p0 = rSignexp, rSignexpM1 // Is -1 < x < 0?
}
;;

// If -1 < x < 0, we turn off p6 and compute result as -0
{ .mfi
(p10) cmp.ne           p6,p0 = r0,r0
(p10) fmerge.s         f8 = fNormX, f0
      nop.i            0
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      nop.m            0
(p6)  fma.s.s0         f8 = fPreResult, f1, fAdj // Result if !int, |x| < 2^23
      nop.i            0
}
{ .mfi
      nop.m            0
(p7)  fma.s.s0         f8 = fNormX, f1, f0    // Result, if |x| >= 2^23
(p10) cmp.eq           p6,p0 = r0,r0          // If -1 < x < 0, turn on p6 again
}
;;

{ .mfi
      nop.m            0
(p6)  fcmp.eq.unc.s1   p8, p9 = fPreResult, fNormX // Is trunc(x) = x ?
      nop.i            0
}
;;

{ .mfi
      nop.m            0
(p9)  fmpy.s0          fTmp = fTmp, fTmp      // Dummy to set inexact
      nop.i            0
}
{ .mfb
      nop.m            0
(p8)  fma.s.s0         f8 = fNormX, f1, f0    // If x int, result normalized x
      br.ret.sptk      b0                     // Exit main path, 0 < |x| < 2^23
}
;;


CEIL_UNORM:
// Here if x unorm
{ .mfb
      getf.exp         rSignexp  = fNormX     // Get signexp, recompute if unorm
      fcmp.eq.s0       p7,p0 = f8, f0         // Dummy op to set denormal flag
      br.cond.sptk     CEIL_COMMON            // Return to main path
}
;;

GLOBAL_LIBM_END(ceilf)
libm_alias_float_other (ceil, ceil)
