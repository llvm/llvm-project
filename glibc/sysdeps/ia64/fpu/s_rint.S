.file "rint.s"


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
// 02/08/01 Corrected behavior for all rounding modes.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 01/20/03 Improved performance
//==============================================================

// API
//==============================================================
// double rint(double x)
//==============================================================

// general input registers:
// r14 - r21

rSignexp   = r14
rExp       = r15
rExpMask   = r16
rBigexp    = r17
rM1        = r18
rFpsr      = r19
rRcs0      = r20
rRcs0Mask  = r21

// floating-point registers:
// f8 - f11

fXInt      = f9
fNormX     = f10
fTmp       = f11

// predicate registers used:
// p6 - p10

// Overview of operation
//==============================================================
// double rint(double x)
// Return an integer value (represented as a double) that is x
// rounded to integer in current rounding mode
// Inexact is set if x != rint(x)
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
GLOBAL_IEEE754_ENTRY(rint)

{ .mfi
      getf.exp         rSignexp  = f8        // Get signexp, recompute if unorm
      fclass.m         p7,p0 = f8, 0x0b      // Test x unorm
      addl             rBigexp = 0x10033, r0 // Set exponent at which is integer
}
{ .mfi
      mov              rM1 = -1              // Set all ones
      fcvt.fx.s1       fXInt  = f8           // Convert to int in significand
      mov              rExpMask    = 0x1FFFF // Form exponent mask
}
;;

{ .mfi
      mov              rFpsr = ar40          // Read fpsr -- check rc.s0
      fclass.m         p6,p0 = f8, 0x1e3     // Test x natval, nan, inf
      nop.i            0
}
{ .mfb
      setf.sig         fTmp = rM1            // Make const for setting inexact
      fnorm.s1         fNormX  = f8          // Normalize input
(p7)  br.cond.spnt     RINT_UNORM            // Branch if x unorm
}
;;


RINT_COMMON:
// Return here from RINT_UNORM
{ .mfb
      and              rExp = rSignexp, rExpMask // Get biased exponent
(p6)  fma.d.s0         f8 = f8, f1, f0       // Result if x natval, nan, inf
(p6)  br.ret.spnt      b0                    // Exit if x natval, nan, inf
}
;;

{ .mfi
      mov              rRcs0Mask = 0x0c00     // Mask for rc.s0
      fcvt.xf          f8 = fXInt             // Result assume |x| < 2^52
      cmp.ge           p7,p8 = rExp, rBigexp  // Is |x| >= 2^52?
}
;;

// We must correct result if |x| >= 2^52
{ .mfi
      nop.m            0
(p7)  fma.d.s0         f8 = fNormX, f1, f0    // If |x| >= 2^52, result x
      nop.i            0
}
;;

{ .mfi
      nop.m            0
      fcmp.eq.unc.s1   p0, p9 = f8, fNormX    // Is result = x ?
      nop.i            0
}
{ .mfi
      nop.m            0
(p8)  fmerge.s         f8 = fNormX, f8        // Make sure sign rint(x) = sign x
      nop.i            0
}
;;

{ .mfi
(p8)  and              rRcs0 = rFpsr, rRcs0Mask // Get rounding mode for sf0
      nop.f            0
      nop.i            0
}
;;

// If |x| < 2^52 we must test for other rounding modes
{ .mfi
(p8)  cmp.ne.unc       p10,p0 = rRcs0, r0     // Test for other rounding modes
(p9)  fmpy.s0          fTmp = fTmp, fTmp      // Dummy to set inexact
      nop.i            0
}
{ .mbb
      nop.m            0
(p10) br.cond.spnt     RINT_NOT_ROUND_NEAREST // Branch if not round nearest
      br.ret.sptk      b0                     // Exit main path if round nearest
}
;;



RINT_UNORM:
// Here if x unorm
{ .mfb
      getf.exp         rSignexp  = fNormX     // Get signexp, recompute if unorm
      fcmp.eq.s0       p7,p0 = f8, f0         // Dummy op to set denormal flag
      br.cond.sptk     RINT_COMMON            // Return to main path
}
;;

RINT_NOT_ROUND_NEAREST:
// Here if not round to nearest, and |x| < 2^52
// Set rounding mode of s2 to that of s0, and repeat the conversion using s2
{ .mfi
      nop.m            0
      fsetc.s2         0x7f, 0x40
      nop.i            0
}
;;

{ .mfi
      nop.m            0
      fcvt.fx.s2       fXInt  = fNormX        // Convert to int in significand
      nop.i            0
}
;;

{ .mfi
      nop.m            0
      fcvt.xf          f8 = fXInt             // Expected result
      nop.i            0
}
;;

// Be sure sign of result = sign of input.  Fixes cases where result is 0.
{ .mfb
      nop.m            0
      fmerge.s         f8 = fNormX, f8
      br.ret.sptk      b0                     // Exit main path
}
;;

GLOBAL_IEEE754_END(rint)
libm_alias_double_other (__rint, rint)
