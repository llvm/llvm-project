.file "truncl.s"


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
// 07/07/00 Created
// 05/20/02 Cleaned up namespace and sf0 syntax
// 01/20/03 Improved performance and reduced code size
//==============================================================

// API
//==============================================================
// long double truncl(long double x)
//==============================================================

// general input registers:
// r14 - r18

rExpBias   = r14
rSignexp   = r15
rExp       = r16
rExpMask   = r17
rBigexp    = r18

// floating-point registers:
// f8 - f10

fXtruncInt = f9
fNormX     = f10

// predicate registers used:
// p6, p7

// Overview of operation
//==============================================================
// long double truncl(long double x)
// Return an integer value (represented as a long double) less than or
// equal to x in magnitude.
// This is x rounded toward zero to an integral value.
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
GLOBAL_LIBM_ENTRY(truncl)

{ .mfi
      getf.exp         rSignexp  = f8        // Get signexp, recompute if unorm
      fcvt.fx.trunc.s1 fXtruncInt  = f8      // Convert to int in significand
      addl             rBigexp = 0x1003e, r0 // Set exponent at which is integer
}
{ .mfi
      mov              rExpBias    = 0x0FFFF // Form exponent bias
      fnorm.s1         fNormX  = f8          // Normalize input
      mov              rExpMask    = 0x1FFFF // Form exponent mask
}
;;

{ .mfi
      nop.m            0
      fclass.m         p7,p0 = f8, 0x0b      // Test x unorm
      nop.i            0
}
;;

{ .mfb
      nop.m            0
      fclass.m         p6,p0 = f8, 0x1e3     // Test x natval, nan, inf
(p7)  br.cond.spnt     TRUNC_UNORM           // Branch if x unorm
}
;;

TRUNC_COMMON:
// Return here from TRUNC_UNORM
{ .mfb
      and              rExp = rSignexp, rExpMask // Get biased exponent
(p6)  fma.s0           f8 = f8, f1, f0       // Result if x natval, nan, inf
(p6)  br.ret.spnt      b0                    // Exit if x natval, nan, inf
}
;;

{ .mfi
      cmp.lt           p6,p0 = rExp, rExpBias // Is |x| < 1?
      fcvt.xf          f8 = fXtruncInt        // Result, assume 1 <= |x| < 2^63
      cmp.ge           p7,p0 = rExp, rBigexp  // Is |x| >= 2^63?
}
;;

// We must correct result if |x| < 1, or |x| >= 2^63
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m            0
(p6)  fmerge.s         f8 = fNormX, f0        // If |x| < 1, result sgn(x)*0
      nop.i            0
}
{ .mfb
      nop.m            0
(p7)  fma.s0           f8 = fNormX, f1, f0    // If |x| >= 2^63, result x
      br.ret.sptk      b0                     // Exit main path
}
;;


TRUNC_UNORM:
// Here if x unorm
{ .mfb
      getf.exp         rSignexp  = fNormX     // Get signexp, recompute if unorm
      fcmp.eq.s0       p7,p0 = f8, f0         // Dummy op to set denormal flag
      br.cond.sptk     TRUNC_COMMON           // Return to main path
}
;;

GLOBAL_LIBM_END(truncl)
libm_alias_ldouble_other (trunc, trunc)
