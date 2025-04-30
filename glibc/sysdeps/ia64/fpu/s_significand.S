.file "significand.s"


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
// 04/04/00 Unwind support added
// 05/31/00 Fixed bug when x a double-extended denormal
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// double significand(double x)
//
// Overview of operation
//==============================================================
// If x = sig * 2**n with 1 <= sig < 2
// significand returns sig
//
// predicate registers used:
// p6, p7
//
// floating-point registers used:
// f8, f9, f10

.section .text
GLOBAL_LIBM_ENTRY(significand)

// qnan snan inf norm     unorm 0 -+
// 1    1    1   0        0     1 11

// f10 gets f8(sign) with f1(exp,significand)
{ .mfi
      nop.m 999
      fmerge.s       f10 = f8,f1
      nop.i 999
}
{ .mfi
      nop.m 999
      fnorm.s0          f9  = f8
      nop.i 999 ;;
}

// Test for denormal input
{ .mfi
      nop.m 999
      fclass.m.unc   p7,p0 = f8, 0x0b
      nop.i 999 ;;
}

// p6 = TRUE ==> x is not (nan,inf,0)
//               return sign(f8) exp(f1) significand(f8)
// else          x is (nan,inf,0)
//               return sign(f8) exp(f8) significand(f8), normalized.
{ .mfi
      nop.m 999
      fclass.m.unc   p0,p6 = f8, 0xe7
      nop.i 999 ;;
}

{ .mmb
      nop.m 999
      nop.m 999
(p7)  br.cond.spnt SIGNIFICAND_DENORM ;; // Branch if x denormal
}

{ .mfi
      nop.m 999
(p6)  fmerge.se      f8 = f10,f8
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
      fnorm.d.s0        f8 = f8
      br.ret.sptk    b0 ;;
}

SIGNIFICAND_DENORM:
// Here if x denorm
{ .mfi
      nop.m 999
      fmerge.se      f8 = f10,f9
      nop.i 999 ;;
}

// Check if fnorm(x) still denormal, means x double-extended denormal
{ .mfi
      nop.m 999
      fclass.m.unc   p7,p0 = f9, 0x0b
      nop.i 999 ;;
}

// This will be the final result unless x double-extended denormal
{ .mfi
      nop.m 999
      fnorm.d.s0        f8 = f8
      nop.i 999 ;;
}

// If x double-extended denorm, then significand ok, but must merge in
//    correct signexp
{ .mfi
      nop.m 999
(p7)  fmerge.se      f8 = f10,f8
      nop.i 999 ;;
}

// Final normalization if x double-extended denorm
{ .mfb
      nop.m 999
(p7)  fnorm.d.s0        f8 = f8
      br.ret.sptk    b0 ;;
}

GLOBAL_LIBM_END(significand)
