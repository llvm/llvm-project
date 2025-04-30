.file "fmaxf.s"


// Copyright (c) 2001 - 2003, Intel Corporation
// All rights reserved.
//
// Contributed 2001 by the Intel Numerics Group, Intel Corporation
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
// 05/31/01 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// float fmaxf  (float x, float y)
//
// Overview of operation
//==============================================================
// returns the algebraic maximum of 2 input values
//
// Special cases:
//    fmaxf(x, nan) returns x if x is numeric  // Must special case this one
//    fmaxf(nan, y) returns y if y is numeric
//    fmaxf(nan1, nan2) returns quietized nan2
//    fmaxf(+0,+0) returns +0
//    fmaxf(-0,+0) returns +0
//    fmaxf(-0,-0) returns -0
//    fmaxf(+0,-0) returns +0                  // Must special case this one
//
// SNaN causes invalid to be set

// floating-point registers used: 2
// f8,  input x, output
// f9,  input y

.section .text
GLOBAL_LIBM_ENTRY(fmaxf)

{ .mfi
      nop.m 999
      fcmp.unord.s0 p6,p7 = f8, f9 // Is x or y a nan? Raise invalid or denormal
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p7)  fclass.m.unc  p8,p9 = f9, 0x06 // If no nan, is y=-0?
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc  p10,p0 = f8, 0xc3 // Is x nan?
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fmax.s0  f8 = f8, f9    // Normal case, no nan and y not -0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fmax.s0  f8 = f9, f8    // No nan and y -0
      nop.i 999
}
;;

{ .mfb
      nop.m 999
(p10) fmerge.s  f8 = f9, f9   // If x nan, return y, else do nothing (returns x)
      br.ret.sptk    b0
}
;;

GLOBAL_LIBM_END(fmaxf)
libm_alias_float_other (fmax, fmax)
