.file "libm_frexp_4.s"

// Copyright (C) 2000, 2001, Intel Corporation
// All rights reserved.
//
// Contributed 2/2/2000 by John Harrison, Ted Kubaska, Bob Norin, Shane Story,
// and Ping Tak Peter Tang of the Computational Software Lab, Intel Corporation.
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
//
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
// http://developer.intel.com/opensource.
//
// History
//==============================================================
// 2/02/00: Initial version
// 3/20/00: Improved speed
// 6/01/00: Fixed bug when x a double-extended denormal
// 12/08/00 Corrected label on .endp
//
// API
//==============================================================
// double frexp(double x, int* y)
// double __libm_frexp_4(double x, int* y)
//   where int* y is a 32-bit integer
//
// Overview of operation
//==============================================================
// break a floating point x number into fraction and an exponent
// The fraction is returned as a double
// The exponent is returned as an integer pointed to by y
//    This is a true (not a biased exponent) but 0fffe is subtracted
//    as a bias instead of 0xffff. This is because the fraction returned
//    is between 0.5 and 1.0, not the expected IEEE range.
//
// The fraction is 0.5 <= fraction < 1.0
//
// Registers used
//==============================================================
//
// general registers:
// r14  exponent bias for x negative
// r15  exponent bias for x positive
// r16  signexp of x
// r17  exponent mask
// r18  exponent of x
// r19  exponent result
// r20  signexp of 2^64
// r32  on input contains the 64-bit IEEE double that is in f8
// r33  on input pointer to 32-bit integer for exponent
//
// predicate registers:
// p6   set if x is Nan, zero, or infinity
// p7   set if x negative
// p8   set if x positive
// p9   set if x double-extended denormal
//
// floating-point registers:
// f8  input, output
// f9  normalized x
// f10 signexp for significand result for x positive
// f11 signexp for significand result for x negative
// f12 2^64

#include "libm_support.h"

.align 32
.global __libm_frexp_4#

.section .text
.proc  __libm_frexp_4#
.align 32

__libm_frexp_4:

// Set signexp for significand result for x>0
// If x is a NaN, zero, or infinity, return it.
// Put 0 in the int pointer.
// x NAN, ZERO, INFINITY?
// Set signexp for significand result for x<0
{ .mfi
(p0)    mov         r15 = 0x0fffe
(p0)    fclass.m.unc p6,p0 = f8, 0xe7
(p0)    mov         r14 = 0x2fffe
}
// Form signexp of 2^64 in case x double-extended denormal
// Save the normalized value of input in f9
// The normalization also sets fault flags and takes faults if necessary
{ .mfi
(p0)    mov         r20 = 0x1003f
(p0)    fnorm       f9 = f8
        nop.i 999 ;;
}

// Move signexp for significand result for x>0 to FP reg
// Form 2^64 in case x double-extended denormal
{ .mmi
(p0)    setf.exp    f10 = r15
(p0)    setf.exp    f12 = r20
        nop.i 999 ;;
}

// Move signexp for significand result for x<0 to FP reg
// If x NAN, ZERO, INFINITY, set *y=0 as a 32-bit integer, and exit
{ .mmb
(p0)    setf.exp    f11 = r14
(p6)    st4         [r33] = r0
(p6)    br.ret.spnt b0 ;;
}

// Form exponent mask
// p7 if x<0, else p8
{ .mfi
(p0)    mov         r17 = 0x1ffff
(p0)    fcmp.lt.unc p7,p8 = f8,f0
        nop.i 999 ;;
}

// Test for fnorm(x) denormal, means x double-extended denormal
{ .mfi
        nop.m 999
(p0)    fclass.m.unc p9,p0 = f9, 0x0b
        nop.i 999 ;;
}

// If x double-extended denormal add 64 to exponent bias for scaling
// If x double-extended denormal multiply x * 2^64 which is normal
{ .mfi
(p9)    add         r15 = 64, r15
(p9)    fmpy        f9 = f9, f12
        nop.i 999 ;;
}

// true exponent stored to int pointer
// the bias is treated as 0xfffe instead of
// normal 0xffff because we want the significand
// to be in the range <=0.5 sig < 1.0
// Store the value of the exponent at the pointer in r33

// If x>0 form significand result
{ .mfi
        nop.m 999
(p8)    fmerge.se   f8 = f10,f9
        nop.i 999  ;;
}

// Get signexp of normalized x
// If x<0 form significand result
{ .mfi
(p0)    getf.exp    r16 = f9
(p7)    fmerge.se   f8 = f11,f9
        nop.i 999  ;;
}

// Get exp of normalized x
// Subtract off bias to get true exponent of x
{ .mmi
(p0)    and         r18 = r17,r16 ;;
(p0)    sub         r19 = r18,r15
        nop.i 999  ;;
}

// Store int y as a 32-bit integer
// Make the value a double
{ .mfb
(p0)    st4         [r33] = r19
(p0)    fnorm.d     f8 = f8
(p0)    br.ret.sptk b0 ;;
}

.endp __libm_frexp_4
ASM_SIZE_DIRECTIVE(__libm_frexp_4)
strong_alias(__libm_frexp_4, _GI___libm_frexp_4)
