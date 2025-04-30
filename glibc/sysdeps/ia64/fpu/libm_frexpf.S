.file "libm_frexpf.s"


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
// 03/20/00 Improved speed
// 06/01/00 Fixed bug when x a double-extended denormal
// 12/08/00 Corrected label on .endp
// 01/23/02 Added handling for int 32 or 64 bits
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// float __libm_frexpf(float x, int* y, int int_type)
// input  floating point f8, pointer to y (r33), int int_type (r34)
// output floating point f8, returns the fraction of x, 0.5 <= fraction < 1.0
// output int* y, returns the true exponent of x
//
// int_type = 0 if int is 32 bits
// int_type = 1 if int is 64 bits
//
// int* y is returned as a 32 bit integer if int_type = 0
// int* y is returned as a 64 bit integer if int_type = 1
//
// Overview of operation
//==============================================================
// break a floating point x number into fraction and an exponent
// The fraction is returned as a float
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
// r32  on input contains the 32-bit IEEE float that is in f8
// r33  on input pointer to 32-bit or 64-bit integer for exponent
// r34  on input contains 0 if output int is 32 bits, else output int is 64 bits
//
// predicate registers:
// p6   set if x is Nan, zero, or infinity
// p7   set if x negative
// p8   set if x positive
// p9   set if x double-extended denormal
// p10  set if int_type = 0, 32-bit integer
// p11  set if int_type = 1, 64-bit integer
//
// floating-point registers:
// f8  input, output
// f9  normalized x
// f10 signexp for significand result for x positive
// f11 signexp for significand result for x negative
// f12 2^64

.section .text
GLOBAL_LIBM_ENTRY(__libm_frexpf)

// Set signexp for significand result for x>0
// If x is a NaN, zero, or infinity, return it.
// Put 0 in the int pointer.
// x NAN, ZERO, INFINITY?
// Set signexp for significand result for x<0
{ .mfi
        mov         r15 = 0x0fffe
        fclass.m    p6,p7 = f8, 0xe7
        mov         r14 = 0x2fffe
}
// Form signexp of 2^64 in case x double-extended denormal
// Save the normalized value of input in f9
// The normalization also sets fault flags and takes faults if necessary
{ .mfi
        mov         r20 = 0x1003f
        fnorm.s0    f9 = f8
        nop.i 999 ;;
}

// Move signexp for significand result for x>0 to FP reg
// Form 2^64 in case x double-extended denormal
{ .mmi
        setf.exp    f10 = r15
        setf.exp    f12 = r20
        nop.i 999 ;;
}

// Move signexp for significand result for x<0 to FP reg
// p7 if x<0, else p8
// If x=0,nan,inf, set p10 if output int to be 32 bits, or set p11 if 64 bits
{ .mfi
        setf.exp    f11 = r14
(p7)    fcmp.lt.s0  p7,p8 = f8,f0
(p6)    cmp.eq.unc  p10,p11 = r34, r0 ;;
}

// If x NAN, ZERO, INFINITY, set *y=0 and exit
{ .mmb
(p10)   st4         [r33] = r0      // Store *y=0 as 32-bit integer
(p11)   st8         [r33] = r0      // Store *y=0 as 64-bit integer
(p6)    br.ret.spnt b0 ;;
}

// Form exponent mask
// Test for fnorm(x) denormal, means x double-extended denormal
{ .mfi
        mov         r17 = 0x1ffff
        fclass.m    p9,p0 = f9, 0x0b
        nop.i 999 ;;
}

// If x double-extended denormal add 64 to exponent bias for scaling
// If x double-extended denormal multiply x * 2^64 which is normal
// Set p10 if output int to be 32 bits, or set p11 if 64 bits
{ .mfi
(p9)    add         r15 = 64, r15
(p9)    fmpy.s0     f9 = f9, f12
        cmp.eq      p10,p11 = r34, r0 ;;
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
        getf.exp    r16 = f9
(p7)    fmerge.se   f8 = f11,f9
        nop.i 999  ;;
}

// Get exp of normalized x
// Subtract off bias to get true exponent of x
{ .mmi
        and         r18 = r17,r16 ;;
        sub         r19 = r18,r15
        nop.i 999  ;;
}

// Store int *y as a 32-bit integer
// Make the value a float
{ .mfi
(p10)   st4         [r33] = r19        // Store *y as 32-bit integer
        fnorm.s.s0  f8 = f8
        nop.i 999
}
{ .mfb
(p11)   st8         [r33] = r19        // Store *y as 64-bit integer
        nop.f 999
        br.ret.sptk b0 ;;
}

GLOBAL_LIBM_END(__libm_frexpf)
