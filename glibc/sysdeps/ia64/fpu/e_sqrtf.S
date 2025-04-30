.file "sqrtf.s"


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
// History:
//
// 02/02/00 Initial version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
//*********************************************************************
//
// Function:   Combined sqrtf(x), where
//                         _
//             sqrtf(x) = |x, for single precision x values
//
//********************************************************************
//
// Accuracy:       Correctly Rounded
//
//********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8  (Input and Return Value)
//                              f7 -f14
//
//    General Purpose Registers:
//      r32-r36 (Locals)
//      r37-r40 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6, p7, p8
//
//********************************************************************
//
// IEEE Special Conditions:
//
//    All faults and exceptions should be raised correctly.
//    sqrtf(QNaN) = QNaN
//    sqrtf(SNaN) = QNaN
//    sqrtf(+/-0) = +/-0
//    sqrtf(negative) = QNaN and error handling is called
//
//********************************************************************
//
// Implementation:
//
//  Modified Newton-Raphson Algorithm
//
//********************************************************************


GR_SAVE_B0                    = r34
GR_SAVE_PFS                   = r33
GR_SAVE_GP                    = r35

GR_Parameter_X                = r37
GR_Parameter_Y                = r38
GR_Parameter_RESULT           = r39
GR_Parameter_TAG              = r40

FR_X             = f13
FR_Y             = f0
FR_RESULT        = f8


.section .text
GLOBAL_IEEE754_ENTRY(sqrtf)
{ .mlx
  // BEGIN SINGLE PRECISION MINIMUM LATENCY SQUARE ROOT ALGORITHM
  alloc r32= ar.pfs,0,5,4,0
  // exponent of +1/2 in r2
  movl r2 = 0x0fffe
} { .mfi
  // +1/2 in f12
  nop.m 0
  frsqrta.s0 f7,p6=f8
  nop.i 0;;
} { .mfi
  setf.exp f12 = r2
  // Step (1)
  // y0 = 1/sqrt(a) in f7
  fclass.m.unc p7,p8 = f8,0x3A
  nop.i 0
} { .mfi
  nop.m 0
  // Make a copy of x just in case
  mov f13 = f8
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (2)
  // H0 = 1/2 * y0 in f9
  (p6) fma.s1 f9=f12,f7,f0
  nop.i 0
} { .mfi
  nop.m 0
  // Step (3)
  // S0 = a * y0 in f7
  (p6) fma.s1 f7=f8,f7,f0
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (4)
  // d = 1/2 - S0 * H0 in f10
  (p6) fnma.s1 f10=f7,f9,f12
  nop.i 0
} { .mfi
  nop.m 0
  // Step (0'')
  // 3/2 = 1 + 1/2 in f12
  (p6) fma.s1 f12=f12,f1,f1
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (5)
  // e = 1 + 3/2 * d in f12
  (p6) fma.s1 f12=f12,f10,f1
  nop.i 0
} { .mfi
  nop.m 0
  // Step (6)
  // T0 = d * S0 in f11
  (p6) fma.s1 f11=f10,f7,f0
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (7)
  // G0 = d * H0 in f10
  (p6) fma.s1 f10=f10,f9,f0
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (8)
  // S1 = S0 + e * T0 in f7
  (p6) fma.s.s1 f7=f12,f11,f7
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (9)
  // H1 = H0 + e * G0 in f12
  (p6) fma.s1 f12=f12,f10,f9
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (10)
  // d1 = a - S1 * S1 in f9
  (p6) fnma.s1 f9=f7,f7,f8
  nop.i 0;;
} { .mfb
  nop.m 0
  // Step (11)
  // S = S1 + d1 * H1 in f7
  (p6) fma.s.s0 f8=f9,f12,f7
  (p6) br.ret.sptk   b0 ;;
// END SINGLE PRECISION MINIMUM LATENCY SQUARE ROOT ALGORITHM
} { .mfb
  nop.m 0
       mov   f8 = f7
  (p8) br.ret.sptk b0 ;;
}
//
// This branch includes all those special values that are not negative,
// with the result equal to frcpa(x)
//
GLOBAL_IEEE754_END(sqrtf)
libm_alias_float_other (__sqrt, sqrt)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mii
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        mov   GR_Parameter_TAG = 50
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
        stfs [GR_Parameter_Y] = FR_Y,16         // Store Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfs [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfs [GR_Parameter_Y] = FR_RESULT       // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT]       // Get return result off stack
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
        mov   b0 = GR_SAVE_B0                  // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
};;

LOCAL_LIBM_END(__libm_error_region)


.type   __libm_error_support#,@function
.global __libm_error_support#
