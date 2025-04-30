.file "sqrt.s"


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
//********************************************************************
// History
//********************************************************************
// 02/02/00 Initial version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
//********************************************************************
//
// Function:   Combined sqrt(x), where
//                        _
//             sqrt(x) = |x, for double precision x values
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
//*********************************************************************
//
// IEEE Special Conditions:
//
//    All faults and exceptions should be raised correctly.
//    sqrt(QNaN) = QNaN
//    sqrt(SNaN) = QNaN
//    sqrt(+/-0) = +/-0
//    sqrt(negative) = QNaN and error handling is called
//
//*********************************************************************
//
// Implementation:
//
//  Modified Newton-Raphson Algorithm
//
//*********************************************************************

GR_SAVE_PFS          = r33
GR_SAVE_B0           = r34
GR_SAVE_GP           = r35

GR_Parameter_X       = r37
GR_Parameter_Y       = r38
GR_Parameter_RESULT  = r39


.section .text
GLOBAL_IEEE754_ENTRY(sqrt)
{ .mfi
  alloc r32= ar.pfs,0,5,4,0
  frsqrta.s0 f7,p6=f8
  nop.i 0
} { .mlx
  // BEGIN DOUBLE PRECISION MINIMUM LATENCY SQUARE ROOT ALGORITHM
  nop.m 0
  // exponent of +1/2 in r2
  movl r2 = 0x0fffe;;
} { .mmi
  // +1/2 in f9
  setf.exp f9 = r2
  nop.m 0
  nop.i 0
} { .mlx
  nop.m 0
  // 3/2 in r3
  movl r3=0x3fc00000;;
} { .mfi
  setf.s f10=r3
  // Step (1)
  // y0 = 1/sqrt(a) in f7
  fclass.m.unc p7,p8 = f8,0x3A
  nop.i 0;;
} { .mlx
  nop.m 0
  // 5/2 in r2
  movl r2 = 0x40200000
} { .mlx
  nop.m 0
  // 63/8 in r3
  movl r3 = 0x40fc0000;;
} { .mfi
  setf.s f11=r2
  // Step (2)
  // h = +1/2 * y0 in f6
  (p6) fma.s1 f6=f9,f7,f0
  nop.i 0
} { .mfi
  setf.s f12=r3
  // Step (3)
  // g = a * y0 in f7
  (p6) fma.s1 f7=f8,f7,f0
  nop.i 0
} { .mfi
  nop.m 0
  mov   f15 = f8
  nop.i 0;;
} { .mlx
  nop.m 0
  // 231/16 in r2
  movl r2 = 0x41670000;;
} { .mfi
  setf.s f13=r2
  // Step (4)
  // e = 1/2 - g * h in f9
  (p6) fnma.s1 f9=f7,f6,f9
  nop.i 0
} { .mlx
  nop.m 0
  // 35/8 in r3
  movl r3 = 0x408c0000;;
} { .mfi
  setf.s f14=r3
  // Step (5)
  // S = 3/2 + 5/2 * e in f10
  (p6) fma.s1 f10=f11,f9,f10
  nop.i 0
} { .mfi
  nop.m 0
  // Step (6)
  // e2 = e * e in f11
  (p6) fma.s1 f11=f9,f9,f0
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (7)
  // t = 63/8 + 231/16 * e in f12
  (p6) fma.s1 f12=f13,f9,f12
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (8)
  // S1 = e + e2 * S in f10
  (p6) fma.s1 f10=f11,f10,f9
  nop.i 0
} { .mfi
  nop.m 0
  // Step (9)
  // e4 = e2 * e2 in f11
  (p6) fma.s1 f11=f11,f11,f0
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (10)
  // t1 = 35/8 + e * t in f9
  (p6) fma.s1 f9=f9,f12,f14
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (11)
  // G = g + S1 * g in f12
  (p6) fma.s1 f12=f10,f7,f7
  nop.i 0
} { .mfi
  nop.m 0
  // Step (12)
  // E = g * e4 in f7
  (p6) fma.s1 f7=f7,f11,f0
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (13)
  // u = S1 + e4 * t1 in f10
  (p6) fma.s1 f10=f11,f9,f10
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (14)
  // g1 = G + t1 * E in f7
  (p6) fma.d.s1 f7=f9,f7,f12
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (15)
  // h1 = h + u * h in f6
  (p6) fma.s1 f6=f10,f6,f6
  nop.i 0;;
} { .mfi
  nop.m 0
  // Step (16)
  // d = a - g1 * g1 in f9
  (p6) fnma.s1 f9=f7,f7,f8
  nop.i 0;;
} { .mfb
  nop.m 0
  // Step (17)
  // g2 = g1 + d * h1 in f7
  (p6) fma.d.s0 f8=f9,f6,f7
  (p6) br.ret.sptk b0 ;;
}

{ .mfb
  nop.m 0
       mov   f8 = f7
  (p8) br.ret.sptk b0 ;;
}
{ .mfb
  (p7) mov   r40 = 49
  nop.f 0
  (p7) br.cond.sptk __libm_error_region ;;
}
// END DOUBLE PRECISION MINIMUM LATENCY SQUARE ROOT ALGORITHM
GLOBAL_IEEE754_END(sqrt)
libm_alias_double_other (__sqrt, sqrt)

// Stack operations when calling error support.
//       (1)               (2)                          (3) (call)              (4)
//   sp   -> +          psp -> +                     psp -> +                   sp -> +
//           |                 |                            |                         |
//           |                 | <- GR_Y               R3 ->| <- GR_RESULT            | -> f8
//           |                 |                            |                         |
//           | <-GR_Y      Y2->|                       Y2 ->| <- GR_Y                 |
//           |                 |                            |                         |
//           |                 | <- GR_X               X1 ->|                         |
//           |                 |                            |                         |
//  sp-64 -> +          sp ->  +                     sp ->  +                         +
//    save ar.pfs          save b0                                               restore gp
//    save gp                                                                    restore ar.pfs


LOCAL_LIBM_ENTRY(__libm_error_region)

//
// This branch includes all those special values that are not negative,
// with the result equal to frcpa(x)
//

.prologue
// We are distinguishing between over(under)flow and letting
// __libm_error_support set ERANGE or do anything else needed.

// (1)
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                          // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                      // Save gp
};;


// (2)
{ .mmi
        stfd [GR_Parameter_Y] = f0,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                     // Save b0
};;

.body
// (3)
{ .mib
        stfd [GR_Parameter_X] = f15                    // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y   // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = f8                     // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#          // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;

// (4)
{ .mmi
        ldfd  f8 = [GR_Parameter_RESULT]       // Get return result off stack
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
