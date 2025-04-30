.file "fdiml.s"


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
// 06/08/01 Initial version
// 08/23/01 Corrected error tag number
// 05/20/02 Cleaned up namespace and sf0 syntax
// 01/28/03 Improved performance; fixed parameters for call to error routine
//
// API
//==============================================================
// long double fdiml( long double x, long double y );
// input  floating point f8, f9
// output floating point f8
//
//
// Overview of operation
//==============================================================
// fdiml determines the positive difference between the arguments
//   Result = x - y   if x > y
//          = +0      if x <= y
//
// Error support is called if x-y overflows for x > y
//

// Registers used
//==============================================================
// General purpose registers: r14, r32 - r39

rExpBig               = r14

// r36-39 parameters for libm_error_support
GR_SAVE_B0            = r33
GR_SAVE_GP            = r34
GR_SAVE_PFS           = r35

GR_Parameter_X        = r36
GR_Parameter_Y        = r37
GR_Parameter_RESULT   = r38
GR_Parameter_TAG      = r39

// Floating-point registers: f8 - f12

f_tmp_result          = f10
fBig                  = f11
fNormX                = f12

// Predicate registers: p6 - p10


.section .text
GLOBAL_LIBM_ENTRY(fdiml)

{ .mfi
      mov             rExpBig = 0x13fff  // Exponent to indicate overflow
      fcmp.le.s1      p6,p7 = f8, f9     // Is x <= y?
      nop.i           0
}
{ .mfi
      nop.m           0
      fnorm.s1        fNormX = f8        // Save x
      nop.i           0
}
;;

{ .mfi
      setf.exp        fBig = rExpBig     // Constant to test for overflow
      fcmp.eq.s0      p8,p0 = f8, f9     // Dummy op to set Denormal or Invalid
      nop.i           0
}
;;

{ .mfi
      nop.m           0
      fclass.m        p9,p10 = f8, 0x1e3 // Test for x natval, nan, inf
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p6)  fmerge.s        f8 = f0, f0        // Result is +0 if x <= y
      nop.i           0
}
{ .mfi
      nop.m           0
(p7)  fms.s0          f8 = f8, f1, f9    // Result is x - y if x > y
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p10) fclass.m        p9,p10 = f9, 0x1e3 // Test for y natval, nan, inf
      nop.i           0
}
;;

{ .mfi
      nop.m           0
(p10) fcmp.ge.s1      p8,p0 = f8, fBig   // Test result for overflow
      nop.i           0
}
;;

{ .mbb
(p9)  cmp.ne          p8,p0 = r0,r0      // Clear p8 if x or y natval,nan,inf
(p8)  br.cond.spnt    FDIM_OVERFLOW      // Branch if result overflows
      br.ret.sptk     b0                 // Normal return
}
;;


// Here if result will overflow
FDIM_OVERFLOW:
{ .mfi
      alloc           r32=ar.pfs,2,2,4,0
      fms.s0          f_tmp_result = f8,f1,f9 // Normalize result force overflow
      nop.i           0
}
{ .mfb
      mov             GR_Parameter_TAG = 195  // Error code
      nop.f           0
      br.cond.sptk    __libm_error_region     // Branch to error code
}
;;

GLOBAL_LIBM_END(fdiml)
libm_alias_ldouble_other (fdim, fdim)


LOCAL_LIBM_ENTRY(__libm_error_region)
// Call error support to report possible range error
.prologue

{ .mfi
        add   GR_Parameter_Y=-32,sp            // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                          // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                      // Save gp
};;

{ .mmi
        stfe [GR_Parameter_Y] = f9,16          // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp             // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                      // Save b0
};;

.body
{ .mib
        stfe [GR_Parameter_X] = fNormX         // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y // Parameter 3 address
        nop.b 0
}
{ .mib
        stfe [GR_Parameter_Y] = f_tmp_result   // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;

{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;

{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]       // Get return result off stack
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
