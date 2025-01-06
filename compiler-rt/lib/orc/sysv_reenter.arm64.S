//===-- sysv_reenter.arm64.s ------------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

// The content of this file is arm64-only
#if defined(__arm64__) || defined(__aarch64__)

        .text

        // Saves GPRs, calls __orc_rt_resolve
        .globl __orc_rt_sysv_reenter
__orc_rt_sysv_reenter:
        // Save register state, set up new stack frome.
        stp  x27, x28, [sp, #-16]!
        stp  x25, x26, [sp, #-16]!
        stp  x23, x24, [sp, #-16]!
        stp  x21, x22, [sp, #-16]!
        stp  x19, x20, [sp, #-16]!
        stp  x14, x15, [sp, #-16]!
        stp  x12, x13, [sp, #-16]!
        stp  x10, x11, [sp, #-16]!
        stp   x8,  x9, [sp, #-16]!
        stp   x6,  x7, [sp, #-16]!
        stp   x4,  x5, [sp, #-16]!
        stp   x2,  x3, [sp, #-16]!
        stp   x0,  x1, [sp, #-16]!
        stp  q30, q31, [sp, #-32]!
        stp  q28, q29, [sp, #-32]!
        stp  q26, q27, [sp, #-32]!
        stp  q24, q25, [sp, #-32]!
        stp  q22, q23, [sp, #-32]!
        stp  q20, q21, [sp, #-32]!
        stp  q18, q19, [sp, #-32]!
        stp  q16, q17, [sp, #-32]!
        stp  q14, q15, [sp, #-32]!
        stp  q12, q13, [sp, #-32]!
        stp  q10, q11, [sp, #-32]!
        stp   q8,  q9, [sp, #-32]!
        stp   q6,  q7, [sp, #-32]!
        stp   q4,  q5, [sp, #-32]!
        stp   q2,  q3, [sp, #-32]!
        stp   q0,  q1, [sp, #-32]!

        // Look up the return address and subtract 8 from it (on the assumption
        // that it's a standard arm64 reentry trampoline) to get back the
	// trampoline's address.
        sub   x0, x30, #8

        // Call __orc_rt_resolve to look up the implementation corresponding to
        // the calling stub, then store this in x17 (which we'll return to
	// below).
#if !defined(__APPLE__)
        bl    __orc_rt_resolve
#else
        bl    ___orc_rt_resolve
#endif
        mov   x17, x0

        // Restore the register state.
        ldp   q0,  q1, [sp], #32
        ldp   q2,  q3, [sp], #32
        ldp   q4,  q5, [sp], #32
        ldp   q6,  q7, [sp], #32
        ldp   q8,  q9, [sp], #32
        ldp  q10, q11, [sp], #32
        ldp  q12, q13, [sp], #32
        ldp  q14, q15, [sp], #32
        ldp  q16, q17, [sp], #32
        ldp  q18, q19, [sp], #32
        ldp  q20, q21, [sp], #32
        ldp  q22, q23, [sp], #32
        ldp  q24, q25, [sp], #32
        ldp  q26, q27, [sp], #32
        ldp  q28, q29, [sp], #32
        ldp  q30, q31, [sp], #32
        ldp   x0,  x1, [sp], #16
        ldp   x2,  x3, [sp], #16
        ldp   x4,  x5, [sp], #16
        ldp   x6,  x7, [sp], #16
        ldp   x8,  x9, [sp], #16
        ldp  x10, x11, [sp], #16
        ldp  x12, x13, [sp], #16
        ldp  x14, x15, [sp], #16
        ldp  x19, x20, [sp], #16
        ldp  x21, x22, [sp], #16
        ldp  x23, x24, [sp], #16
        ldp  x25, x26, [sp], #16
        ldp  x27, x28, [sp], #16
        ldp  x29, x30, [sp], #16

        // Return to the function implementation (rather than the stub).
        ret  x17

#endif // defined(__arm64__) || defined(__aarch64__)
