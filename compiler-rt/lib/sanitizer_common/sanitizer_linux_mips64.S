// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Avoid being marked as needing an executable stack:
#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif

// Further contents are mips64 only:
#if defined(__linux__) && defined(__mips64)

.section .text
.set noreorder
.globl internal_sigreturn
.type internal_sigreturn, @function
internal_sigreturn:

        li $v0,5211 // #5211 is for SYS_rt_sigreturn
        syscall

.size internal_sigreturn, .-internal_sigreturn

#endif // defined(__linux__) && defined(__mips64)
