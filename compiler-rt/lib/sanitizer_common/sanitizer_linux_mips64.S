// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sanitizer_common/sanitizer_asm.h"

// Avoid being marked as needing an executable stack:
NO_EXEC_STACK_DIRECTIVE

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
