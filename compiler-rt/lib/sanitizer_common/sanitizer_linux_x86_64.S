// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Avoid being marked as needing an executable stack:
#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif

// Further contents are x86_64-only:
#if defined(__linux__) && defined(__x86_64__)

#include "../builtins/assembly.h"

// If the "naked" function attribute were supported for x86 we could
// do this via inline asm.
.text
.balign 4
DEFINE_COMPILERRT_FUNCTION(internal_sigreturn)
        mov           $0xf,             %eax    // 0xf == SYS_rt_sigreturn
        mov           %rcx,             %r10
        syscall
        ret                                     // Won't normally reach here.
END_COMPILERRT_FUNCTION(internal_sigreturn)

#endif // defined(__linux__) && defined(__x86_64__)
