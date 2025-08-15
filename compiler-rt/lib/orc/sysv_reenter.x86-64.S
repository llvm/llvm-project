//===-- orc_rt_macho_tlv.x86-64.s -------------------------------*- ASM -*-===//
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

// The content of this file is x86_64-only
#if defined(__x86_64__)

// Save all GRPS except %rsp.
// This value is also subtracted from %rsp below, despite the fact that %rbp
// has already been pushed, because we need %rsp to stay 16-byte aligned.
#define GPR_SAVE_SPACE_SIZE             15 * 8
#define FXSAVE64_SAVE_SPACE_SIZE        512
#define REGISTER_SAVE_SPACE_SIZE        \
        GPR_SAVE_SPACE_SIZE + FXSAVE64_SAVE_SPACE_SIZE

        .text

        // returns address of TLV in %rax, all other registers preserved
        .globl __orc_rt_sysv_reenter
__orc_rt_sysv_reenter:
        pushq           %rbp
        movq            %rsp,        %rbp
        subq            $REGISTER_SAVE_SPACE_SIZE, %rsp
        movq            %rax,     -8(%rbp)
        movq            %rbx,    -16(%rbp)
        movq            %rcx,    -24(%rbp)
        movq            %rdx,    -32(%rbp)
        movq            %rsi,    -40(%rbp)
        movq            %rdi,    -48(%rbp)
        movq            %r8,     -56(%rbp)
        movq            %r9,     -64(%rbp)
        movq            %r10,    -72(%rbp)
        movq            %r11,    -80(%rbp)
        movq            %r12,    -88(%rbp)
        movq            %r13,    -96(%rbp)
        movq            %r14,   -104(%rbp)
        movq            %r15,   -112(%rbp)
        fxsave64        (%rsp)
        movq            8(%rbp), %rdi

        // Load return address and subtract five from it (on the assumption
        // that it's a call instruction).
        subq            $5, %rdi

        // Call __orc_rt_resolve to look up the implementation corresponding to
        // the calling stub, then store this in x17 (which we'll return to
        // below).
#if !defined(__APPLE__)
        call            __orc_rt_resolve
#else
        call            ___orc_rt_resolve
#endif
        movq            %rax,   8(%rbp)
        fxrstor64       (%rsp)
        movq            -112(%rbp),     %r15
        movq            -104(%rbp),     %r14
        movq            -96(%rbp),      %r13
        movq            -88(%rbp),      %r12
        movq            -80(%rbp),      %r11
        movq            -72(%rbp),      %r10
        movq            -64(%rbp),      %r9
        movq            -56(%rbp),      %r8
        movq            -48(%rbp),      %rdi
        movq            -40(%rbp),      %rsi
        movq            -32(%rbp),      %rdx
        movq            -24(%rbp),      %rcx
        movq            -16(%rbp),      %rbx
        movq            -8(%rbp),       %rax
        addq            $REGISTER_SAVE_SPACE_SIZE, %rsp
        popq            %rbp
        ret

#endif // defined(__x86_64__)
