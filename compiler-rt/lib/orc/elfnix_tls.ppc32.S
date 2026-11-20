//===-- elfnix_tls.ppc32.S -------------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__) && !defined(__powerpc64__)

        .text
        .p2align 2
        .globl ___orc_rt_elfnix_tls_get_addr
        .type ___orc_rt_elfnix_tls_get_addr,@function
___orc_rt_elfnix_tls_get_addr:
        .cfi_startproc
        // r3: TLS info in, instance address out.
        mflr 0
        stw 0, 4(1)
        stwu 1, -16(1)
        .cfi_def_cfa_offset 16
        .cfi_offset lr, 4
        bl __orc_rt_elfnix_tls_get_addr_impl
        addi 1, 1, 16
        .cfi_def_cfa_offset 0
        lwz 0, 4(1)
        mtlr 0
        .cfi_restore lr
        blr
        .cfi_endproc
        .size ___orc_rt_elfnix_tls_get_addr, .-___orc_rt_elfnix_tls_get_addr

        .section .note.GNU-stack,"",@progbits
#endif
