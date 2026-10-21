//===-- orc_rt_elfnix_tls.mips.S --------------------------------*- ASM -*-===//
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

#if defined(__mips__)

#if _MIPS_SIM == _ABI64
#define PTR_ADDIU daddiu
#define PTR_LA dla
#else
#define PTR_ADDIU addiu
#define PTR_LA la
#endif

        .text
        .option pic2

        // Returns the TLS address in $2, passing the thread pointer in $5.
        .globl ___orc_rt_elfnix_tls_get_addr
___orc_rt_elfnix_tls_get_addr:
#if _MIPS_SIM == _ABIO32
        .set noreorder
        .cpload $25
        .set reorder
        PTR_ADDIU $sp, $sp, -24
        sw $ra, 20($sp)
#else
        PTR_ADDIU $sp, $sp, -16
        .cpsetup $25, 0, ___orc_rt_elfnix_tls_get_addr
        sd $ra, 8($sp)
#endif

        .set push
#if !defined(__mips_isa_rev) || __mips_isa_rev < 2
        // Linux trap-and-emulates rdhwr on pre-r2 CPUs, so it is always valid.
        .set mips32r2
#endif
        rdhwr $5, $29
        .set pop

        PTR_LA $25, __orc_rt_elfnix_tls_get_addr_impl
        jalr $25

#if _MIPS_SIM == _ABIO32
        lw $ra, 20($sp)
        PTR_ADDIU $sp, $sp, 24
#else
        .cpreturn
        ld $ra, 8($sp)
        PTR_ADDIU $sp, $sp, 16
#endif
        jr $ra

#endif // defined(__mips__)
