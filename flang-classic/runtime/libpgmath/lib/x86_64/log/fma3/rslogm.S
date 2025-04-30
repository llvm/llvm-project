/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        ALN_QUAD
.L_slog_zeromask:  .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF

.L_slog_one:       .long 0x3F800000
                   .long 0x3F800000
                   .long 0x3F800000
                   .long 0x3F800000
                   .long 0x3F800000
                   .long 0x3F800000
                   .long 0x3F800000
                   .long 0x3F800000


        .text
        ALN_FUNC
        .globl ENT(__rvs_log_fma3_mask)
ENT(__rvs_log_fma3_mask):

        subq $8, %rsp

        vptest  .L_slog_zeromask(%rip), %xmm1
        je      .L_rvs_log_fma3_done

        vmovups .L_slog_one(%rip), %xmm2

        vblendvps %xmm1,%xmm0,%xmm2,%xmm0

        CALL(ENT(__rvs_log_fma3))

.L_rvs_log_fma3_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__rvs_log_fma3_mask)
        ELF_SIZE(__rvs_log_fma3_mask)


        .text
        ALN_FUNC
        .globl ENT(__rvs_log_fma3_256_mask)
ENT(__rvs_log_fma3_256_mask):

        subq $8, %rsp

        vptest  .L_slog_zeromask(%rip), %ymm1
        je      .L_rvs_log_fma3_256_done

        vmovups .L_slog_one(%rip), %ymm2

        vblendvps %ymm1,%ymm0,%ymm2,%ymm0
        
        CALL(ENT(__rvs_log_fma3_256))

.L_rvs_log_fma3_256_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__rvs_log_fma3_256_mask)
        ELF_SIZE(__rvs_log_fma3_256_mask)

