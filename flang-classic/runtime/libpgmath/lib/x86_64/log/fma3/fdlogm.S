/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        ALN_QUAD
.L_dlog_zeromask:
        .quad 0xFFFFFFFFFFFFFFFF
        .quad 0xFFFFFFFFFFFFFFFF
        .quad 0xFFFFFFFFFFFFFFFF
        .quad 0xFFFFFFFFFFFFFFFF

        ALN_QUAD
.L_dlog_one_for_mask:
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */


        .text
        ALN_FUNC
        .globl ENT(__fvd_log_fma3_mask)
ENT(__fvd_log_fma3_mask):

        subq $8, %rsp

        vptest  .L_dlog_zeromask(%rip), %xmm1
        je      .L_fvd_log_fma3_done

        vmovupd .L_dlog_one_for_mask(%rip), %xmm2

        vblendvpd %xmm1,%xmm0,%xmm2,%xmm0
        CALL(ENT(__fvd_log_fma3))

.L_fvd_log_fma3_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvd_log_fma3_mask)
        ELF_SIZE(__fvd_log_fma3_mask)


        .text
        ALN_FUNC
        .globl ENT(__fvd_log_fma3_256_mask)
ENT(__fvd_log_fma3_256_mask):

        subq $8, %rsp

        vptest  .L_dlog_zeromask(%rip), %ymm1
        je      .L_fvd_log_fma3_256_done

        vmovupd .L_dlog_one_for_mask(%rip), %ymm2

        vblendvpd %ymm1,%ymm0,%ymm2,%ymm0
        CALL(ENT(__fvd_log_fma3_256))

.L_fvd_log_fma3_256_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvd_log_fma3_256_mask)
        ELF_SIZE(__fvd_log_fma3_256_mask)

