/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

.L_datan_zeromask: .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF
                   .quad 0xFFFFFFFFFFFFFFFF

        .text
        ALN_FUNC
        .globl ENT(__fvd_atan_mask)
ENT(__fvd_atan_mask):

        subq $8, %rsp

        vptest  .L_datan_zeromask(%rip), %xmm1
        je      .L_fvd_atan_done

        vandpd  %xmm0,%xmm1,%xmm0
        CALL(ENT(__fvd_atan))

.L_fvd_atan_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvd_atan_mask)
        ELF_SIZE(__fvd_atan_mask)


        .text
        ALN_FUNC
        .globl ENT(__fvd_atan_256_mask)
ENT(__fvd_atan_256_mask):

        subq $8, %rsp

        vptest  .L_datan_zeromask(%rip), %ymm1
        je      .L_fvd_atan_256_done

        vandpd  %ymm0,%ymm1,%ymm0
        CALL(ENT(__fvd_atan_256))

.L_fvd_atan_256_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvd_atan_256_mask)
        ELF_SIZE(__fvd_atan_256_mask)

