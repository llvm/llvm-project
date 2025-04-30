/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

.L_satan_zeromask: .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF


        .text
        ALN_FUNC
        .globl ENT(__fvs_atan_mask)
ENT(__fvs_atan_mask):

        subq $8, %rsp

        vptest  .L_satan_zeromask(%rip), %xmm1
        je      .L_fvs_atan_done

        vandps  %xmm0,%xmm1,%xmm0
        CALL(ENT(__fvs_atan))

.L_fvs_atan_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvs_atan_mask)
        ELF_SIZE(__fvs_atan_mask)


        .text
        ALN_FUNC
        .globl ENT(__fvs_atan_256_mask)
ENT(__fvs_atan_256_mask):

        subq $8, %rsp

        vptest  .L_satan_zeromask(%rip), %ymm1
        je      .L_fvs_atan_256_done

        vandps  %ymm0,%ymm1,%ymm0
        CALL(ENT(__fvs_atan_256))

.L_fvs_atan_256_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvs_atan_256_mask)
        ELF_SIZE(__fvs_atan_256_mask)

