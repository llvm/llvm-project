/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"


        ALN_QUAD
.L_spow_zeromask:  .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
                   .long 0xFFFFFFFF
.L_spow_mask_two:  .long 0x40000000
                   .long 0x40000000
                   .long 0x40000000
                   .long 0x40000000
                   .long 0x40000000
                   .long 0x40000000
                   .long 0x40000000
                   .long 0x40000000

        .text
        ALN_FUNC
        .globl ENT(__fvs_pow_fma3_mask)
ENT(__fvs_pow_fma3_mask):

        subq $8, %rsp

        vptest  .L_spow_zeromask(%rip), %xmm2
        je      .L_fvs_pow_fma3_done

        vmovups .L_spow_mask_two(%rip),%xmm3
        vblendvps %xmm2,%xmm0,%xmm3,%xmm0
        vblendvps %xmm2,%xmm1,%xmm3,%xmm1

        CALL(ENT(__fvs_pow_fma3))

.L_fvs_pow_fma3_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvs_pow_fma3_mask)
        ELF_SIZE(__fvs_pow_fma3_mask)


        .text
        ALN_FUNC
        .globl ENT(__fvs_pow_fma3_256_mask)
ENT(__fvs_pow_fma3_256_mask):

        subq $8, %rsp

        vptest  .L_spow_zeromask(%rip), %ymm2
        je      .L_fvs_pow_fma3_256_done

        vmovups .L_spow_mask_two(%rip),%ymm3
        vblendvps %ymm2,%ymm0,%ymm3,%ymm0
        vblendvps %ymm2,%ymm1,%ymm3,%ymm1
        
        CALL(ENT(__fvs_pow_fma3_256))

.L_fvs_pow_fma3_256_done:

        addq $8, %rsp
        ret

        ELF_FUNC(__fvs_pow_fma3_256_mask)
        ELF_SIZE(__fvs_pow_fma3_256_mask)

