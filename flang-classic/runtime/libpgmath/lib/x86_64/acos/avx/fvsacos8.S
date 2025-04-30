/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        .text
        ALN_FUNC
        .globl ENT(__fvs_acos_256)
ENT(__fvs_acos_256):

        subq    $72, %rsp

        vmovups %ymm0, (%rsp)

        CALL(ENT(__fvs_acos))

        vmovups (%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm0
        vmovups %ymm1, 32(%rsp)

        CALL(ENT(__fvs_acos))
        vmovups 32(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        addq    $72, %rsp
        ret

        ELF_FUNC(__fvs_acos_256)
        ELF_SIZE(__fvs_acos_256)

