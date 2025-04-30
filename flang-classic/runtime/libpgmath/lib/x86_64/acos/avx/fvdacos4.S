/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        .text
        ALN_FUNC
        .globl ENT(__fvd_acos_256)
ENT(__fvd_acos_256):

        subq    $72, %rsp

        vmovups %ymm0, (%rsp)

        CALL(ENT(__fvd_acos))

        vmovupd (%rsp), %ymm2
        vmovapd %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm0
        vmovupd %ymm1, 32(%rsp)

        CALL(ENT(__fvd_acos))
        vmovupd 32(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        addq    $72, %rsp
        ret

        ELF_FUNC(__fvd_acos_256)
        ELF_SIZE(__fvd_acos_256)

