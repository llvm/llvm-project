/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        .text
        ALN_FUNC
        .globl ENT(__fvd_acos)
ENT(__fvd_acos):

        subq    $40, %rsp

        vmovupd  %xmm0, (%rsp)                 /* Save xmm0 */

        CALL(ENT(__mth_i_dacos))               /* acos(x(1)) */
        vmovsd   %xmm0, 16(%rsp)               /* Save first result */

        vmovsd 8(%rsp),%xmm0                   /* Fetch x(2) */
        CALL(ENT(__mth_i_dacos))               /* acos(x(2)) */
        vmovsd   %xmm0, 24(%rsp)               /* Save second result */

        vmovupd  16(%rsp), %xmm0               /* Put all results in xmm0 */

        addq    $40, %rsp
        ret

        ELF_FUNC(__fvd_acos)
        ELF_SIZE(__fvd_acos)
