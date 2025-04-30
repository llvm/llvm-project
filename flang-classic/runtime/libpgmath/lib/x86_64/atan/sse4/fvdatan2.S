/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        .text
        ALN_FUNC
        .globl ENT(__fvdatan)
ENT(__fvdatan):

        subq    $40, %rsp

        movupd  %xmm0, (%rsp)                  /* Save xmm0 */

        CALL(ENT(__mth_i_datan))               /* atan(x(1)) */
        movsd   %xmm0, 16(%rsp)                /* Save first result */

        movsd 8(%rsp),%xmm0                    /* Fetch x(2) */
        CALL(ENT(__mth_i_datan))               /* atan(x(2)) */
        movsd   %xmm0, 24(%rsp)                /* Save second result */

        movupd  16(%rsp), %xmm0                /* Put all results in xmm0 */

        addq    $40, %rsp
        ret

        ELF_FUNC(__fvdatan)
        ELF_SIZE(__fvdatan)








