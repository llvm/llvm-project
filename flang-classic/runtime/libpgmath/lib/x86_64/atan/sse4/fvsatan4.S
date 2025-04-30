/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        .text
        ALN_FUNC
        .globl ENT(__fvsatan)
ENT(__fvsatan):

        subq    $40, %rsp

        movupd  %xmm0, (%rsp)                  /* Save xmm0 */

        CALL(ENT(__mth_i_atan))                /* atan(x(1)) */
        movss   %xmm0, 16(%rsp)                /* Save first result */

        movss 4(%rsp),%xmm0                    /* Fetch x(2) */
        CALL(ENT(__mth_i_atan))                /* atan(x(2)) */
        movss   %xmm0, 20(%rsp)                /* Save second result */
        movss 8(%rsp),%xmm0                    /* Fetch x(3) */
        CALL(ENT(__mth_i_atan))                /* atan(x(3)) */
        movss   %xmm0, 24(%rsp)                /* Save third result */

        movss 12(%rsp),%xmm0                   /* Fetch x(4) */
        CALL(ENT(__mth_i_atan))                /* atan(x(4)) */
        movss   %xmm0, 28(%rsp)                /* Save fourth result */

        movupd  16(%rsp), %xmm0                /* Put all results in xmm0 */

        addq    $40, %rsp
        ret

        ELF_FUNC(__fvsatan)
        ELF_SIZE(__fvsatan)

