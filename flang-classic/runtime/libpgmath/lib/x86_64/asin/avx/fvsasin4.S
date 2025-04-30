/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "directives.h"

        .text
        ALN_FUNC
        .globl ENT(__fvs_asin)
ENT(__fvs_asin):

        subq    $40, %rsp

        vmovupd  %xmm0, (%rsp)                 /* Save xmm0 */

        CALL(ENT(__mth_i_asin))                /* asin(x(1)) */
        vmovss   %xmm0, 16(%rsp)               /* Save first result */

        vmovss 4(%rsp),%xmm0                   /* Fetch x(2) */
        CALL(ENT(__mth_i_asin))                /* asin(x(2)) */
        vmovss   %xmm0, 20(%rsp)               /* Save second result */
        vmovss 8(%rsp),%xmm0                   /* Fetch x(3) */
        CALL(ENT(__mth_i_asin))                /* asin(x(3)) */
        vmovss   %xmm0, 24(%rsp)               /* Save third result */

        vmovss 12(%rsp),%xmm0                  /* Fetch x(4) */
        CALL(ENT(__mth_i_asin))                /* asin(x(4)) */
        vmovss   %xmm0, 28(%rsp)               /* Save fourth result */

        vmovupd  16(%rsp), %xmm0               /* Put all results in xmm0 */

        addq    $40, %rsp
        ret

        ELF_FUNC(__fvs_asin)
        ELF_SIZE(__fvs_asin)

