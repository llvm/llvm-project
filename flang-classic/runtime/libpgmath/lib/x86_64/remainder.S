/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * ============================================================
 */

#include "directives.h"

	.text
        ALN_FUNC
	.globl ENT(__mth_i_remainder)
ENT(__mth_i_remainder):
	RZ_PUSH

        /* Move arguments to fp stack */
        movss     %xmm1, RZ_OFF(12)(%rsp)
        movss     %xmm0, RZ_OFF(8)(%rsp)
        flds      RZ_OFF(12)(%rsp)
        flds      RZ_OFF(8)(%rsp)

        /* Loop over partial remainder until done */
.L_remlps:
        fprem1
        fstsw     %ax
        test      $4, %ah
        jnz       .L_remlps

        /* Store result back to xmm0 */
        fstps     RZ_OFF(8)(%rsp)
        fstp      %st(0)
        movss     RZ_OFF(8)(%rsp), %xmm0
        RZ_POP
        ret

	ELF_FUNC(__mth_i_remainder)
	ELF_SIZE(__mth_i_remainder)
