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
	.globl ENT(__mth_i_dremainder)
ENT(__mth_i_dremainder):
	RZ_PUSH

        /* Move arguments to fp stack */
        movsd     %xmm1, RZ_OFF(24)(%rsp)
        movsd     %xmm0, RZ_OFF(16)(%rsp)
        fldl      RZ_OFF(24)(%rsp)
        fldl      RZ_OFF(16)(%rsp)

        /* Loop over partial remainder until done */
.L_remlpd:
        fprem1
        fstsw     %ax
        test      $4, %ah
        jnz       .L_remlpd

        /* Store result back to xmm0 */
        fstpl     RZ_OFF(16)(%rsp)
        fstp      %st(0)
        movsd     RZ_OFF(16)(%rsp), %xmm0
        RZ_POP
        ret

	ELF_FUNC(__mth_i_dremainder)
	ELF_SIZE(__mth_i_dremainder)
