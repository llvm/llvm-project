/* Thread-local storage handling in the ELF dynamic linker.  ARM version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <arm-features.h>
#include <tls.h>
#include "tlsdesc.h"

	.text
	@ emit debug information with cfi
	@ use arm-specific pseudos for unwinding itself
	CFI_SECTIONS
	.hidden _dl_tlsdesc_return
	.global	_dl_tlsdesc_return
	.type	_dl_tlsdesc_return,#function
	cfi_startproc
	eabi_fnstart
	.align 2
_dl_tlsdesc_return:
	ldr	r0, [r0]
	BX	(lr)
	eabi_fnend
	cfi_endproc
	.size	_dl_tlsdesc_return, .-_dl_tlsdesc_return

	.hidden _dl_tlsdesc_undefweak
	.global	_dl_tlsdesc_undefweak
	.type	_dl_tlsdesc_undefweak,#function
	cfi_startproc
	eabi_fnstart
	.align 2
_dl_tlsdesc_undefweak:
	GET_TLS (r1)
	rsb 	r0, r0, #0
	BX	(lr)
	cfi_endproc
	eabi_fnend
	.size	_dl_tlsdesc_undefweak, .-_dl_tlsdesc_undefweak

#ifdef SHARED
	.hidden _dl_tlsdesc_dynamic
	.global	_dl_tlsdesc_dynamic
	.type	_dl_tlsdesc_dynamic,#function


/*
	The assembly code that follows is a rendition of the following
	C code, hand-optimized a little bit.

ptrdiff_t
_dl_tlsdesc_dynamic(struct tlsdesc *tdp)
{
       struct tlsdesc_dynamic_arg *td = tdp->argument.pointer;
       dtv_t *dtv = (dtv_t *)THREAD_DTV();
       if (__builtin_expect (td->gen_count <= dtv[0].counter
                             && dtv[td->tlsinfo.ti_module].pointer.val
                                != TLS_DTV_UNALLOCATED,
                             1))
               return dtv[td->tlsinfo.ti_module].pointer.val +
                       td->tlsinfo.ti_offset - __builtin_thread_pointer();

       return __tls_get_addr (&td->tlsinfo) - __builtin_thread_pointer();
}

*/
	cfi_startproc
	eabi_fnstart
	.align 2
_dl_tlsdesc_dynamic:
	/* Our calling convention is to clobber r0, r1 and the processor
	   flags.  All others that are modified must be saved */
	eabi_save ({r2,r3,r4,lr})
	push	{r2,r3,r4,lr}
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r2,0)
	cfi_rel_offset (r3,4)
	cfi_rel_offset (r4,8)
	cfi_rel_offset (lr,12)
	ldr	r1, [r0] /* td */
	GET_TLS (lr)
	mov	r4, r0 /* r4 = tp */
	ldr	r0, [r0]
	ldr	r2, [r1, #8] /* gen_count */
	ldr	r3, [r0]
	cmp	r2, r3
	bhi	1f
	ldr	r3, [r1]
#ifndef ARM_NO_INDEX_REGISTER
	ldr	r2, [r0, r3, lsl #3]
#else
	add	lr, r0, r3, lsl #3
	ldr	r2, [lr]
#endif
	cmn	r2, #1
	ittt	ne
	ldrne	r3, [r1, #4]
	addne	r3, r2, r3
	rsbne	r0, r4, r3
	bne	2f
1:	mov	r0, r1
	bl	__tls_get_addr
	rsb	r0, r4, r0
2:
#if ((defined (__ARM_ARCH_4T__) && defined (__THUMB_INTERWORK__)) \
     || defined (ARM_ALWAYS_BX))
	pop	{r2,r3,r4, lr}
	cfi_adjust_cfa_offset (-16)
	cfi_restore (lr)
	cfi_restore (r4)
	cfi_restore (r3)
	cfi_restore (r2)
	bx	lr
#else
	pop	{r2,r3,r4, pc}
#endif
	eabi_fnend
	cfi_endproc
	.size	_dl_tlsdesc_dynamic, .-_dl_tlsdesc_dynamic
#endif /* SHARED */
