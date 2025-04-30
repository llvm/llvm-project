/* Thread-local storage handling in the ELF dynamic linker.
   AArch64 version.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.

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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <tls.h>
#include "tlsdesc.h"

#define NSAVEDQREGPAIRS	16
#define SAVE_Q_REGISTERS				\
	stp	q0, q1,	[sp, #-32*NSAVEDQREGPAIRS]!;	\
	cfi_adjust_cfa_offset (32*NSAVEDQREGPAIRS);	\
	stp	 q2,  q3, [sp, #32*1];			\
	stp	 q4,  q5, [sp, #32*2];			\
	stp	 q6,  q7, [sp, #32*3];			\
	stp	 q8,  q9, [sp, #32*4];			\
	stp	q10, q11, [sp, #32*5];			\
	stp	q12, q13, [sp, #32*6];			\
	stp	q14, q15, [sp, #32*7];			\
	stp	q16, q17, [sp, #32*8];			\
	stp	q18, q19, [sp, #32*9];			\
	stp	q20, q21, [sp, #32*10];			\
	stp	q22, q23, [sp, #32*11];			\
	stp	q24, q25, [sp, #32*12];			\
	stp	q26, q27, [sp, #32*13];			\
	stp	q28, q29, [sp, #32*14];			\
	stp	q30, q31, [sp, #32*15];

#define RESTORE_Q_REGISTERS				\
	ldp	 q2,  q3, [sp, #32*1];			\
	ldp	 q4,  q5, [sp, #32*2];			\
	ldp	 q6,  q7, [sp, #32*3];			\
	ldp	 q8,  q9, [sp, #32*4];			\
	ldp	q10, q11, [sp, #32*5];			\
	ldp	q12, q13, [sp, #32*6];			\
	ldp	q14, q15, [sp, #32*7];			\
	ldp	q16, q17, [sp, #32*8];			\
	ldp	q18, q19, [sp, #32*9];			\
	ldp	q20, q21, [sp, #32*10];			\
	ldp	q22, q23, [sp, #32*11];			\
	ldp	q24, q25, [sp, #32*12];			\
	ldp	q26, q27, [sp, #32*13];			\
	ldp	q28, q29, [sp, #32*14];			\
	ldp	q30, q31, [sp, #32*15];			\
	ldp	 q0,  q1, [sp], #32*NSAVEDQREGPAIRS;	\
	cfi_adjust_cfa_offset (-32*NSAVEDQREGPAIRS);

	.text

	/* Compute the thread pointer offset for symbols in the static
	   TLS block. The offset is the same for all threads.
	   Prototype:
	   _dl_tlsdesc_return (tlsdesc *) ;
	 */
	.hidden _dl_tlsdesc_return
	.global	_dl_tlsdesc_return
	.type	_dl_tlsdesc_return,%function
	cfi_startproc
	.align 2
_dl_tlsdesc_return:
	BTI_C
	PTR_ARG (0)
	ldr	PTR_REG (0), [x0, #PTR_SIZE]
	RET
	cfi_endproc
	.size	_dl_tlsdesc_return, .-_dl_tlsdesc_return

	/* Handler for undefined weak TLS symbols.
	   Prototype:
	   _dl_tlsdesc_undefweak (tlsdesc *);

	   The second word of the descriptor contains the addend.
	   Return the addend minus the thread pointer. This ensures
	   that when the caller adds on the thread pointer it gets back
	   the addend.  */

	.hidden _dl_tlsdesc_undefweak
	.global	_dl_tlsdesc_undefweak
	.type	_dl_tlsdesc_undefweak,%function
	cfi_startproc
	.align  2
_dl_tlsdesc_undefweak:
	BTI_C
	str	x1, [sp, #-16]!
	cfi_adjust_cfa_offset (16)
	PTR_ARG (0)
	ldr	PTR_REG (0), [x0, #PTR_SIZE]
	mrs	x1, tpidr_el0
	sub	PTR_REG (0), PTR_REG (0), PTR_REG (1)
	ldr	x1, [sp], #16
	cfi_adjust_cfa_offset (-16)
	RET
	cfi_endproc
	.size	_dl_tlsdesc_undefweak, .-_dl_tlsdesc_undefweak

#ifdef SHARED
	/* Handler for dynamic TLS symbols.
	   Prototype:
	   _dl_tlsdesc_dynamic (tlsdesc *) ;

	   The second word of the descriptor points to a
	   tlsdesc_dynamic_arg structure.

	   Returns the offset between the thread pointer and the
	   object referenced by the argument.

	   ptrdiff_t
	   __attribute__ ((__regparm__ (1)))
	   _dl_tlsdesc_dynamic (struct tlsdesc *tdp)
	   {
	     struct tlsdesc_dynamic_arg *td = tdp->arg;
	     dtv_t *dtv = *(dtv_t **)((char *)__thread_pointer + TCBHEAD_DTV);
	     if (__builtin_expect (td->gen_count <= dtv[0].counter
		&& (dtv[td->tlsinfo.ti_module].pointer.val
		    != TLS_DTV_UNALLOCATED),
		1))
	       return dtv[td->tlsinfo.ti_module].pointer.val
		+ td->tlsinfo.ti_offset
		- __thread_pointer;

	     return ___tls_get_addr (&td->tlsinfo) - __thread_pointer;
	   }
	 */

	.hidden _dl_tlsdesc_dynamic
	.global	_dl_tlsdesc_dynamic
	.type	_dl_tlsdesc_dynamic,%function
	cfi_startproc
	.align 2
_dl_tlsdesc_dynamic:
	BTI_C
	PTR_ARG (0)

	/* Save just enough registers to support fast path, if we fall
	   into slow path we will save additional registers.  */
	stp	x1,  x2, [sp, #-32]!
	cfi_adjust_cfa_offset (32)
	stp	x3,  x4, [sp, #16]
	cfi_rel_offset (x1, 0)
	cfi_rel_offset (x2, 8)
	cfi_rel_offset (x3, 16)
	cfi_rel_offset (x4, 24)

	mrs	x4, tpidr_el0
	ldr	PTR_REG (1), [x0,#TLSDESC_ARG]
	ldr	PTR_REG (0), [x4,#TCBHEAD_DTV]
	ldr	PTR_REG (3), [x1,#TLSDESC_GEN_COUNT]
	ldr	PTR_REG (2), [x0,#DTV_COUNTER]
	cmp	PTR_REG (3), PTR_REG (2)
	b.hi	2f
	/* Load r2 = td->tlsinfo.ti_module and r3 = td->tlsinfo.ti_offset.  */
	ldp	PTR_REG (2), PTR_REG (3), [x1,#TLSDESC_MODID]
	add	PTR_REG (0), PTR_REG (0), PTR_REG (2), lsl #(PTR_LOG_SIZE + 1)
	ldr	PTR_REG (0), [x0] /* Load val member of DTV entry.  */
	cmp	PTR_REG (0), #TLS_DTV_UNALLOCATED
	b.eq	2f
	cfi_remember_state
	sub	PTR_REG (3), PTR_REG (3), PTR_REG (4)
	add	PTR_REG (0), PTR_REG (0), PTR_REG (3)
1:
	ldp	 x3,  x4, [sp, #16]
	ldp	 x1,  x2, [sp], #32
	cfi_adjust_cfa_offset (-32)
	RET
2:
	/* This is the slow path. We need to call __tls_get_addr() which
	   means we need to save and restore all the register that the
	   callee will trash.  */

	/* Save the remaining registers that we must treat as caller save.  */
	cfi_restore_state
# if HAVE_AARCH64_PAC_RET
	PACIASP
	cfi_window_save
# endif
# define NSAVEXREGPAIRS 8
	stp	x29, x30, [sp,#-16*NSAVEXREGPAIRS]!
	cfi_adjust_cfa_offset (16*NSAVEXREGPAIRS)
	cfi_rel_offset (x29, 0)
	cfi_rel_offset (x30, 8)
	mov	x29, sp
	stp	 x5,  x6, [sp, #16*1]
	stp	 x7,  x8, [sp, #16*2]
	stp	 x9, x10, [sp, #16*3]
	stp	x11, x12, [sp, #16*4]
	stp	x13, x14, [sp, #16*5]
	stp	x15, x16, [sp, #16*6]
	stp	x17, x18, [sp, #16*7]
	cfi_rel_offset (x5, 16*1)
	cfi_rel_offset (x6, 16*1+8)
	cfi_rel_offset (x7, 16*2)
	cfi_rel_offset (x8, 16*2+8)
	cfi_rel_offset (x9, 16*3)
	cfi_rel_offset (x10, 16*3+8)
	cfi_rel_offset (x11, 16*4)
	cfi_rel_offset (x12, 16*4+8)
	cfi_rel_offset (x13, 16*5)
	cfi_rel_offset (x14, 16*5+8)
	cfi_rel_offset (x15, 16*6)
	cfi_rel_offset (x16, 16*6+8)
	cfi_rel_offset (x17, 16*7)
	cfi_rel_offset (x18, 16*7+8)

	SAVE_Q_REGISTERS

	mov	x0, x1
	bl	__tls_get_addr

	mrs	x1, tpidr_el0
	sub	PTR_REG (0), PTR_REG (0), PTR_REG (1)

	RESTORE_Q_REGISTERS

	ldp	 x5,  x6, [sp, #16*1]
	ldp	 x7,  x8, [sp, #16*2]
	ldp	 x9, x10, [sp, #16*3]
	ldp	x11, x12, [sp, #16*4]
	ldp	x13, x14, [sp, #16*5]
	ldp	x15, x16, [sp, #16*6]
	ldp	x17, x18, [sp, #16*7]

	ldp	x29, x30, [sp], #16*NSAVEXREGPAIRS
	cfi_adjust_cfa_offset (-16*NSAVEXREGPAIRS)
	cfi_restore (x29)
	cfi_restore (x30)
# if HAVE_AARCH64_PAC_RET
	AUTIASP
	cfi_window_save
# endif
	b	1b
	cfi_endproc
	.size	_dl_tlsdesc_dynamic, .-_dl_tlsdesc_dynamic
# undef NSAVEXREGPAIRS
#endif
