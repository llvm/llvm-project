/* Save and set current context.  C-SKY ABIV2 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include "ucontext_i.h"

/* int swapcontext (ucontext_t *oucp, const ucontext_t *ucp) */

ENTRY (swapcontext)
	/* Save params and lr.  */
	subi	sp, 16
	stw	a0, (sp, 0)
	stw	a1, (sp, 4)
	stw	r15, (sp, 8)

	/* Call getcontext.  */
#ifdef __PIC__
	grs	t1, .Lgetpc1
.Lgetpc1:
	lrw	t0, .Lgetpc1@GOTPC
	addu	t1, t1, t0
	lrw	t0, __getcontext@GOT
	ldr.w	t0, (t1, t0 << 0)
	jsr	t0
#else /* !__PIC__ */
	jsri	__getcontext
#endif

	mov	a3, a0	/* Save return value.  */

	/* Restore params and lr.  */
	ldw	a0, (sp, 0)
	ldw	a1, (sp, 4)
	ldw	r15, (sp, 8)
	addi	sp, 16

	cmpnei	a3, 0
	bt	error_exit

	/* Fix up LR and the PC.  */
	stw	sp, (a0, MCONTEXT_CSKY_SP)
	stw	r15, (a0, MCONTEXT_CSKY_LR)
	stw	r15, (a0, MCONTEXT_CSKY_PC)

	/* Set setcontext's arg.  */
	mov	a0, a1

#ifdef __PIC__
	grs	t1, .Lgetpc2
.Lgetpc2:
	lrw	a3, .Lgetpc2@GOTPC
	addu	t1, t1, a3
	lrw	a3, __setcontext@GOT
	ldr.w	a3, (t1, a3 << 0)
#else /* __PIC__ */
	lrw	a3, __setcontext
#endif /* __PIC__ */
	jmp	a3

error_exit:
	jmp	r15
END (swapcontext)
