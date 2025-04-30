/* Set saved user context to current context.  C-SKY ABIV2 version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

#include "ucontext_i.h"

/* int setcontext (const ucontext_t *ucp) */

ENTRY (__setcontext)
	mov	t0, a0
	subi	sp, 8
	stw	a0, (sp, 0)

	/* Set sigmask.  */
	movi	a0, SIG_SETMASK
	addi	a1, t0, UCONTEXT_SIGMASK
	movi	a2, 0

	/* Do sigprocmask syscall.  */
#ifdef __PIC__
	subi	sp, 8
	stw	gb, (sp, 0)
	grs	gb, .Lgetpc1
.Lgetpc1:
	lrw	a3, .Lgetpc1@GOTPC
	addu	gb, gb, a3
	lrw	a3, __sigprocmask@PLT
	ldr.w	a3, (gb, a3 << 0)
	jsr	a3
	ldw	gb, (sp, 0)
	addi	sp, 8
#else
	jsri	__sigprocmask
#endif /* __PIC__ */

	ldw	t0, (sp, 0)
	addi	sp, 8
	/* Set r0-r11. Load a0-a3, for makecontext requires.  */
	addi	t1, t0, MCONTEXT_CSKY_A0
	ldm	r0-r11, (t1)
	ldw	sp, (t0, MCONTEXT_CSKY_SP)	/* Load sp.  */
	ldw	r15, (t0, MCONTEXT_CSKY_LR)	/* Load lr.  */
	ldw	t0, (t0, MCONTEXT_CSKY_PC)	/* Load start addr.  */
	jmp	t0
END (setcontext)
weak_alias (__setcontext, setcontext)

ENTRY (__startcontext)
	.cfi_label .Ldummy
	cfi_undefined (lr)
	mov	a0, r9
	cmpnei	r9, 0	/* r9 was set in makecontext.  */
	bf	1f	/* null, then exit.  */

	/* Call setcontext.  */
#ifdef __PIC__
	grs	t1, .Lgetpc2
.Lgetpc2:
	lrw	t0, .Lgetpc2@GOTPC
	addu	t1, t1, t0
	lrw	t0, __setcontext@GOT
	ldr.w	t0, (t1, t0 << 0)
#else
	lrw	t0, __setcontext
#endif
	jsr	t0

1:
	/* Call _exit.  */
#ifdef __PIC__
	grs	t1, .Lgetpc3
.Lgetpc3:
	lrw	t0, .Lgetpc3@GOTPC
	addu	t1, t1, t0
	lrw	t0, _exit@GOT
	ldr.w	t0, (t1, t0 << 0)
#else
	lrw	t0, _exit
#endif
	jsr	t0

END (__startcontext)
