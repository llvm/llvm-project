/* Wrapper around clone system call.  PowerPC64 version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#define _ERRNO_H	1
#include <bits/errno.h>

/* This is the only really unusual system call in PPC linux, but not
   because of any weirdness in the system call itself; because of
   all the freaky stuff we have to do to make the call useful.  */

/* int [r3] clone(int (*fn)(void *arg) [r3], void *child_stack [r4],
                  int flags [r5], void *arg [r6], void *parent_tid [r7],
                  void *tls [r8], void *child_tid [r9]); */

ENTRY (__clone)
	CALL_MCOUNT 7

	/* Check for child_stack == NULL || fn == NULL.  */
	cmpdi	cr0,r4,0
	cmpdi	cr1,r3,0
	cror	cr0*4+eq,cr1*4+eq,cr0*4+eq
	beq-	cr0,L(badargs)

	/* Save some regs in the "red zone".  */
#ifdef USE_PPC_SCV
	std	r28,-32(r1)
	cfi_offset(r28,-32)
#endif
	std	r29,-24(r1)
	std	r30,-16(r1)
	std	r31,-8(r1)
	cfi_offset(r29,-24)
	cfi_offset(r30,-16)
	cfi_offset(r31,-8)

	/* Set up stack frame for child.  */
	clrrdi	r4,r4,4
	li	r0,0
	stdu	r0,-FRAME_MIN_SIZE_PARM(r4)

	/* Save fn, args, stack across syscall.  */
	mr	r30,r3			/* Function in r30.  */
	mr	r29,r5			/* Flags in r29.  */
	mr	r31,r6			/* Argument in r31.  */

	/* 'flags' argument is first parameter to clone syscall.
	   Second is the stack pointer, already in r4.  */
	mr	r3,r5
	/* Move the parent_tid, child_tid and tls arguments. */
	mr	r5,r7
	mr	r6,r8
	mr	r7,r9

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc

	/* Do the call.  */
	li 	r0,SYS_ify(clone)
#ifdef USE_PPC_SCV
	CHECK_SCV_SUPPORT r28 0f
	/* This is equivalent to DO_CALL_SCV, but we cannot use the macro here
	because it uses CFI directives and we just called cfi_endproc.  */
	mflr 	r9
	std 	r9,FRAME_LR_SAVE(r1)
	.machine "push"
	.machine "power9"
	scv 	0
	.machine "pop"
	ld 	r9,FRAME_LR_SAVE(r1)
	mtlr 	r9

	/* Check for child process.  */
	/* When using scv, error is indicated by negative r3.  */
	cmpdi	cr1,r3,0
	b 1f
#endif /* USE_PPC_SCV */
0:      DO_CALL_SC

	/* Check for child process.  */
	/* With sc, error is indicated by cr0.SO.  */
	cmpdi	cr1,r3,0
	crandc	cr1*4+eq,cr1*4+eq,cr0*4+so
1:
	bne-	cr1,L(parent)		/* The '-' is to minimise the race.  */

	std	r2,FRAME_TOC_SAVE(r1)
	/* Call procedure.  */
	PPC64_LOAD_FUNCPTR r30
	mr	r3,r31
	bctrl
	ld	r2,FRAME_TOC_SAVE(r1)

	li	r0,(SYS_ify(exit))
	DO_CALL_SC
	/* We won't ever get here but provide a nop so that the linker
	   will insert a toc adjusting stub if necessary.  */
	nop

L(badargs):
	cfi_startproc
	li	r3,EINVAL
	TAIL_CALL_SYSCALL_ERROR

L(parent):
	/* Check if scv is available. */
	cmpdi cr1,r28,0

	/* Parent.  Restore registers & return.  */
#ifdef USE_PPC_SCV
	cfi_offset(r28,-32)
	ld	r28,-32(r1)
	cfi_restore(r28)
#endif
	cfi_offset(r29,-24)
	cfi_offset(r30,-16)
	cfi_offset(r31,-8)
	ld	r29,-24(r1)
	ld	r30,-16(r1)
	ld	r31,-8(r1)
	cfi_restore(r29)
	cfi_restore(r30)
	cfi_restore(r31)

#ifdef USE_PPC_SCV
	beq cr1,0f
	RET_SCV
	b 1f
#endif
0:	RET_SC
1:	TAIL_CALL_SYSCALL_ERROR


END (__clone)

libc_hidden_def (__clone)
weak_alias (__clone, clone)
