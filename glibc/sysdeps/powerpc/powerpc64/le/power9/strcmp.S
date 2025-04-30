/* Optimized strcmp implementation for PowerPC64/POWER9.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef STRCMP
# define STRCMP strcmp
#endif

/* Implements the function

   int [r3] strcmp (const char *s1 [r3], const char *s2 [r4])

   The implementation uses unaligned doubleword access for first 32 bytes
   as in POWER8 patch and uses vectorised loops after that.  */

/* TODO: Change this to actual instructions when minimum binutils is upgraded
   to 2.27.  Macros are defined below for these newer instructions in order
   to maintain compatibility.  */
#define VCTZLSBB(r,v) .long (0x10010602 | ((r)<<(32-11)) | ((v)<<(32-21)))

#define VEXTUBRX(t,a,b) .long (0x1000070d \
				| ((t)<<(32-11))  \
				| ((a)<<(32-16))  \
				| ((b)<<(32-21)) )

#define VCMPNEZB(t,a,b) .long (0x10000507 \
				| ((t)<<(32-11))  \
				| ((a)<<(32-16))  \
				| ((b)<<(32-21)) )

/* Get 16 bytes for unaligned case.
   reg1: Vector to hold next 16 bytes.
   reg2: Address to read from.
   reg3: Permute control vector.  */
#define GET16BYTES(reg1, reg2, reg3) \
	lvx	reg1, 0, reg2; \
	vperm	v8, v2, reg1, reg3; \
	vcmpequb.	v8, v0, v8; \
	beq	cr6, 1f; \
	vspltisb	v9, 0; \
	b	2f; \
	.align 4; \
1: \
	addi    r6, reg2, 16; \
	lvx     v9, 0, r6; \
2: \
	vperm   reg1, v9, reg1, reg3;

/* TODO: change this to .machine power9 when the minimum required binutils
   allows it.  */

	.machine  power7
ENTRY_TOCLESS (STRCMP, 4)
	li	r0, 0

	/* Check if [s1]+16 or [s2]+16 will cross a 4K page boundary using
	   the code:

	    (((size_t) s1) % PAGE_SIZE > (PAGE_SIZE - ITER_SIZE))

	   with PAGE_SIZE being 4096 and ITER_SIZE begin 16.  */

	rldicl	r7, r3, 0, 52
	rldicl	r9, r4, 0, 52
	cmpldi	cr7, r7, 4096-16
	bgt	cr7, L(pagecross_check)
	cmpldi	cr5, r9, 4096-16
	bgt	cr5, L(pagecross_check)

	/* For short strings up to 16 bytes,  load both s1 and s2 using
	   unaligned dwords and compare.  */
	ld	r8, 0(r3)
	ld	r10, 0(r4)
	cmpb	r12, r8, r0
	cmpb	r11, r8, r10
	orc.	r9, r12, r11
	bne	cr0, L(different_nocmpb)

	ld	r8, 8(r3)
	ld	r10, 8(r4)
	cmpb	r12, r8, r0
	cmpb	r11, r8, r10
	orc.	r9, r12, r11
	bne	cr0, L(different_nocmpb)

	addi	r7, r3, 16
	addi	r4, r4, 16

L(align):
	/* Now it has checked for first 16 bytes.  */
	vspltisb	v0, 0
	vspltisb	v2, -1
	lvsr	v6, 0, r4   /* Compute mask.  */
	or	r5, r4, r7
	andi.	r5, r5, 0xF
	beq	cr0, L(aligned)
	andi.	r5, r7, 0xF
	beq	cr0, L(s1_align)
	lvsr	v10, 0, r7   /* Compute mask.  */

	/* Both s1 and s2 are unaligned.  */
	GET16BYTES(v4, r7, v10)
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	beq	cr6, L(match)
	b	L(different)

	/* Align s1 to qw and adjust s2 address.  */
	.align  4
L(match):
	clrldi	r6, r7, 60
	subfic	r5, r6, 16
	add	r7, r7, r5
	add	r4, r4, r5
	andi.	r5, r4, 0xF
	beq	cr0, L(aligned)
	lvsr	v6, 0, r4
	/* There are 2 loops depending on the input alignment.
	   Each loop gets 16 bytes from s1 and s2 and compares.
	   Loop until a mismatch or null occurs.  */
L(s1_align):
	lvx	v4, r7, r0
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	bne	cr6, L(different)

	lvx	v4, r7, r0
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	bne	cr6, L(different)

	lvx	v4, r7, r0
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	bne	cr6, L(different)

	lvx	v4, r7, r0
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	beq	cr6, L(s1_align)
	b	L(different)

	.align  4
L(aligned):
	lvx	v4, 0, r7
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	bne	cr6, L(different)

	lvx	v4, 0, r7
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	bne	cr6, L(different)

	lvx	v4, 0, r7
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	bne	cr6, L(different)

	lvx	v4, 0, r7
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	addi	r7, r7, 16
	addi	r4, r4, 16
	beq	cr6, L(aligned)

	/* Calculate and return the difference.  */
L(different):
	VCTZLSBB(r6, v7)
	VEXTUBRX(r5, r6, v4)
	VEXTUBRX(r4, r6, v5)
	subf	r3, r4, r5
	extsw	r3, r3
	blr

	.align  4
L(different_nocmpb):
	neg	r3, r9
	and	r9, r9, r3
	cntlzd	r9, r9
	subfic	r9, r9, 63
	srd	r3, r8, r9
	srd	r10, r10, r9
	rldicl	r10, r10, 0, 56
	rldicl	r3, r3, 0, 56
	subf	r3, r10, r3
	extsw	r3, r3
	blr

	.align	4
L(pagecross_check):
	subfic	r9, r9, 4096
	subfic	r7, r7, 4096
	cmpld	cr7, r7, r9
	bge	cr7, L(pagecross)
	mr	r7, r9

	/* If unaligned 16 bytes reads across a 4K page boundary, it uses
	   a simple byte a byte comparison until the page alignment for s1
	   is reached.  */
L(pagecross):
	add	r7, r3, r7
	subf	r9, r3, r7
	mtctr	r9

	.align	4
L(pagecross_loop):
	/* Loads a byte from s1 and s2, compare if *s1 is equal to *s2
	   and if *s1 is '\0'.  */
	lbz	r9, 0(r3)
	lbz	r10, 0(r4)
	addi	r3, r3, 1
	addi	r4, r4, 1
	cmplw	cr7, r9, r10
	cmpdi	cr5, r9, r0
	bne	cr7, L(pagecross_ne)
	beq	cr5, L(pagecross_nullfound)
	bdnz	L(pagecross_loop)
	b	L(align)

	.align	4
L(pagecross_ne):
	extsw	r3, r9
	mr	r9, r10
L(pagecross_retdiff):
	subf	r9, r9, r3
	extsw	r3, r9
	blr

	.align	4
L(pagecross_nullfound):
	li	r3, 0
	b	L(pagecross_retdiff)
END (STRCMP)
libc_hidden_builtin_def (strcmp)
