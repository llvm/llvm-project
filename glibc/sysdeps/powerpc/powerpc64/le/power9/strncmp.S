/* Optimized strncmp implementation for PowerPC64/POWER9.
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

/* Implements the function

   int [r3] strncmp (const char *s1 [r3], const char *s2 [r4], size_t [r5] n)

   The implementation uses unaligned doubleword access to avoid specialized
   code paths depending of data alignment for first 32 bytes and uses
   vectorised loops after that.  */

#ifndef STRNCMP
# define STRNCMP strncmp
#endif

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
	cmplw	cr6, r5, r11; \
	ble	cr6, 2f; \
	addi	r6, reg2, 16; \
	lvx	v9, 0, r6; \
2: \
	vperm	reg1, v9, reg1, reg3;

/* TODO: change this to .machine power9 when minimum binutils
   is upgraded to 2.27.  */
	.machine  power7
ENTRY_TOCLESS (STRNCMP, 4)
	/* Check if size is 0.  */
	cmpdi	cr0, r5, 0
	beq	cr0, L(ret0)
	li	r0, 0

	/* Check if [s1]+32 or [s2]+32 will cross a 4K page boundary using
	   the code:

	    (((size_t) s1) % PAGE_SIZE > (PAGE_SIZE - ITER_SIZE))

	   with PAGE_SIZE being 4096 and ITER_SIZE begin 32.  */
	rldicl	r8, r3, 0, 52
	cmpldi	cr7, r8, 4096-32
	bgt	cr7, L(pagecross)
	rldicl	r9, r4, 0, 52
	cmpldi	cr7, r9, 4096-32
	bgt	cr7, L(pagecross)

	/* For short strings up to 32 bytes, load both s1 and s2 using
	   unaligned dwords and compare.  */

	ld	r7, 0(r3)
	ld	r9, 0(r4)
	li	r8, 0
	cmpb	r8, r7, r8
	cmpb	r6, r7, r9
	orc.	r8, r8, r6
	bne	cr0, L(different1)

	/* If the strings compared are equal, but size is less or equal
	   to 8, return 0.  */
	cmpldi	cr7, r5, 8
	li	r9, 0
	ble	cr7, L(ret1)
	addi	r5, r5, -8

	ld	r7, 8(r3)
	ld	r9, 8(r4)
	cmpb	r8, r7, r8
	cmpb	r6, r7, r9
	orc.	r8, r8, r6
	bne	cr0, L(different1)
	cmpldi	cr7, r5, 8
	mr	r9, r8
	ble	cr7, L(ret1)
	/* Update pointers and size.  */
	addi	r5, r5, -8
	addi	r3, r3, 16
	addi	r4, r4, 16

	ld	r7, 0(r3)
	ld	r9, 0(r4)
	li	r8, 0
	cmpb	r8, r7, r8
	cmpb	r6, r7, r9
	orc.	r8, r8, r6
	bne	cr0, L(different1)
	cmpldi	cr7, r5, 8
	li	r9, 0
	ble	cr7, L(ret1)
	addi	r5, r5, -8

	ld	r7, 8(r3)
	ld	r9, 8(r4)
	cmpb	r8, r7, r8
	cmpb	r6, r7, r9
	orc.	r8, r8, r6
	bne	cr0, L(different1)
	cmpldi	cr7, r5, 8
	mr	r9, r8
	ble	cr7, L(ret1)

	/* Update pointers and size.  */
	addi	r5, r5, -8
	addi	r3, r3, 16
	addi	r4, r4, 16
L(align):
	/* Now it has checked for first 32 bytes, align source1 to doubleword
	   and adjust source2 address.  */
	vspltisb	v0, 0
	vspltisb	v2, -1
	or	r6, r4, r3
	andi.	r6, r6, 0xF
	beq	cr0, L(aligned)
	lvsr	v6, 0, r4   /* Compute mask.  */
	clrldi	r6, r4, 60
	subfic	r11, r6, 16
	andi.	r6, r3, 0xF
	beq	cr0, L(s1_align)
	/* Both s1 and s2 are unaligned.  */
	GET16BYTES(v5, r4, v6)
	lvsr	v10, 0, r3   /* Compute mask.  */
	clrldi	r6, r3, 60
	subfic	r11, r6, 16
	GET16BYTES(v4, r3, v10)
	VCMPNEZB(v7, v5, v4)
	beq	cr6, L(match)
	b	L(different)

	/* Align s1 to qw and adjust s2 address.  */
	.align  4
L(match):
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	subf	r5, r11, r5
	add	r3, r3, r11
	add	r4, r4, r11
	andi.	r11, r4, 0xF
	beq	cr0, L(aligned)
	lvsr	v6, 0, r4
	clrldi	r6, r4, 60
	subfic	r11, r6, 16
	/* There are 2 loops depending on the input alignment.
	   Each loop gets 16 bytes from s1 and s2, checks for null
	   and compares them. Loops until a mismatch or  null occurs.  */
L(s1_align):
	lvx	v4, 0, r3
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16

	lvx	v4, 0, r3
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16

	lvx	v4, 0, r3
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16

	lvx	v4, 0, r3
	GET16BYTES(v5, r4, v6)
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16
	b	L(s1_align)
	.align  4
L(aligned):
	lvx	v4, 0, r3
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16

	lvx	v4, 0, r3
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16

	lvx	v4, 0, r3
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16

	lvx	v4, 0, r3
	lvx	v5, 0, r4
	VCMPNEZB(v7, v5, v4)
	bne	cr6, L(different)
	cmpldi	cr7, r5, 16
	ble	cr7, L(ret0)
	addi	r5, r5, -16
	addi	r3, r3, 16
	addi	r4, r4, 16
	b	L(aligned)
	/* Calculate and return the difference.  */
L(different):
	VCTZLSBB(r6, v7)
	cmplw	cr7, r5, r6
	ble	cr7, L(ret0)
	VEXTUBRX(r5, r6, v4)
	VEXTUBRX(r4, r6, v5)
	subf	r3, r4, r5
	extsw	r3, r3
	blr

	.align 4
L(ret0):
	li	r9, 0
L(ret1):
	mr	r3, r9
	blr

	/* The code now checks if r8 and r5 are different by issuing a
	   cmpb and shifts the result based on its output:

	  leadzero = (__builtin_ffsl (z1) - 1);
	  leadzero = leadzero > (n-1)*8 ? (n-1)*8 : leadzero;
	  r1 = (r1 >> leadzero) & 0xFFUL;
	  r2 = (r2 >> leadzero) & 0xFFUL;
	  return r1 - r2;  */

	.align 4
L(different1):
	neg	r11, r8
	sldi	r5, r5, 3
	and	r8, r11, r8
	addi	r5, r5, -8
	cntlzd	r8, r8
	subfic	r8, r8, 63
	extsw 	r8, r8
	cmpld	cr7, r8, r5
	ble	cr7, L(different2)
	mr	r8, r5
L(different2):
	extsw	r8, r8
	srd	r7, r7, r8
	srd	r9, r9, r8
	rldicl	r3, r7, 0, 56
	rldicl	r9, r9, 0, 56
	subf	r9, r9, 3
	extsw	r9, r9
	mr	r3, r9
	blr

	/* If unaligned 16 bytes reads across a 4K page boundary, it uses
	   a simple byte a byte comparison until the page alignment for s1
	   is reached.  */
	.align 4
L(pagecross):
	lbz	r7, 0(r3)
	lbz	r9, 0(r4)
	subfic	r8, r8,4095
	cmplw	cr7, r9, r7
	bne	cr7, L(byte_ne_3)
	cmpdi	cr7, r9, 0
	beq	cr7, L(byte_ne_0)
	addi	r5, r5, -1
	subf	r7, r8, r5
	subf	r9, r7, r5
	addi	r9, r9, 1
	mtctr	r9
	b	L(pagecross_loop1)

	.align 4
L(pagecross_loop0):
	beq	cr7, L(ret0)
	lbz	r9, 0(r3)
	lbz	r8, 0(r4)
	addi	r5, r5, -1
	cmplw	cr7, r9, r8
	cmpdi	cr5, r9, 0
	bne	cr7, L(byte_ne_2)
	beq	cr5, L(byte_ne_0)
L(pagecross_loop1):
	cmpdi	cr7, r5, 0
	addi	r3, r3, 1
	addi	r4, r4, 1
	bdnz	L(pagecross_loop0)
	cmpdi	cr7, r7, 0
	li	r9, 0
	bne+	cr7, L(align)
	b	L(ret1)

	.align 4
L(byte_ne_0):
	li	r7, 0
L(byte_ne_1):
	subf	r9, r9, r7
	extsw	r9, r9
	b	L(ret1)

	.align 4
L(byte_ne_2):
	extsw	r7, r9
	mr	r9, r8
	b	L(byte_ne_1)
L(byte_ne_3):
	extsw	r7, r7
	b	L(byte_ne_1)
END(STRNCMP)
libc_hidden_builtin_def(strncmp)
