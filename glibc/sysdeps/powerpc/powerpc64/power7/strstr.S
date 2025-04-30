/* Optimized strstr implementation for PowerPC64/POWER7.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Char * [r3] strstr (char *s [r3], char * pat[r4])  */

/* The performance gain is obtained using aligned memory access, load
 * doubleword and usage of cmpb instruction for quicker comparison.  */

#define ITERATIONS	64

#ifndef STRSTR
# define STRSTR strstr
#endif

#ifndef STRLEN
/* For builds with no IFUNC support, local calls should be made to internal
   GLIBC symbol (created by libc_hidden_builtin_def).  */
# ifdef SHARED
#  define STRLEN   __GI_strlen
#  define STRLEN_is_local
# else
#  define STRLEN   strlen
# endif
#endif

#ifndef STRNLEN
/* For builds with no IFUNC support, local calls should be made to internal
   GLIBC symbol (created by libc_hidden_builtin_def).  */
# ifdef SHARED
#  define STRNLEN   __GI_strnlen
#  define STRNLEN_is_local
# else
#  define STRNLEN  __strnlen
# endif
#endif

#ifndef STRCHR
# ifdef SHARED
#  define STRCHR   __GI_strchr
#  define STRCHR_is_local
# else
#  define STRCHR   strchr
# endif
#endif

#define	FRAMESIZE	(FRAME_MIN_SIZE+32)
	.machine  power7
/* Can't be ENTRY_TOCLESS due to calling __strstr_ppc which uses r2.  */
ENTRY (STRSTR, 4)
	CALL_MCOUNT 2
	mflr	r0			/* Load link register LR to r0.  */
	std	r31, -8(r1)		/* Save callers register r31.  */
	std	r30, -16(r1)		/* Save callers register r30.  */
	std	r29, -24(r1)		/* Save callers register r29.  */
	std	r28, -32(r1)		/* Save callers register r28.  */
	std	r0, 16(r1)		/* Store the link register.  */
	cfi_offset(r31, -8)
	cfi_offset(r30, -16)
	cfi_offset(r28, -32)
	cfi_offset(r29, -24)
	cfi_offset(lr, 16)
	stdu	r1, -FRAMESIZE(r1)	/* Create the stack frame.  */
	cfi_adjust_cfa_offset(FRAMESIZE)

	dcbt	0, r3
	dcbt	0, r4
	cmpdi	cr7, r3, 0
	beq	cr7, L(retnull)
	cmpdi	cr7, r4, 0
	beq	cr7, L(retnull)

	mr	r29, r3
	mr	r30, r4
	mr	r3, r4
	bl	STRLEN
#ifndef STRLEN_is_local
	nop
#endif

	cmpdi	cr7, r3, 0	/* If search str is null.  */
	beq	cr7, L(ret_r3)

	mr	r31, r3
	mr	r4, r3
	mr	r3, r29
	bl	STRNLEN
#ifndef STRNLEN_is_local
	nop
#endif

	cmpd	cr7, r3, r31 	/* If len(r3) < len(r4).  */
	blt	cr7, L(retnull)
	mr	r3, r29
	lbz	r4, 0(r30)
	bl	STRCHR
#ifndef STRCHR_is_local
	nop
#endif

	mr	r11, r3
	/* If first char of search str is not present.  */
	cmpdi	cr7, r3, 0
	ble	cr7, L(end)
	/* Reg r28 is used to count the number of iterations. */
	li	r28, 0
	rldicl	r8, r3, 0, 52	/* Page cross check.  */
	cmpldi	cr7, r8, 4096-16
	bgt	cr7, L(bytebybyte)

	rldicl	r8, r30, 0, 52
	cmpldi	cr7, r8, 4096-16
	bgt	cr7, L(bytebybyte)

	/* If len(r4) < 8 handle in a different way.  */
	/* Shift position based on null and use cmpb.  */
	cmpdi	cr7, r31, 8
	blt	cr7, L(lessthan8)

	/* Len(r4) >= 8 reaches here.  */
	mr	r8, r3		/* Save r3 for future use.  */
	mr	r4, r30		/* Restore r4.  */
	li	r0, 0
	rlwinm	r10, r30, 3, 26, 28	/* Calculate padding in bits.  */
	clrrdi	r4, r4, 3	/* Make r4 aligned to 8.  */
	ld	r6, 0(r4)
	addi	r4, r4, 8
	cmpdi	cr7, r10, 0	/* Check if its already aligned?  */
	beq	cr7, L(begin1)
#ifdef __LITTLE_ENDIAN__
	srd	r6, r6, r10	/* Discard unwanted bits.  */
#else
	sld	r6, r6, r10
#endif
	ld	r9, 0(r4)
	subfic	r10, r10, 64
#ifdef __LITTLE_ENDIAN__
	sld	r9, r9, r10	/* Discard unwanted bits.  */
#else
	srd	r9, r9, r10
#endif
	or	r6, r6, r9	/* Form complete search str.  */
L(begin1):
	mr	r29, r6
	rlwinm	r10, r3, 3, 26, 28
	clrrdi	r3, r3, 3
	ld	r5, 0(r3)
	cmpb	r9, r0, r6	/* Check if input has null.  */
	cmpdi	cr7, r9, 0
	bne	cr7, L(return3)
	cmpb	r9, r0, r5	/* Check if input has null.  */
#ifdef __LITTLE_ENDIAN__
	srd	r9, r9, r10
#else
	sld	r9, r9, r10
#endif
	cmpdi	cr7, r9, 0
	bne	cr7, L(retnull)

	li	r12, -8		/* Shift values.  */
	li	r11, 72		/* Shift values.  */
	cmpdi	cr7, r10, 0
	beq	cr7, L(nextbyte1)
	mr	r12, r10
	addi	r12, r12, -8
	subfic	r11, r12, 64

L(nextbyte1):
	ldu	r7, 8(r3) 	/* Load next dw.  */
	addi	r12, r12, 8	/* Shift one byte and compare.  */
	addi	r11, r11, -8
#ifdef __LITTLE_ENDIAN__
	srd	r9, r5, r12	/* Rotate based on mask.  */
	sld	r10, r7, r11
#else
	sld	r9, r5, r12
	srd	r10, r7, r11
#endif
	/* Form single dw from few bytes on first load and second load.  */
	or	r10, r9, r10
	/* Check for null in the formed dw.  */
	cmpb	r9, r0, r10
	cmpdi	cr7, r9, 0
	bne	cr7, L(retnull)
	/* Cmpb search str and input str.  */
	cmpb	r9, r10, r6
	cmpdi	cr7, r9, -1
	beq	cr7, L(match)
	addi	r8, r8, 1
	b	L(begin)

	.align	4
L(match):
	/* There is a match of 8 bytes, check next bytes.  */
	cmpdi	cr7, r31, 8
	beq	cr7, L(return)
	/* Update next starting point r8.  */
	srdi	r9, r11, 3
	subf	r9, r9, r3
	mr	r8, r9

L(secondmatch):
	mr	r5, r7
	rlwinm	r10, r30, 3, 26, 28	/* Calculate padding in bits.  */
	ld	r6, 0(r4)
	addi	r4, r4, 8
	cmpdi	cr7, r10, 0	/* Check if its already aligned?  */
	beq	cr7, L(proceed3)
#ifdef __LITTLE_ENDIAN__
	srd	r6, r6, r10	/* Discard unwanted bits.  */
	cmpb	r9, r0, r6
	sld	r9, r9, r10
#else
	sld	r6, r6, r10
	cmpb	r9, r0, r6
	srd	r9, r9, r10
#endif
	cmpdi	cr7, r9, 0
	bne	cr7, L(proceed3)
	ld	r9, 0(r4)
	subfic	r10, r10, 64
#ifdef __LITTLE_ENDIAN__
	sld	r9, r9, r10	/* Discard unwanted bits.  */
#else
	srd	r9, r9, r10
#endif
	or	r6, r6, r9	/* Form complete search str.  */

L(proceed3):
	li	r7, 0
	addi	r3, r3, 8
	cmpb	r9, r0, r5
	cmpdi	cr7, r9, 0
	bne	cr7, L(proceed4)
	ld	r7, 0(r3)
L(proceed4):
#ifdef __LITTLE_ENDIAN__
	srd	r9, r5, r12
	sld	r10, r7, r11
#else
	sld	r9, r5, r12
	srd	r10, r7, r11
#endif
	/* Form single dw with few bytes from first and second load.  */
	or	r10, r9, r10
	cmpb	r9, r0, r6
	cmpdi	cr7, r9, 0
	bne	cr7, L(return4)
	/* Check for null in the formed dw.  */
	cmpb	r9, r0, r10
	cmpdi	cr7, r9, 0
	bne	cr7, L(retnull)
	/* If the next 8 bytes dont match, start search again.  */
	cmpb	r9, r10, r6
	cmpdi	cr7, r9, -1
	bne	cr7, L(reset)
	/* If the next 8 bytes match, load and compare next 8.  */
	b	L(secondmatch)

	.align	4
L(reset):
	/* Start the search again.  */
	addi	r8, r8, 1
	b	L(begin)

	.align	4
L(return3):
	/* Count leading zeros and compare partial dw.  */
#ifdef __LITTLE_ENDIAN__
	addi	r7, r9, -1
	andc	r7, r7, r9
	popcntd	r7, r7
	subfic	r7, r7, 64
	sld	r10, r5, r7
	sld	r6, r6, r7
#else
	cntlzd	r7, r9
	subfic	r7, r7, 64
	srd	r10, r5, r7
	srd	r6, r6, r7
#endif
	cmpb	r9, r10, r6
	cmpdi	cr7, r9, -1
	addi	r8, r8, 1
	/* Start search again if there is no match.  */
	bne	cr7, L(begin)
	/* If the words match, update return values.  */
	subfic	r7, r7, 64
	srdi	r7, r7, 3
	add	r3, r3, r7
	subf	r3, r31, r3
	b	L(end)

	.align	4
L(return4):
	/* Count leading zeros and compare partial dw.  */
#ifdef __LITTLE_ENDIAN__
	addi	r7, r9, -1
	andc	r7, r7, r9
	popcntd	r7, r7
	subfic	r7, r7, 64
	sld	r10, r10, r7
	sld	r6, r6, r7
#else
	cntlzd	r7, r9
	subfic	r7, r7, 64
	srd	r10, r10, r7
	srd	r6, r6, r7
#endif
	cmpb	r9, r10, r6
	cmpdi	cr7, r9, -1
	addi	r8, r8, 1
	bne	cr7, L(begin)
	subfic	r7, r7, 64
	srdi	r11, r11, 3
	subf	r3, r11, r3
	srdi	r7, r7, 3
	add	r3, r3, r7
	subf	r3, r31, r3
	b	L(end)

	.align	4
L(begin):
	mr	r3, r8
	/* When our iterations exceed ITERATIONS,fall back to default. */
	addi	r28, r28, 1
	cmpdi	cr7, r28, ITERATIONS
	beq	cr7, L(default)
	lbz	r4, 0(r30)
	bl	STRCHR
#ifndef STRCHR_is_local
	nop
#endif
	/* If first char of search str is not present.  */
	cmpdi	cr7, r3, 0
	ble	cr7, L(end)
	mr	r8, r3
	mr	r4, r30		/* Restore r4.  */
	li	r0, 0
	mr	r6, r29
	clrrdi	r4, r4, 3
	addi	r4, r4, 8
	b	L(begin1)

	/* Handle less than 8 search string.  */
	.align	4
L(lessthan8):
	mr	r4, r3
	mr	r9, r30
	li	r0, 0

	rlwinm	r10, r9, 3, 26, 28	/* Calculate padding in bits.  */
	srdi	r8, r10, 3	/* Padding in bytes.  */
	clrrdi	r9, r9, 3	/* Make r4 aligned to 8.  */
	ld	r6, 0(r9)
	cmpdi	cr7, r10, 0	/* Check if its already aligned?  */
	beq	cr7, L(proceed2)
#ifdef __LITTLE_ENDIAN__
	srd	r6, r6, r10	/* Discard unwanted bits.  */
#else
	sld	r6, r6, r10
#endif
	subfic	r8, r8, 8
	cmpd	cr7, r8, r31	/* Next load needed?  */
	bge	cr7, L(proceed2)
	ld	r7, 8(r9)
	subfic	r10, r10, 64
#ifdef __LITTLE_ENDIAN__
	sld	r7, r7, r10	/* Discard unwanted bits.  */
#else
	srd	r7, r7, r10
#endif
	or	r6, r6, r7	/* Form complete search str.  */
L(proceed2):
	mr	r29, r6
	rlwinm	r10, r3, 3, 26, 28
	clrrdi	r7, r3, 3	/* Make r3 aligned.  */
	ld	r5, 0(r7)
	sldi	r8, r31, 3
	subfic	r8, r8, 64
#ifdef __LITTLE_ENDIAN__
	sld	r6, r6, r8
	cmpb	r9, r0, r5
	srd	r9, r9, r10
#else
	srd	r6, r6, r8
	cmpb	r9, r0, r5
	sld	r9, r9, r10
#endif
	cmpdi	cr7, r9, 0
	bne	cr7, L(noload)
	cmpdi	cr7, r10, 0
	beq	cr7, L(continue)
	ld	r7, 8(r7)
L(continue1):
	mr	r12, r10
	addi	r12, r12, -8
	subfic	r11, r12, 64
	b	L(nextbyte)

	.align	4
L(continue):
	ld	r7, 8(r7)
	li	r12, -8		/* Shift values.  */
	li	r11, 72		/* Shift values.  */
L(nextbyte):
	addi	r12, r12, 8	/* Mask for rotation.  */
	addi	r11, r11, -8
#ifdef __LITTLE_ENDIAN__
	srd	r9, r5, r12
	sld	r10, r7, r11
	or	r10, r9, r10
	sld	r10, r10, r8
	cmpb	r9, r0, r10
	srd	r9, r9, r8
#else
	sld	r9, r5, r12
	srd	r10, r7, r11
	or	r10, r9, r10
	srd	r10, r10, r8
	cmpb	r9, r0, r10
	sld	r9, r9, r8
#endif
	cmpdi	cr7, r9, 0
	bne	cr7, L(retnull)
	cmpb	r9, r10, r6
	cmpdi	cr7, r9, -1
	beq	cr7, L(end)
	addi	r3, r4, 1
	/* When our iterations exceed ITERATIONS,fall back to default. */
	addi	r28, r28, 1
	cmpdi	cr7, r28, ITERATIONS
	beq	cr7, L(default)
	lbz	r4, 0(r30)
	bl	STRCHR
#ifndef STRCHR_is_local
	nop
#endif
	/* If first char of search str is not present.  */
	cmpdi	cr7, r3, 0
	ble	cr7, L(end)
	mr	r4, r3
	mr	r6, r29
	li	r0, 0
	b	L(proceed2)

	.align	4
L(noload):
	/* Reached null in r3, so skip next load.  */
	li 	r7, 0
	b	L(continue1)

	.align	4
L(return):
	/* Update return values.  */
	srdi	r9, r11, 3
	subf	r3, r9, r3
	b	L(end)

	/* Handling byte by byte.  */
	.align	4
L(bytebybyte):
	mr	r8, r3
	addi	r8, r8, -1
L(loop1):
	addi	r8, r8, 1
	mr	r3, r8
	mr	r4, r30
	lbz	r6, 0(r4)
	cmpdi	cr7, r6, 0
	beq	cr7, L(updater3)
L(loop):
	lbz	r5, 0(r3)
	cmpdi	cr7, r5, 0
	beq	cr7, L(retnull)
	cmpld	cr7, r6, r5
	bne	cr7, L(loop1)
	addi	r3, r3, 1
	addi	r4, r4, 1
	lbz	r6, 0(r4)
	cmpdi	cr7, r6, 0
	beq	cr7, L(updater3)
	b	L(loop)

	/* Handling return values.  */
	.align	4
L(updater3):
	subf	r3, r31, r3	/* Reduce len of r4 from r3.  */
	b	L(end)

	.align	4
L(ret_r3):
	mr	r3, r29		/* Return r3.  */
	b	L(end)

	.align	4
L(retnull):
	li	r3, 0		/* Return NULL.  */
	b	L(end)

	.align	4
L(default):
	mr	r4, r30
	bl	__strstr_ppc
	nop

	.align	4
L(end):
	addi	r1, r1, FRAMESIZE	/* Restore stack pointer.  */
	cfi_adjust_cfa_offset(-FRAMESIZE)
	ld	r0, 16(r1)	/* Restore the saved link register.  */
	ld	r28, -32(r1)	/* Restore callers save register r28.  */
	ld	r29, -24(r1)	/* Restore callers save register r29.  */
	ld	r30, -16(r1)	/* Restore callers save register r30.  */
	ld	r31, -8(r1)	/* Restore callers save register r31.  */
	mtlr	r0		/* Branch to link register.  */
	blr
END (STRSTR)
libc_hidden_builtin_def (strstr)
