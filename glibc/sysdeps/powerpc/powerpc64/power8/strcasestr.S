/* Optimized strcasestr implementation for PowerPC64/POWER8.
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
#include <locale-defines.h>

/* Char * [r3] strcasestr (char *s [r3], char * pat[r4])  */

/* The performance gain is obtained by comparing 16 bytes.  */

/* When the first char of r4 is hit ITERATIONS times in r3
   fallback to default.  */
#define ITERATIONS	64

#ifndef STRCASESTR
# define STRCASESTR __strcasestr
#endif

#ifndef STRLEN
/* For builds without IFUNC support, local calls should be made to internal
   GLIBC symbol (created by libc_hidden_builtin_def).  */
# ifdef SHARED
#  define STRLEN   __GI_strlen
# else
#  define STRLEN   strlen
# endif
#endif

#ifndef STRNLEN
/* For builds without IFUNC support, local calls should be made to internal
   GLIBC symbol (created by libc_hidden_builtin_def).  */
# ifdef SHARED
#  define STRNLEN   __GI_strnlen
# else
#  define STRNLEN    __strnlen
# endif
#endif

#ifndef STRCHR
# ifdef SHARED
#  define STRCHR   __GI_strchr
# else
#  define STRCHR   strchr
# endif
#endif

/* Convert 16 bytes of v4 and reg to lowercase and compare.  */
#define TOLOWER(reg)     \
	vcmpgtub	v6, v4, v1; \
	vcmpgtub	v7, v2, v4; \
	vand	v8, v7, v6; \
	vand	v8, v8, v3; \
	vor	v4, v8, v4; \
	vcmpgtub	v6, reg, v1; \
	vcmpgtub	v7, v2, reg; \
	vand	v8, v7, v6; \
	vand	v8, v8, v3; \
	vor	reg, v8, reg; \
	vcmpequb.	v6, reg, v4;

#define	FRAMESIZE	(FRAME_MIN_SIZE+48)
	.machine  power8
ENTRY (STRCASESTR, 4)
	CALL_MCOUNT 2
	mflr	r0			/* Load link register LR to r0.  */
	std	r31, -8(r1)		/* Save callers register r31.  */
	std	r30, -16(r1)		/* Save callers register r30.  */
	std	r29, -24(r1)		/* Save callers register r29.  */
	std	r28, -32(r1)		/* Save callers register r28.  */
	std	r27, -40(r1)		/* Save callers register r27.  */
	std	r0, 16(r1)		/* Store the link register.  */
	cfi_offset(r31, -8)
	cfi_offset(r30, -16)
	cfi_offset(r29, -24)
	cfi_offset(r28, -32)
	cfi_offset(r27, -40)
	cfi_offset(lr, 16)
	stdu	r1, -FRAMESIZE(r1)	/* Create the stack frame.  */
	cfi_adjust_cfa_offset(FRAMESIZE)

	dcbt	0, r3
	dcbt	0, r4
	cmpdi	cr7, r3, 0		/* Input validation.  */
	beq	cr7, L(retnull)
	cmpdi	cr7, r4, 0
	beq	cr7, L(retnull)

	mr	r29, r3
	mr	r30, r4
	/* Load first byte from r4 and check if its null.  */
	lbz	r6, 0(r4)
	cmpdi	cr7, r6, 0
	beq	cr7, L(ret_r3)

	ld	r10, __libc_tsd_LOCALE@got@tprel(r2)
	add	r9, r10, __libc_tsd_LOCALE@tls
	ld	r9, 0(r9)
	ld	r9, LOCALE_CTYPE_TOUPPER(r9)
	sldi	r10, r6, 2		/* Convert to upper case.  */
	lwzx	r28, r9, r10

	ld	r10, __libc_tsd_LOCALE@got@tprel(r2)
	add	r11, r10, __libc_tsd_LOCALE@tls
	ld	r11, 0(r11)
	ld	r11, LOCALE_CTYPE_TOLOWER(r11)
	sldi	r10, r6, 2              /* Convert to lower case.  */
	lwzx	r27, r11, r10

	/* Check if the first char is present.  */
	mr	r4, r27
	bl	STRCHR
	nop
	mr	r5, r3
	mr	r3, r29
	mr	r29, r5
	mr	r4, r28
	bl	STRCHR
	nop
	cmpdi	cr7, r29, 0
	beq	cr7, L(firstpos)
	cmpdi	cr7, r3, 0
	beq	cr7, L(skipcheck)
	cmpw	cr7, r3, r29
	ble 	cr7, L(firstpos)
	/* Move r3 to the first occurence.  */
L(skipcheck):
	mr	r3, r29
L(firstpos):
	mr	r29, r3

	sldi	r9, r27, 8
	or	r28, r9, r28
	/* Reg r27 is used to count the number of iterations.  */
	li	r27, 0
	/* If first char of search str is not present.  */
	cmpdi	cr7, r3, 0
	ble	cr7, L(end)

	/* Find the length of pattern.  */
	mr	r3, r30
	bl	STRLEN
	nop

	cmpdi	cr7, r3, 0	/* If search str is null.  */
	beq	cr7, L(ret_r3)

	mr	r31, r3
	mr	r4, r3
	mr	r3, r29
	bl	STRNLEN
	nop

	cmpd	cr7, r3, r31 	/* If len(r3) < len(r4).  */
	blt	cr7, L(retnull)

	mr	r3, r29

	/* Locales not matching ASCII for single bytes.  */
	ld	r10, __libc_tsd_LOCALE@got@tprel(r2)
	add	r9, r10, __libc_tsd_LOCALE@tls
	ld	r9, 0(r9)
	ld	r7, 0(r9)
	addi	r7, r7, LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES
	lwz	r8, 0(r7)
	cmpdi	cr7, r8, 1
	beq	cr7, L(bytebybyte)

	/* If len(r4) < 16 handle byte by byte.  */
	/* For shorter strings we will not use vector registers.  */
	cmpdi	cr7, r31, 16
	blt	cr7, L(bytebybyte)

	/* Comparison values used for TOLOWER.  */
	/* Load v1 = 64('A' - 1), v2 = 91('Z' + 1), v3 = 32 in each byte.  */
	vspltish	v0, 0
	vspltisb	v5, 2
	vspltisb	v4, 4
	vsl	v3, v5, v4
	vaddubm	v1, v3, v3
	vspltisb	v5, 15
	vaddubm	v2, v5, v5
	vaddubm	v2, v1, v2
	vspltisb	v4, -3
	vaddubm	v2, v2, v4

	/*
	1. Load 16 bytes from r3 and r4
	2. Check if there is null, If yes, proceed byte by byte path.
	3. Else,Convert both to lowercase and compare.
	4. If they are same proceed to 1.
	5. If they dont match, find if first char of r4 is present in the
	   loaded 16 byte of r3.
	6. If yes, move position, load next 16 bytes of r3 and proceed to 2.
	*/

	mr	r8, r3		/* Save r3 for future use.  */
	mr	r4, r30		/* Restore r4.  */
	clrldi	r10, r4, 60
	lvx	v5, 0, r4	/* Load 16 bytes from r4.  */
	cmpdi	cr7, r10, 0
	beq	cr7, L(begin2)
	/* If r4 is unaligned, load another 16 bytes.  */
#ifdef __LITTLE_ENDIAN__
	lvsr	v7, 0, r4
#else
	lvsl	v7, 0, r4
#endif
	addi	r5, r4, 16
	lvx	v9, 0, r5
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v9, v5, v7
#else
	vperm	v5, v5, v9, v7
#endif
L(begin2):
	lvx	v4, 0, r3
	vcmpequb.	v7, v0, v4	/* Check for null.  */
	beq	cr6, L(nullchk6)
	b	L(trailcheck)

        .align  4
L(nullchk6):
	clrldi	r10, r3, 60
	cmpdi	cr7, r10, 0
	beq	cr7, L(next16)
#ifdef __LITTLE_ENDIAN__
	lvsr	v7, 0, r3
#else
	lvsl	v7, 0, r3
#endif
	addi	r5, r3, 16
	/* If r3 is unaligned, load another 16 bytes.  */
	lvx	v10, 0, r5
#ifdef __LITTLE_ENDIAN__
	vperm	v4, v10, v4, v7
#else
	vperm	v4, v4, v10, v7
#endif
L(next16):
	vcmpequb.	v6, v0, v5	/* Check for null.  */
	beq	cr6, L(nullchk)
	b	L(trailcheck)

	.align	4
L(nullchk):
	vcmpequb.	v6, v0, v4
	beq	cr6, L(nullchk1)
	b	L(retnull)

	.align	4
L(nullchk1):
	/* Convert both v3 and v4 to lower.  */
	TOLOWER(v5)
	/* If both are same, branch to match.  */
	blt	cr6, L(match)
	/* Find if the first char is present in next 15 bytes.  */
#ifdef __LITTLE_ENDIAN__
	vspltb	v6, v5, 15
	vsldoi	v7, v0, v4, 15
#else
	vspltb	v6, v5, 0
	vspltisb	v7, 8
	vslo	v7, v4, v7
#endif
	vcmpequb	v7, v6, v7
	vcmpequb.	v6, v0, v7
	/* Shift r3 by 16 bytes and proceed.  */
	blt	cr6, L(shift16)
	vclzd	v8, v7
#ifdef __LITTLE_ENDIAN__
	vspltb	v6, v8, 15
#else
	vspltb	v6, v8, 7
#endif
	vcmpequb.	v6, v6, v1
	/* Shift r3 by 8  bytes and proceed.  */
	blt	cr6, L(shift8)
	b	L(begin)

	.align	4
L(match):
	/* There is a match of 16 bytes, check next bytes.  */
	cmpdi	cr7, r31, 16
	mr	r29, r3
	beq	cr7, L(ret_r3)

L(secondmatch):
	addi	r3, r3, 16
	addi	r4, r4, 16
	/* Load next 16 bytes of r3 and r4 and compare.  */
	clrldi	r10, r4, 60
	cmpdi	cr7, r10, 0
	beq	cr7, L(nextload)
	/* Handle unaligned case.  */
	vor	v6, v9, v9
	vcmpequb.	v7, v0, v6
	beq	cr6, L(nullchk2)
	b	L(trailcheck)

	.align	4
L(nullchk2):
#ifdef __LITTLE_ENDIAN__
	lvsr	v7, 0, r4
#else
	lvsl	v7, 0, r4
#endif
	addi	r5, r4, 16
	/* If r4 is unaligned, load another 16 bytes.  */
	lvx	v9, 0, r5
#ifdef __LITTLE_ENDIAN__
	vperm	v11, v9, v6, v7
#else
	vperm	v11, v6, v9, v7
#endif
	b	L(compare)

	.align	4
L(nextload):
	lvx	v11, 0, r4
L(compare):
	vcmpequb.	v7, v0, v11
	beq	cr6, L(nullchk3)
	b	L(trailcheck)

	.align	4
L(nullchk3):
	clrldi	r10, r3, 60
	cmpdi 	cr7, r10, 0
	beq 	cr7, L(nextload1)
	/* Handle unaligned case.  */
	vor	v4, v10, v10
	vcmpequb.	v7, v0, v4
	beq	cr6, L(nullchk4)
	b	L(retnull)

	.align	4
L(nullchk4):
#ifdef __LITTLE_ENDIAN__
	lvsr	v7, 0, r3
#else
	lvsl	v7, 0, r3
#endif
	addi	r5, r3, 16
	/* If r3 is unaligned, load another 16 bytes.  */
	lvx	v10, 0, r5
#ifdef __LITTLE_ENDIAN__
	vperm	v4, v10, v4, v7
#else
	vperm	v4, v4, v10, v7
#endif
	b	L(compare1)

	.align	4
L(nextload1):
	lvx	v4, 0, r3
L(compare1):
	vcmpequb.	v7, v0, v4
	beq	cr6, L(nullchk5)
	b	L(retnull)

	.align	4
L(nullchk5):
	/* Convert both v3 and v4 to lower.  */
	TOLOWER(v11)
	/* If both are same, branch to secondmatch.  */
	blt 	cr6, L(secondmatch)
	/* Continue the search.  */
        b	L(begin)

	.align	4
L(trailcheck):
	ld	r10, __libc_tsd_LOCALE@got@tprel(r2)
	add	r11, r10, __libc_tsd_LOCALE@tls
	ld	r11, 0(r11)
	ld	r11, LOCALE_CTYPE_TOLOWER(r11)
L(loop2):
	lbz	r5, 0(r3)               /* Load byte from r3.  */
	lbz	r6, 0(r4)               /* Load next byte from r4.  */
	cmpdi 	cr7, r6, 0              /* Is it null?  */
	beq 	cr7, L(updater3)
	cmpdi 	cr7, r5, 0              /* Is it null?  */
	beq 	cr7, L(retnull)         /* If yes, return.  */
	addi	r3, r3, 1
	addi	r4, r4, 1               /* Increment r4.  */
	sldi	r10, r5, 2              /* Convert to lower case.  */
	lwzx	r10, r11, r10
	sldi	r7, r6, 2               /* Convert to lower case.  */
	lwzx	r7, r11, r7
	cmpw	cr7, r7, r10            /* Compare with byte from r4.  */
	bne	cr7, L(begin)
	b	L(loop2)

	.align	4
L(shift8):
	addi	r8, r8, 7
	b	L(begin)
	.align	4
L(shift16):
	addi	r8, r8, 15
	.align	4
L(begin):
	addi	r8, r8, 1
	mr	r3, r8
	/* When our iterations exceed ITERATIONS,fall back to default.  */
	addi	r27, r27, 1
	cmpdi	cr7, r27, ITERATIONS
	beq	cr7, L(default)
	mr	r4, r30         /* Restore r4.  */
	b	L(begin2)

	/* Handling byte by byte.  */
	.align	4
L(loop1):
	mr	r3, r8
	addi	r27, r27, 1
	cmpdi	cr7, r27, ITERATIONS
	beq	cr7, L(default)
	mr	r29, r8
	srdi	r4, r28, 8
	/* Check if the first char is present.  */
	bl	STRCHR
	nop
	mr	r5, r3
	mr	r3, r29
	mr	r29, r5
	sldi	r4, r28, 56
	srdi	r4, r4, 56
	bl	STRCHR
	nop
	cmpdi	cr7, r29, 0
	beq	cr7, L(nextpos)
	cmpdi	cr7, r3, 0
	beq	cr7, L(skipcheck1)
	cmpw	cr7, r3, r29
	ble 	cr7, L(nextpos)
	/* Move r3 to first occurence.  */
L(skipcheck1):
	mr	r3, r29
L(nextpos):
	mr	r29, r3
	cmpdi 	cr7, r3, 0
	ble 	cr7, L(retnull)
L(bytebybyte):
	ld	r10, __libc_tsd_LOCALE@got@tprel(r2)
	add	r11, r10, __libc_tsd_LOCALE@tls
	ld	r11, 0(r11)
	ld	r11, LOCALE_CTYPE_TOLOWER(r11)
	mr	r4, r30                 /* Restore r4.  */
	mr	r8, r3                  /* Save r3.  */
	addi	r8, r8, 1

L(loop):
	addi	r3, r3, 1
	lbz	r5, 0(r3)               /* Load byte from r3.  */
	addi	r4, r4, 1               /* Increment r4.  */
	lbz	r6, 0(r4)               /* Load next byte from r4.  */
	cmpdi 	cr7, r6, 0              /* Is it null?  */
	beq 	cr7, L(updater3)
	cmpdi 	cr7, r5, 0              /* Is it null?  */
	beq 	cr7, L(retnull)         /* If yes, return.  */
	sldi	r10, r5, 2              /* Convert to lower case.  */
	lwzx	r10, r11, r10
	sldi	r7, r6, 2               /* Convert to lower case.  */
	lwzx	r7, r11, r7
	cmpw	cr7, r7, r10            /* Compare with byte from r4.  */
	bne 	cr7, L(loop1)
	b	L(loop)

	/* Handling return values.  */
	.align	4
L(updater3):
	subf	r3, r31, r3	/* Reduce r31 (len of r4) from r3.  */
	b	L(end)

	.align	4
L(ret_r3):
	mr	r3, r29		/* Return point of match.  */
	b	L(end)

	.align	4
L(retnull):
	li	r3, 0		/* Substring was not found.  */
	b	L(end)

	.align	4
L(default):
	mr	r4, r30
	bl	__strcasestr_ppc
	nop

	.align	4
L(end):
	addi	r1, r1, FRAMESIZE	/* Restore stack pointer.  */
	cfi_adjust_cfa_offset(-FRAMESIZE)
	ld	r0, 16(r1)	/* Restore the saved link register.  */
	ld	r27, -40(r1)
	ld	r28, -32(r1)
	ld	r29, -24(r1)	/* Restore callers save register r29.  */
	ld	r30, -16(r1)	/* Restore callers save register r30.  */
	ld	r31, -8(r1)	/* Restore callers save register r31.  */
	cfi_restore(lr)
	cfi_restore(r27)
	cfi_restore(r28)
	cfi_restore(r29)
	cfi_restore(r30)
	cfi_restore(r31)
	mtlr	r0		/* Branch to link register.  */
	blr
END (STRCASESTR)

weak_alias (__strcasestr, strcasestr)
libc_hidden_def (__strcasestr)
libc_hidden_builtin_def (strcasestr)
