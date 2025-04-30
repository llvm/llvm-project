/* Optimized strcasecmp implementation for PowerPC64.
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

/* int [r3] strcasecmp (const char *s1 [r3], const char *s2 [r4] ) */

#ifndef USE_AS_STRNCASECMP
#  define __STRCASECMP __strcasecmp
#  define STRCASECMP   strcasecmp
#else
#  define __STRCASECMP __strncasecmp
#  define STRCASECMP   strncasecmp
#endif
/* Convert 16 bytes to lowercase and compare */
#define TOLOWER()     \
	vaddubm	v8, v4, v1; \
	vaddubm	v7, v4, v3; \
	vcmpgtub	v8, v8, v2; \
	vsel	v4, v7, v4, v8; \
	vaddubm	v8, v5, v1; \
	vaddubm	v7, v5, v3; \
	vcmpgtub	v8, v8, v2; \
	vsel	v5, v7, v5, v8; \
	vcmpequb.	v7, v5, v4;

/*
 * Get 16 bytes for unaligned case.
 * reg1: Vector to hold next 16 bytes.
 * reg2: Address to read from.
 * reg3: Permute control vector.
 * v8: Tmp vector used to mask unwanted bytes.
 * v9: Tmp vector,0 when null is found on first 16 bytes
 */
#ifdef __LITTLE_ENDIAN__
#define GET16BYTES(reg1, reg2, reg3) \
	lvx	reg1, 0, reg2; \
	vspltisb	v8, -1; \
	vperm	v8, v8, reg1, reg3; \
	vcmpequb.	v8, v0, v8; \
	beq	cr6, 1f; \
	vspltisb	v9, 0; \
	b	2f; \
	.align 4; \
1: \
	addi	r6, reg2, 16; \
	lvx	v9, 0, r6; \
2: \
	vperm	reg1, v9, reg1, reg3;
#else
#define GET16BYTES(reg1, reg2, reg3) \
	lvx	reg1, 0, reg2; \
	vspltisb	 v8, -1; \
	vperm	v8, reg1, v8,  reg3; \
	vcmpequb.	v8, v0, v8; \
	beq	cr6, 1f; \
	vspltisb	v9, 0; \
	b	2f; \
	.align 4; \
1: \
	addi	r6, reg2, 16; \
	lvx	v9, 0, r6; \
2: \
	vperm	reg1, reg1, v9, reg3;
#endif

/* Check null in v4, v5 and convert to lower.  */
#define CHECKNULLANDCONVERT() \
	vcmpequb.	v7, v0, v5; \
	beq	cr6, 3f; \
	vcmpequb.	v7, v0, v4; \
	beq	cr6, 3f; \
	b	L(null_found); \
	.align  4; \
3: \
	TOLOWER()

	.machine  power8

ENTRY (__STRCASECMP)
#ifdef USE_AS_STRNCASECMP
	CALL_MCOUNT 3
#else
	CALL_MCOUNT 2
#endif
#define rRTN	r3	/* Return value */
#define rSTR1	r10	/* 1st string */
#define rSTR2	r4	/* 2nd string */
#define rCHAR1	r6	/* Byte read from 1st string */
#define rCHAR2	r7	/* Byte read from 2nd string */
#define rADDR1	r8	/* Address of tolower(rCHAR1) */
#define rADDR2	r12	/* Address of tolower(rCHAR2) */
#define rLWR1	r8	/* Word tolower(rCHAR1) */
#define rLWR2	r12	/* Word tolower(rCHAR2) */
#define rTMP	r9
#define rLOC	r11	/* Default locale address */

	cmpd	cr7, rRTN, rSTR2

	/* Get locale address.  */
	ld 	rTMP, __libc_tsd_LOCALE@got@tprel(r2)
	add 	rLOC, rTMP, __libc_tsd_LOCALE@tls
	ld	rLOC, 0(rLOC)

	mr	rSTR1, rRTN
	li	rRTN, 0
	beqlr	cr7
#ifdef USE_AS_STRNCASECMP
	cmpdi	cr7, r5, 0
	beq	cr7, L(retnull)
	cmpdi	cr7, r5, 16
	blt	cr7, L(bytebybyte)
#endif
	vspltisb	v0, 0
	vspltisb	v8, -1
	/* Check for null in initial characters.
	   Check max of 16 char depending on the alignment.
	   If null is present, proceed byte by byte.  */
	lvx	v4, 0, rSTR1
#ifdef  __LITTLE_ENDIAN__
	lvsr	v10, 0, rSTR1	/* Compute mask.  */
	vperm	v9, v8, v4, v10	/* Mask bits that are not part of string.  */
#else
	lvsl	v10, 0, rSTR1
	vperm	v9, v4, v8, v10
#endif
	vcmpequb.	v9, v0, v9	/* Check for null bytes.  */
	bne	cr6, L(bytebybyte)
	lvx	v5, 0, rSTR2
	/* Calculate alignment.  */
#ifdef __LITTLE_ENDIAN__
	lvsr	v6, 0, rSTR2
	vperm	v9, v8, v5, v6	/* Mask bits that are not part of string.  */
#else
	lvsl	v6, 0, rSTR2
	vperm	v9, v5, v8, v6
#endif
	vcmpequb.	v9, v0, v9	/* Check for null bytes.  */
	bne	cr6, L(bytebybyte)
	/* Check if locale has non ascii characters.  */
	ld	rTMP, 0(rLOC)
	addi r6, rTMP,LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES
	lwz	rTMP, 0(r6)
	cmpdi	cr7, rTMP, 1
	beq	cr7, L(bytebybyte)

	/* Load vector registers with values used for TOLOWER.  */
	/* Load v1 = 0xbf, v2 = 0x19 v3 = 0x20 in each byte.  */
	vspltisb	v3, 2
	vspltisb	v9, 4
	vsl	v3, v3, v9
	vaddubm	v1, v3, v3
	vnor	v1, v1, v1
	vspltisb	v2, 7
	vsububm	v2, v3, v2

	andi.	rADDR1, rSTR1, 0xF
	beq	cr0, L(align)
	addi	r6, rSTR1, 16
	lvx	v9, 0, r6
	/* Compute 16 bytes from previous two loads.  */
#ifdef __LITTLE_ENDIAN__
	vperm	v4, v9, v4, v10
#else
	vperm	v4, v4, v9, v10
#endif
L(align):
	andi.	rADDR2, rSTR2, 0xF
	beq	cr0, L(align1)
	addi	r6, rSTR2, 16
	lvx	v9, 0, r6
	/* Compute 16 bytes from previous two loads.  */
#ifdef __LITTLE_ENDIAN__
	vperm	v5, v9, v5, v6
#else
	vperm	v5, v5, v9, v6
#endif
L(align1):
	CHECKNULLANDCONVERT()
	blt	cr6, L(match)
	b	L(different)
	.align 	4
L(match):
	clrldi	r6, rSTR1, 60
	subfic	r7, r6, 16
#ifdef USE_AS_STRNCASECMP
	sub	r5, r5, r7
#endif
	add	rSTR1, rSTR1, r7
	add	rSTR2, rSTR2, r7
	andi.	rADDR2, rSTR2, 0xF
	addi	rSTR1, rSTR1, -16
	addi	rSTR2, rSTR2, -16
	beq	cr0, L(aligned)
#ifdef __LITTLE_ENDIAN__
	lvsr	v6, 0, rSTR2
#else
	lvsl	v6, 0, rSTR2
#endif
	/* There are 2 loops depending on the input alignment.
	   Each loop gets 16 bytes from s1 and s2, check for null,
	   convert to lowercase and compare. Loop till difference
	   or null occurs. */
L(s1_align):
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
#ifdef USE_AS_STRNCASECMP
	cmpdi	cr7, r5, 16
	blt	cr7, L(bytebybyte)
	addi	r5, r5, -16
#endif
	lvx	v4, 0, rSTR1
	GET16BYTES(v5, rSTR2, v6)
	CHECKNULLANDCONVERT()
	blt	cr6, L(s1_align)
	b	L(different)
	.align 	4
L(aligned):
	addi	rSTR1, rSTR1, 16
	addi	rSTR2, rSTR2, 16
#ifdef USE_AS_STRNCASECMP
	cmpdi	cr7, r5, 16
	blt	cr7, L(bytebybyte)
	addi	r5, r5, -16
#endif
	lvx	v4, 0, rSTR1
	lvx	v5, 0, rSTR2
	CHECKNULLANDCONVERT()
	blt	cr6, L(aligned)

	/* Calculate and return the difference. */
L(different):
	vaddubm	v1, v3, v3
	vcmpequb	v7, v0, v7
#ifdef __LITTLE_ENDIAN__
	/* Count trailing zero.  */
	vspltisb	v8, -1
	vadduqm	v9, v7, v8
	vandc	v8, v9, v7
	vpopcntd	v8, v8
	vspltb	v6, v8, 15
	vcmpequb.	v6, v6, v1
	blt	cr6, L(shift8)
#else
	/* Count leading zero.  */
	vclzd	v8, v7
	vspltb	v6, v8, 7
	vcmpequb.	v6, v6, v1
	blt	cr6, L(shift8)
	vsro	v8, v8, v1
#endif
	b	L(skipsum)
	.align  4
L(shift8):
	vsumsws		v8, v8, v0
L(skipsum):
#ifdef __LITTLE_ENDIAN__
	/* Shift registers based on leading zero count.  */
	vsro	v6, v5, v8
	vsro	v7, v4, v8
	/* Merge and move to GPR.  */
	vmrglb	v6, v6, v7
	vslo	v1, v6, v1
	mfvrd	r3, v1
	/* Place the characters that are different in first position.  */
	sldi	rSTR2, rRTN, 56
	srdi	rSTR2, rSTR2, 56
	sldi	rSTR1, rRTN, 48
	srdi	rSTR1, rSTR1, 56
#else
	vslo	v6, v5, v8
	vslo	v7, v4, v8
	vmrghb	v1, v6, v7
	mfvrd	r3, v1
	srdi	rSTR2, rRTN, 48
	sldi	rSTR2, rSTR2, 56
	srdi	rSTR2, rSTR2, 56
	srdi	rSTR1, rRTN, 56
#endif
	subf  	rRTN, rSTR1, rSTR2
	extsw 	rRTN, rRTN
	blr

	.align  4
	/* OK. We've hit the end of the string. We need to be careful that
	   we don't compare two strings as different because of junk beyond
	   the end of the strings...  */
L(null_found):
	vaddubm	v10, v3, v3
#ifdef __LITTLE_ENDIAN__
	/* Count trailing zero.  */
	vspltisb	v8, -1
	vadduqm	v9, v7, v8
	vandc	v8, v9, v7
	vpopcntd	v8, v8
	vspltb	v6, v8, 15
	vcmpequb.	v6, v6, v10
	blt	cr6, L(shift_8)
#else
	/* Count leading zero.  */
	vclzd	v8, v7
	vspltb	v6, v8, 7
	vcmpequb.	v6, v6, v10
	blt	cr6, L(shift_8)
	vsro	v8, v8, v10
#endif
	b	L(skipsum1)
	.align  4
L(shift_8):
	vsumsws	v8, v8, v0
L(skipsum1):
	/* Calculate shift count based on count of zero.  */
	vspltisb	v10, 7
	vslb	v10, v10, v10
	vsldoi	v9, v0, v10, 1
	vsubudm	v9, v9, v8
	vspltisb	v8, 8
	vsldoi	v8, v0, v8, 1
	vsubudm	v9, v9, v8
	/* Shift and remove junk after null character.  */
#ifdef __LITTLE_ENDIAN__
	vslo	v5, v5, v9
	vslo	v4, v4, v9
#else
	vsro	v5, v5, v9
	vsro	v4, v4, v9
#endif
	/* Convert and compare 16 bytes.  */
	TOLOWER()
	blt	cr6, L(retnull)
	b	L(different)
	.align  4
L(retnull):
	li	rRTN, 0
	blr
	.align  4
L(bytebybyte):
	/* Unrolling loop for POWER: loads are done with 'lbz' plus
	offset and string descriptors are only updated in the end
	of loop unrolling. */
	ld	rLOC, LOCALE_CTYPE_TOLOWER(rLOC)
	lbz	rCHAR1, 0(rSTR1)	/* Load char from s1 */
	lbz	rCHAR2, 0(rSTR2)	/* Load char from s2 */
#ifdef USE_AS_STRNCASECMP
	rldicl	rTMP, r5, 62, 2
	cmpdi	cr7, rTMP, 0
	beq	cr7, L(lessthan4)
	mtctr	rTMP
#endif
L(loop):
	cmpdi	rCHAR1, 0		/* *s1 == '\0' ? */
	sldi	rADDR1, rCHAR1, 2	/* Calculate address for tolower(*s1) */
	sldi	rADDR2, rCHAR2, 2	/* Calculate address for tolower(*s2) */
	lwzx	rLWR1, rLOC, rADDR1	/* Load tolower(*s1) */
	lwzx	rLWR2, rLOC, rADDR2	/* Load tolower(*s2) */
	cmpw	cr1, rLWR1, rLWR2	/* r = tolower(*s1) == tolower(*s2) ? */
	crorc	4*cr1+eq,eq,4*cr1+eq	/* (*s1 != '\0') || (r == 1) */
	beq	cr1, L(done)
	lbz	rCHAR1, 1(rSTR1)
	lbz	rCHAR2, 1(rSTR2)
	cmpdi	rCHAR1, 0
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpw	cr1, rLWR1, rLWR2
	crorc	4*cr1+eq,eq,4*cr1+eq
	beq	cr1, L(done)
	lbz	rCHAR1, 2(rSTR1)
	lbz	rCHAR2, 2(rSTR2)
	cmpdi	rCHAR1, 0
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpw	cr1, rLWR1, rLWR2
	crorc	4*cr1+eq,eq,4*cr1+eq
	beq	cr1, L(done)
	lbz	rCHAR1, 3(rSTR1)
	lbz	rCHAR2, 3(rSTR2)
	cmpdi	rCHAR1, 0
	/* Increment both string descriptors */
	addi	rSTR1, rSTR1, 4
	addi	rSTR2, rSTR2, 4
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpw	cr1, rLWR1, rLWR2
	crorc	4*cr1+eq,eq,4*cr1+eq
	beq     cr1, L(done)
	lbz	rCHAR1, 0(rSTR1)	/* Load char from s1 */
	lbz	rCHAR2, 0(rSTR2)	/* Load char from s2 */
#ifdef USE_AS_STRNCASECMP
	bdnz	L(loop)
#else
	b	L(loop)
#endif
#ifdef USE_AS_STRNCASECMP
L(lessthan4):
	clrldi	r5, r5, 62
	cmpdi	cr7, r5, 0
	beq	cr7, L(retnull)
	mtctr	r5
L(loop1):
	cmpdi	rCHAR1, 0
	sldi	rADDR1, rCHAR1, 2
	sldi	rADDR2, rCHAR2, 2
	lwzx	rLWR1, rLOC, rADDR1
	lwzx	rLWR2, rLOC, rADDR2
	cmpw	cr1, rLWR1, rLWR2
	crorc	4*cr1+eq,eq,4*cr1+eq
	beq	cr1, L(done)
	addi	rSTR1, rSTR1, 1
	addi	rSTR2, rSTR2, 1
	lbz	rCHAR1, 0(rSTR1)
	lbz	rCHAR2, 0(rSTR2)
	bdnz	L(loop1)
#endif
L(done):
	subf	r0, rLWR2, rLWR1
	extsw	rRTN, r0
	blr
END (__STRCASECMP)

weak_alias (__STRCASECMP, STRCASECMP)
libc_hidden_builtin_def (__STRCASECMP)
