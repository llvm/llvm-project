/* strcmp implementation for ARMv7-A, optimized for Cortex-A15.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <arm-features.h>
#include <sysdep.h>

/* Implementation of strcmp for ARMv7 when DSP instructions are
   available.  Use ldrd to support wider loads, provided the data
   is sufficiently aligned.  Use saturating arithmetic to optimize
   the compares.  */

/* Build Options:
   STRCMP_PRECHECK: Run a quick pre-check of the first byte in the
   string.  If comparing completely random strings the pre-check will
   save time, since there is a very high probability of a mismatch in
   the first character: we save significant overhead if this is the
   common case.  However, if strings are likely to be identical (e.g.
   because we're verifying a hit in a hash table), then this check
   is largely redundant.  */

#define STRCMP_PRECHECK	1

	.syntax unified

#ifdef __ARM_BIG_ENDIAN
# define S2LO lsl
# define S2LOEQ lsleq
# define S2HI lsr
# define MSB 0x000000ff
# define LSB 0xff000000
# define BYTE0_OFFSET 24
# define BYTE1_OFFSET 16
# define BYTE2_OFFSET 8
# define BYTE3_OFFSET 0
#else /* not  __ARM_BIG_ENDIAN */
# define S2LO lsr
# define S2LOEQ lsreq
# define S2HI lsl
# define BYTE0_OFFSET 0
# define BYTE1_OFFSET 8
# define BYTE2_OFFSET 16
# define BYTE3_OFFSET 24
# define MSB 0xff000000
# define LSB 0x000000ff
#endif /* not  __ARM_BIG_ENDIAN */

/* Parameters and result.  */
#define src1		r0
#define src2		r1
#define result		r0	/* Overlaps src1.  */

/* Internal variables.  */
#define tmp1		r4
#define tmp2		r5
#define const_m1	r12

/* Additional internal variables for 64-bit aligned data.  */
#define data1a		r2
#define data1b		r3
#define data2a		r6
#define data2b		r7
#define syndrome_a	tmp1
#define syndrome_b	tmp2

/* Additional internal variables for 32-bit aligned data.  */
#define data1		r2
#define data2		r3
#define syndrome	tmp2


	.thumb

/* In Thumb code we can't use MVN with a register shift, but we do have ORN.  */
.macro prepare_mask mask_reg, nbits_reg
	S2HI \mask_reg, const_m1, \nbits_reg
.endm
.macro apply_mask data_reg, mask_reg
	orn \data_reg, \data_reg, \mask_reg
.endm

	/* Macro to compute and return the result value for word-aligned
	   cases.  */
	.macro strcmp_epilogue_aligned synd d1 d2 restore_r6
#ifdef __ARM_BIG_ENDIAN
	/* If data1 contains a zero byte, then syndrome will contain a 1 in
	   bit 7 of that byte.  Otherwise, the highest set bit in the
	   syndrome will highlight the first different bit.  It is therefore
	   sufficient to extract the eight bits starting with the syndrome
	   bit.  */
	clz	tmp1, \synd
	lsl	r1, \d2, tmp1
	.if \restore_r6
	ldrd	r6, r7, [sp, #8]
	.endif
	lsl	\d1, \d1, tmp1
	lsr	result, \d1, #24
	ldrd	r4, r5, [sp], #16
	cfi_remember_state
	cfi_def_cfa_offset (0)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)
	sub	result, result, r1, lsr #24
	bx	lr
#else
	/* To use the big-endian trick we'd have to reverse all three words.
	   that's slower than this approach.  */
	rev	\synd, \synd
	clz	tmp1, \synd
	bic	tmp1, tmp1, #7
	lsr	r1, \d2, tmp1
	.if \restore_r6
	ldrd	r6, r7, [sp, #8]
	.endif
	lsr	\d1, \d1, tmp1
	and	result, \d1, #255
	and	r1, r1, #255
	ldrd	r4, r5, [sp], #16
	cfi_remember_state
	cfi_def_cfa_offset (0)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)
	sub	result, result, r1

	bx	lr
#endif
	.endm

	.text
	.p2align	5
.Lstrcmp_start_addr:
#if STRCMP_PRECHECK == 1
.Lfastpath_exit:
	sub	r0, r2, r3
	bx	lr
	nop
#endif
ENTRY (strcmp)
#if STRCMP_PRECHECK == 1
	ldrb	r2, [src1]
	ldrb	r3, [src2]
	cmp	r2, #1
	it	cs
	cmpcs	r2, r3
	bne	.Lfastpath_exit
#endif
	strd	r4, r5, [sp, #-16]!
	cfi_def_cfa_offset (16)
	cfi_offset (r4, -16)
	cfi_offset (r5, -12)
	orr	tmp1, src1, src2
	strd	r6, r7, [sp, #8]
	cfi_offset (r6, -8)
	cfi_offset (r7, -4)
	mvn	const_m1, #0
	lsl	r2, tmp1, #29
	cbz	r2, .Lloop_aligned8

.Lnot_aligned:
	eor	tmp1, src1, src2
	tst	tmp1, #7
	bne	.Lmisaligned8

	/* Deal with mutual misalignment by aligning downwards and then
	   masking off the unwanted loaded data to prevent a difference.  */
	and	tmp1, src1, #7
	bic	src1, src1, #7
	and	tmp2, tmp1, #3
	bic	src2, src2, #7
	lsl	tmp2, tmp2, #3	/* Bytes -> bits.  */
	ldrd	data1a, data1b, [src1], #16
	tst	tmp1, #4
	ldrd	data2a, data2b, [src2], #16
	prepare_mask tmp1, tmp2
	apply_mask data1a, tmp1
	apply_mask data2a, tmp1
	beq	.Lstart_realigned8
	apply_mask data1b, tmp1
	mov	data1a, const_m1
	apply_mask data2b, tmp1
	mov	data2a, const_m1
	b	.Lstart_realigned8

	/* Unwind the inner loop by a factor of 2, giving 16 bytes per
	   pass.  */
	.p2align 5,,12  /* Don't start in the tail bytes of a cache line.  */
	.p2align 2	/* Always word aligned.  */
.Lloop_aligned8:
	ldrd	data1a, data1b, [src1], #16
	ldrd	data2a, data2b, [src2], #16
.Lstart_realigned8:
	uadd8	syndrome_b, data1a, const_m1	/* Only want GE bits,  */
	eor	syndrome_a, data1a, data2a
	sel	syndrome_a, syndrome_a, const_m1
	cbnz	syndrome_a, .Ldiff_in_a
	uadd8	syndrome_b, data1b, const_m1	/* Only want GE bits.  */
	eor	syndrome_b, data1b, data2b
	sel	syndrome_b, syndrome_b, const_m1
	cbnz	syndrome_b, .Ldiff_in_b

	ldrd	data1a, data1b, [src1, #-8]
	ldrd	data2a, data2b, [src2, #-8]
	uadd8	syndrome_b, data1a, const_m1	/* Only want GE bits,  */
	eor	syndrome_a, data1a, data2a
	sel	syndrome_a, syndrome_a, const_m1
	uadd8	syndrome_b, data1b, const_m1	/* Only want GE bits.  */
	eor	syndrome_b, data1b, data2b
	sel	syndrome_b, syndrome_b, const_m1
	/* Can't use CBZ for backwards branch.  */
	orrs	syndrome_b, syndrome_b, syndrome_a /* Only need if s_a == 0 */
	beq	.Lloop_aligned8

.Ldiff_found:
	cbnz	syndrome_a, .Ldiff_in_a

.Ldiff_in_b:
	strcmp_epilogue_aligned syndrome_b, data1b, data2b 1

.Ldiff_in_a:
	cfi_restore_state
	strcmp_epilogue_aligned syndrome_a, data1a, data2a 1

	cfi_restore_state
.Lmisaligned8:
	tst	tmp1, #3
	bne	.Lmisaligned4
	ands	tmp1, src1, #3
	bne	.Lmutual_align4

	/* Unrolled by a factor of 2, to reduce the number of post-increment
	   operations.  */
.Lloop_aligned4:
	ldr	data1, [src1], #8
	ldr	data2, [src2], #8
.Lstart_realigned4:
	uadd8	syndrome, data1, const_m1	/* Only need GE bits.  */
	eor	syndrome, data1, data2
	sel	syndrome, syndrome, const_m1
	cbnz	syndrome, .Laligned4_done
	ldr	data1, [src1, #-4]
	ldr	data2, [src2, #-4]
	uadd8	syndrome, data1, const_m1
	eor	syndrome, data1, data2
	sel	syndrome, syndrome, const_m1
	cmp	syndrome, #0
	beq	.Lloop_aligned4

.Laligned4_done:
	strcmp_epilogue_aligned syndrome, data1, data2, 0

.Lmutual_align4:
	cfi_restore_state
	/* Deal with mutual misalignment by aligning downwards and then
	   masking off the unwanted loaded data to prevent a difference.  */
	lsl	tmp1, tmp1, #3	/* Bytes -> bits.  */
	bic	src1, src1, #3
	ldr	data1, [src1], #8
	bic	src2, src2, #3
	ldr	data2, [src2], #8

	prepare_mask tmp1, tmp1
	apply_mask data1, tmp1
	apply_mask data2, tmp1
	b	.Lstart_realigned4

.Lmisaligned4:
	ands	tmp1, src1, #3
	beq	.Lsrc1_aligned
	sub	src2, src2, tmp1
	bic	src1, src1, #3
	lsls	tmp1, tmp1, #31
	ldr	data1, [src1], #4
	beq	.Laligned_m2
	bcs	.Laligned_m1

#if STRCMP_PRECHECK == 0
	ldrb	data2, [src2, #1]
	uxtb	tmp1, data1, ror #BYTE1_OFFSET
	subs	tmp1, tmp1, data2
	bne	.Lmisaligned_exit
	cbz	data2, .Lmisaligned_exit

.Laligned_m2:
	ldrb	data2, [src2, #2]
	uxtb	tmp1, data1, ror #BYTE2_OFFSET
	subs	tmp1, tmp1, data2
	bne	.Lmisaligned_exit
	cbz	data2, .Lmisaligned_exit

.Laligned_m1:
	ldrb	data2, [src2, #3]
	uxtb	tmp1, data1, ror #BYTE3_OFFSET
	subs	tmp1, tmp1, data2
	bne	.Lmisaligned_exit
	add	src2, src2, #4
	cbnz	data2, .Lsrc1_aligned
#else  /* STRCMP_PRECHECK */
	/* If we've done the pre-check, then we don't need to check the
	   first byte again here.  */
	ldrb	data2, [src2, #2]
	uxtb	tmp1, data1, ror #BYTE2_OFFSET
	subs	tmp1, tmp1, data2
	bne	.Lmisaligned_exit
	cbz	data2, .Lmisaligned_exit

.Laligned_m2:
	ldrb	data2, [src2, #3]
	uxtb	tmp1, data1, ror #BYTE3_OFFSET
	subs	tmp1, tmp1, data2
	bne	.Lmisaligned_exit
	cbnz	data2, .Laligned_m1
#endif

.Lmisaligned_exit:
	mov	result, tmp1
	ldr	r4, [sp], #16
	cfi_remember_state
	cfi_def_cfa_offset (0)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)
	bx	lr

#if STRCMP_PRECHECK == 1
.Laligned_m1:
	add	src2, src2, #4
#endif
.Lsrc1_aligned:
	cfi_restore_state
	/* src1 is word aligned, but src2 has no common alignment
	   with it.  */
	ldr	data1, [src1], #4
	lsls	tmp1, src2, #31		/* C=src2[1], Z=src2[0].  */

	bic	src2, src2, #3
	ldr	data2, [src2], #4
	bhi	.Loverlap1		/* C=1, Z=0 => src2[1:0] = 0b11.  */
	bcs	.Loverlap2		/* C=1, Z=1 => src2[1:0] = 0b10.  */

	/* (overlap3) C=0, Z=0 => src2[1:0] = 0b01.  */
.Loverlap3:
	bic	tmp1, data1, #MSB
	uadd8	syndrome, data1, const_m1
	eors	syndrome, tmp1, data2, S2LO #8
	sel	syndrome, syndrome, const_m1
	bne	4f
	cbnz	syndrome, 5f
	ldr	data2, [src2], #4
	eor	tmp1, tmp1, data1
	cmp	tmp1, data2, S2HI #24
	bne	6f
	ldr	data1, [src1], #4
	b	.Loverlap3
4:
	S2LO	data2, data2, #8
	b	.Lstrcmp_tail

5:
	bics	syndrome, syndrome, #MSB
	bne	.Lstrcmp_done_equal

	/* We can only get here if the MSB of data1 contains 0, so
	   fast-path the exit.  */
	ldrb	result, [src2]
	ldrd	r4, r5, [sp], #16
	cfi_remember_state
	cfi_def_cfa_offset (0)
	cfi_restore (r4)
	cfi_restore (r5)
	/* R6/7 Not used in this sequence.  */
	cfi_restore (r6)
	cfi_restore (r7)
	neg	result, result
	bx	lr

6:
	cfi_restore_state
	S2LO	data1, data1, #24
	and	data2, data2, #LSB
	b	.Lstrcmp_tail

	.p2align 5,,12	/* Ensure at least 3 instructions in cache line.  */
.Loverlap2:
	and	tmp1, data1, const_m1, S2LO #16
	uadd8	syndrome, data1, const_m1
	eors	syndrome, tmp1, data2, S2LO #16
	sel	syndrome, syndrome, const_m1
	bne	4f
	cbnz	syndrome, 5f
	ldr	data2, [src2], #4
	eor	tmp1, tmp1, data1
	cmp	tmp1, data2, S2HI #16
	bne	6f
	ldr	data1, [src1], #4
	b	.Loverlap2
4:
	S2LO	data2, data2, #16
	b	.Lstrcmp_tail
5:
	ands	syndrome, syndrome, const_m1, S2LO #16
	bne	.Lstrcmp_done_equal

	ldrh	data2, [src2]
	S2LO	data1, data1, #16
#ifdef __ARM_BIG_ENDIAN
	lsl	data2, data2, #16
#endif
	b	.Lstrcmp_tail

6:
	S2LO	data1, data1, #16
	and	data2, data2, const_m1, S2LO #16
	b	.Lstrcmp_tail

	.p2align 5,,12	/* Ensure at least 3 instructions in cache line.  */
.Loverlap1:
	and	tmp1, data1, #LSB
	uadd8	syndrome, data1, const_m1
	eors	syndrome, tmp1, data2, S2LO #24
	sel	syndrome, syndrome, const_m1
	bne	4f
	cbnz	syndrome, 5f
	ldr	data2, [src2], #4
	eor	tmp1, tmp1, data1
	cmp	tmp1, data2, S2HI #8
	bne	6f
	ldr	data1, [src1], #4
	b	.Loverlap1
4:
	S2LO	data2, data2, #24
	b	.Lstrcmp_tail
5:
	tst	syndrome, #LSB
	bne	.Lstrcmp_done_equal
	ldr	data2, [src2]
6:
	S2LO	data1, data1, #8
	bic	data2, data2, #MSB
	b	.Lstrcmp_tail

.Lstrcmp_done_equal:
	mov	result, #0
	ldrd	r4, r5, [sp], #16
	cfi_remember_state
	cfi_def_cfa_offset (0)
	cfi_restore (r4)
	cfi_restore (r5)
	/* R6/7 not used in this sequence.  */
	cfi_restore (r6)
	cfi_restore (r7)
	bx	lr

.Lstrcmp_tail:
	cfi_restore_state
#ifndef __ARM_BIG_ENDIAN
	rev	data1, data1
	rev	data2, data2
	/* Now everything looks big-endian...  */
#endif
	uadd8	tmp1, data1, const_m1
	eor	tmp1, data1, data2
	sel	syndrome, tmp1, const_m1
	clz	tmp1, syndrome
	lsl	data1, data1, tmp1
	lsl	data2, data2, tmp1
	lsr	result, data1, #24
	ldrd	r4, r5, [sp], #16
	cfi_def_cfa_offset (0)
	cfi_restore (r4)
	cfi_restore (r5)
	/* R6/7 not used in this sequence.  */
	cfi_restore (r6)
	cfi_restore (r7)
	sub	result, result, data2, lsr #24
	bx	lr
END (strcmp)
libc_hidden_builtin_def (strcmp)
