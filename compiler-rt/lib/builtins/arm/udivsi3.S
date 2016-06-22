/*===-- udivsi3.S - 32-bit unsigned integer divide ------------------------===//
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===//
 *
 * This file implements the __udivsi3 (32-bit unsigned integer divide)
 * function for the ARM 32-bit architecture.
 *
 *===----------------------------------------------------------------------===*/

#include "../assembly.h"

	.syntax unified
	.text

#if __ARM_ARCH_ISA_THUMB == 2
	.thumb
#endif

	.p2align 2
DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_uidiv, __udivsi3)

@ unsigned int __udivsi3(unsigned int divident, unsigned int divisor)
@   Calculate and return the quotient of the (unsigned) division.

#if __ARM_ARCH_ISA_THUMB == 2
DEFINE_COMPILERRT_THUMB_FUNCTION(__udivsi3)
#else
DEFINE_COMPILERRT_FUNCTION(__udivsi3)
#endif
#if __ARM_ARCH_EXT_IDIV__
	tst     r1, r1
	beq     LOCAL_LABEL(divby0)
	udiv	r0, r0, r1
	bx  	lr
#else
	cmp	r1, #1
	bcc	LOCAL_LABEL(divby0)
	IT(eq)
	JMPc(lr, eq)
	cmp	r0, r1
	ITT(cc)
	movcc	r0, #0
	JMPc(lr, cc)
	/*
	 * Implement division using binary long division algorithm.
	 *
	 * r0 is the numerator, r1 the denominator.
	 *
	 * The code before JMP computes the correct shift I, so that
	 * r0 and (r1 << I) have the highest bit set in the same position.
	 * At the time of JMP, ip := .Ldiv0block - 12 * I.
	 * This depends on the fixed instruction size of block.
	 * For ARM mode, this is 12 Bytes, for THUMB mode 14 Bytes.
	 *
	 * block(shift) implements the test-and-update-quotient core.
	 * It assumes (r0 << shift) can be computed without overflow and
	 * that (r0 << shift) < 2 * r1. The quotient is stored in r3.
	 */

#  ifdef __ARM_FEATURE_CLZ
	clz	ip, r0
	clz	r3, r1
	/* r0 >= r1 implies clz(r0) <= clz(r1), so ip <= r3. */
	sub	r3, r3, ip
#    if __ARM_ARCH_ISA_THUMB == 2
	adr	ip, LOCAL_LABEL(div0block) + 1
	sub	ip, ip, r3, lsl #1
#    else
	adr	ip, LOCAL_LABEL(div0block)
#    endif
	sub	ip, ip, r3, lsl #2
	sub	ip, ip, r3, lsl #3
	mov	r3, #0
	bx	ip
#  else
#    if __ARM_ARCH_ISA_THUMB == 2
#    error THUMB mode requires CLZ or UDIV
#    endif
	mov	r2, r0
	adr	ip, LOCAL_LABEL(div0block)

	lsr	r3, r2, #16
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, ip, #(16 * 12)

	lsr	r3, r2, #8
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, ip, #(8 * 12)

	lsr	r3, r2, #4
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, #(4 * 12)

	lsr	r3, r2, #2
	cmp	r3, r1
	movhs	r2, r3
	subhs	ip, ip, #(2 * 12)

	/* Last block, no need to update r2 or r3. */
	cmp	r1, r2, lsr #1
	subls	ip, ip, #(1 * 12)

	mov	r3, #0

	JMP(ip)
#  endif

#define	IMM	#

#define block(shift)                                                           \
	cmp	r0, r1, lsl IMM shift;                                         \
	ITT(hs);                                                               \
	WIDE(addhs)	r3, r3, IMM (1 << shift);                              \
	WIDE(subhs)	r0, r0, r1, lsl IMM shift

	block(31)
	block(30)
	block(29)
	block(28)
	block(27)
	block(26)
	block(25)
	block(24)
	block(23)
	block(22)
	block(21)
	block(20)
	block(19)
	block(18)
	block(17)
	block(16)
	block(15)
	block(14)
	block(13)
	block(12)
	block(11)
	block(10)
	block(9)
	block(8)
	block(7)
	block(6)
	block(5)
	block(4)
	block(3)
	block(2)
	block(1)
LOCAL_LABEL(div0block):
	block(0)

	mov	r0, r3
	JMP(lr)
#endif /* __ARM_ARCH_EXT_IDIV__ */

LOCAL_LABEL(divby0):
	mov	r0, #0
#ifdef __ARM_EABI__
	b	__aeabi_idiv0
#else
	JMP(lr)
#endif

END_COMPILERRT_FUNCTION(__udivsi3)

NO_EXEC_STACK_DIRECTIVE

